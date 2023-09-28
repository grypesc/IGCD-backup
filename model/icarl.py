import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import copy
import pandas as pd

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model.loss import info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups



class iCarlNet(nn.Module):
    def __init__(self, feature_extractor, dino_head, args=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.dino_head = dino_head
        self.args = args
        self.n_classes = len(args.classes_in_labelled)
        
        # List containing exemplar_sets
        # each entry is a list of pathes to the images
        """
        [
            [(path1.png, label1), (path2.png, label1), ...],
            [(path1.png, label2), (path2.png, label2), ...],
        ]
        """
        self.exemplar_sets = []

        # Learning method
        self.icarl_cls_loss = nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()

        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

    def forward(self, x, return_x=False):
        feature = self.feature_extractor(x)
        output = self.dino_head(feature, return_x=return_x)
        return output

    def incremental_classes(self, n):
        self.n_classes += n
        
    def classify(self, x, transform):
        batch_size = x.shape[0]
        
        if self.compute_means:
            self.args.logger.info("Computing means of exemplars")
            exemplar_means = []
            for exemplar_set in self.exemplar_sets:
                features = []
                for ex, _ in exemplar_set:
                    ex = transform(Image.open(ex)).cuda()
                    ex = ex.unsqueeze(0)
                    feature = self.feature_extractor(ex)
                    features.append(feature)
                features = torch.cat(features, dim=0)
                features = features / features.norm(dim=-1, keepdim=True)
                mean = features.mean(dim=0)
                mean = mean / mean.norm(dim=-1, keepdim=True)
                exemplar_means.append(mean)
            self.exemplar_means = exemplar_means
            self.compute_means = False
            self.args.logger.info("Done computing means of exemplars")

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means, dim=0)
        means = torch.stack([means] * batch_size, dim=0)
        means = torch.transpose(means, 1, 2)
        
        features = self.feature_extractor(x)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        
        dist = (features - means).pow(2).sum(dim=1).squeeze()
        _, preds = dist.min(dim=1)
        return preds

    @torch.no_grad()
    def construct_exemplar_sets(self, train_loader, n, args):
        # import ipdb; ipdb.set_trace()
        self.args.logger.info("Constructing exemplar sets")
        self.eval()
        preds, labels, mask_labs, paths = [], [], [], []
        feats = []
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            images, class_labels, uq_idxs, path, mask_lab = batch
            paths.append(path)
            images = images[0].cuda(non_blocking=True) # only one view of the image
            features = self.feature_extractor(images)
            features = features / features.norm(dim=-1, keepdim=True)
            _, pred = self.dino_head(features)
            preds.append(pred.argmax(1).cpu().numpy())
            labels.append(class_labels.cpu().numpy())
            mask_labs.append(mask_lab.cpu().numpy())
            feats.append(features.cpu().numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        mask_labs = np.concatenate(mask_labs).astype(bool)
        features = np.concatenate(feats)
        paths = np.concatenate(paths)
        
        labels_used = np.zeros(labels.shape) # when mask_lab is True, use labels, when False, use preds
        labels_used[mask_labs] = labels[mask_labs]
        labels_used[~mask_labs] = preds[~mask_labs]
        labels_used = labels_used.astype(np.int32)
        
        for lbl in np.unique(labels_used):
            feats = features[labels_used == lbl]
            lbl_paths = paths[labels_used == lbl]
            sub_labels_used = labels_used[labels_used == lbl]
            mean_feat = feats.mean(axis=0) # mean of features
            mean_feat = mean_feat / np.linalg.norm(mean_feat) # normalize
            
            exemplar_set = []
            exemplar_features = []
            for k in range(n):
                S = np.sum(exemplar_features, axis=0)
                phi = feats
                mu = mean_feat
                mu_p = 1.0/(k+1) * (phi + S)
                mu_p = mu_p / np.linalg.norm(mu_p)
                i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

                exemplar_set.append(lbl_paths[i])
                exemplar_features.append(feats[i])

            self.exemplar_sets.append([(item, lbl) for item in exemplar_set])
    
    def reduce_exemplar_sets(self, n):
        for idx, exemplar_set in enumerate(self.exemplar_sets):
            self.exemplar_sets[idx] = exemplar_set[:n]
        
    def combine_dataset_with_exemplars(self, train_dataset, n, args):
        # return train_dataset + self.exemplar_sets
        # import ipdb; ipdb.set_trace()
        self.reduce_exemplar_sets(n)
        file_names, category_ids = [], []
        for exemplar_set in self.exemplar_sets:
            for ex, cat in exemplar_set:
                file_names.append(ex)
                category_ids.append(cat)
        new_df = pd.concat([train_dataset.data, pd.DataFrame({'file_name': file_names, 'category_id': category_ids})])
        new_df = new_df.drop_duplicates(subset=['file_name', ], keep='first')
        new_dataset = copy.deepcopy(train_dataset)
        new_dataset.data = new_df
        return new_dataset

