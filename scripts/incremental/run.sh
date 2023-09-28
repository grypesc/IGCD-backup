#!/bin/bash

set -e
set -x

python -m model.simgcd_icarl \
    --dataset_name 'inat21_incremental' \
    --batch_size 512 \
    --grad_from_block 9 \
    --epochs 20 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2i' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 3 \
    --memax_weight 2 \
    --use_small_set \
    --split_crit 'loc_year' \
    --runner_name 'incremental' --return_path \
    --exp_name inat21_incre_simgcd_debug_icarl \
    --exp_id inat21_incre_simgcd_icarl1 \
    --tags 'incremental-inat21-debug-icarl'
