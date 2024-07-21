#!/bin/bash

# torchrun --master_port=12345 --nproc_per_node=2 train.py \
# --root /gpu-data2/nyian/vis \
# --use_checkpoint \
# --checkpoint_path runs/checkpoint.pt \
# --model_save_path runs

# torchrun --master_port=12345 --nproc_per_node=2 train.py \
# --root /gpu-data2/nyian/ssv2 \
# --dataset ssv2 \
# --use_checkpoint \
# --checkpoint_path runs/checkpoint.pt \
# --model_save_path runs \


python train_AcE_SOLV.py --root /gpu-data2/nyian/ssv2 \
--dataset ssv2 \
--use_checkpoint \
--checkpoint_path externals/SOLV/runs/checkpoint_epoch_99.pt \
--model_save_path runs \