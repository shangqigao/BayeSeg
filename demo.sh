#!/bin/sh

#---------------------------------------BayeSeg--------------------------------------#
# train BayeSeg
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --batch_size 8 --output_dir logs/BayeSeg1-6-8-100 --device cuda >train.log2 2>&1 &

# test Unet
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model Unet --eval --resume logs/Unet_zk/best_checkpoint.pth --output_dir results/MSCMR_dataset/test/Unet/T2 --device cuda --GPU_ids 3

# test PUnet
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model PUnet --eval --resume logs/PUnet/best_checkpoint.pth --output_dir results/MSCMR_dataset/test/PUnet/LGE --device cuda --GPU_ids 3

# test baseline
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model BayeSeg --eval --resume logs/baseline/best_checkpoint.pth --output_dir results/MSCMR_dataset/test/baseline/T2 --device cuda --GPU_ids 3

# test BayeSeg
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model BayeSeg --eval --resume logs/BayeSeg1-6-8-100/checkpoint1799.pth --output_dir results/ACDC/test/BayeSeg --device cuda --GPU_ids 3

#CUDA_VISIBLE_DEVICES=3 python -u main.py --model BayeSeg --eval --resume logs/BayeSeg1-6-8-100/checkpoint1799.pth --output_dir results/visualization/T2 --device cuda --GPU_ids 3
