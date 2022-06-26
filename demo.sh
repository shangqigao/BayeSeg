#!/bin/sh

#---------------------------------------BayeSeg--------------------------------------#
# train Unet
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --model Unet --batch_size 8 --output_dir logs/Unet --device cuda >train.log 2>&1 &

# train PUnet
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --model PUnet --batch_size 8 --output_dir logs/PUnet --device cuda >train.log 2>&1 &

# train Baseline
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --model BayeSeg --batch_size 8 --output_dir logs/Baseline --device cuda >train.log 2>&1 &

# train BayeSeg
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --model BayeSeg --batch_size 8 --output_dir logs/BayeSeg --device cuda >train.log 2>&1 &

# test Unet
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model Unet --eval --dataset MSCMR --sequence LGR --resume logs/Unet_zk/best_checkpoint.pth --output_dir results/MSCMR_dataset/test/Unet/T2 --device cuda --GPU_ids 3

# test PUnet
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model PUnet --eval --dataset MSCMR --sequence LGR --resume logs/PUnet/best_checkpoint.pth --output_dir results/MSCMR_dataset/test/PUnet/LGE --device cuda --GPU_ids 3

# test baseline
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model BayeSeg --eval --dataset MSCMR --sequence LGR --resume logs/baseline/best_checkpoint.pth --output_dir results/MSCMR_dataset/test/baseline/T2 --device cuda --GPU_ids 3

# test BayeSeg
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model BayeSeg --eval --dataset MSCMR --sequence LGR --resume logs/BayeSeg1-6-8-100/checkpoint1799.pth --output_dir results/ACDC/test/BayeSeg --device cuda --GPU_ids 3
