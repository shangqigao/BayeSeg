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
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model Unet --eval --dataset MSCMR --sequence LGR --resume logs/Unet/checkpoint.pth --output_dir results --device cuda

# test PUnet
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model PUnet --eval --dataset MSCMR --sequence LGR --resume logs/PUnet/checkpoint.pth --output_dir results --device cuda

# test baseline
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model BayeSeg --eval --dataset MSCMR --sequence LGR --resume logs/Baseline/checkpoint.pth --output_dir results --device cuda

# test BayeSeg
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model BayeSeg --eval --dataset MSCMR --sequence LGR --resume logs/BayeSeg/checkpoint.pth --output_dir results --device cuda
