import os
import argparse
import datetime
import random
import json
import time
from pathlib import Path
from tensorboardX import SummaryWriter
from copy import deepcopy
from inference import infer
from visualize import visual
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader, DistributedSampler
import data
#from mmdet import datasets
import util.misc as utils
from data import build
from engine import evaluate, train_one_epoch
from models import build_model
########################################
from args import add_management_args, add_experiment_args, add_bayes_args
import logging


def main(args):
    # setup logger
    utils.setup_logger('base', args.output_dir, 'train', level=logging.INFO,
                          screen=True, tofile=True)
    logger = logging.getLogger('base')
    
    utils.init_distributed_mode(args)
    writer = SummaryWriter(log_dir=args.output_dir + '/summary')
    logger.info(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors, visualizer = build_model(args)
    model.to(device)
    logger.info(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:{}'.format( n_parameters))

    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    logger.info('Building training dataset...')
    dataset_train_dict = build(image_set='train', args=args)
    num_train = [len(v) for v in dataset_train_dict.values()]
    logger.info('Number of training images: {}'.format(sum(num_train)))

    logger.info('Building validation dataset...')
    dataset_val_dict = build(image_set='val', args=args)
    num_val = [len(v) for v in dataset_val_dict.values()]
    logger.info('Number of validation images: {}'.format(sum(num_val)))

    if args.distributed:
        sampler_train_dict = {k : DistributedSampler(v) for k, v in dataset_train_dict.items()}
        sampler_val_dict = {k: DistributedSampler(v, shuffle=False) for k, v in dataset_val_dict.items()}
    else:
        sampler_train_dict = {k : torch.utils.data.RandomSampler(v) for k, v in dataset_train_dict.items()}
        sampler_val_dict = {k : torch.utils.data.SequentialSampler(v) for k, v in dataset_val_dict.items()}

    batch_sampler_train = { 
        k : torch.utils.data.BatchSampler(v, args.batch_size, drop_last=True) for k, v in sampler_train_dict.items()
        }
    dataloader_train_dict = {
        k : DataLoader(v1, batch_sampler=v2, collate_fn=utils.collate_fn, num_workers=args.num_workers) 
        for (k, v1), v2 in zip(dataset_train_dict.items(), batch_sampler_train.values())
        }
    dataloader_val_dict = {
        k : DataLoader(v1, args.batch_size, sampler=v2, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) 
        for (k, v1), v2 in zip(dataset_val_dict.items(), sampler_val_dict.values())
        }

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.whst.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
        #if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        #test_stats = evaluate(model, criterion, postprocessors, dataloader_val_dict, device, args.output_dir, visualizer, 0, writer)
        test_df = infer(model, args.model, args.dataset, args.sequence, dataloader_val_dict, args.output_dir, device)
        #test_df = visual(model, args.model, dataloader_val_dict, args.output_dir, device)
    else:
        logger.info("Start training")
        best_dic = None
        best_dice = None
        start_time = time.time()
        min_delta=0.1
        patience = 40
        hist_loss = 1e16
        patience_counter = 0
        for epoch in range(args.start_epoch, args.epochs):
            # optimizer.param_groups[0]['lr'] = clr.cyclic_learning_rate(epoch, mode='exp_range', gamma=1)
            train_stats = train_one_epoch(model, criterion, dataloader_train_dict, optimizer, device, epoch,args)
            test_stats = evaluate(model, criterion, postprocessors, dataloader_val_dict, device, args.output_dir, visualizer, epoch, writer)
            #test_df = infer(model, criterion, dataloader_val_dict, device)
            dice_score = test_stats["loss_AvgDice"]
            logger.info('dice score:{}'.format(dice_score))
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                if best_dice == None or dice_score > best_dice:
                    best_dice = dice_score
                    best_dic = deepcopy(test_stats)
                    logger.info("Update best model!")
                    checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
                if dice_score > 0.81:
                    logger.info("Update high dice score model!")
                    file_name = str(dice_score)[0:6]+'new_checkpoint.pth'
                    checkpoint_paths.append(output_dir / file_name)
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BayeSeg training and evaluation', allow_abbrev=False)
    add_experiment_args(parser)
    add_management_args(parser)
    add_bayes_args(parser)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    print(torch.cuda.is_available())
    main(args)
