#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : train_drink.py
# @time    : 10/14/20 9:07 AM
# @desc    :
'''

import argparse
import itertools
import logging
import os
import sys
from tqdm import tqdm

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader

from vision.datasets.txt_dataset import TXTDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config.fd_config import define_img_size
from vision.utils.misc import str2bool, Timer
from vision.ssd.config import fd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.ssd.ssd import MatchPrior

parser = argparse.ArgumentParser(description='train With Pytorch')
parser.add_argument('--datasets', nargs='+', default= "/home/zhex/data/electronic_tag", help='Dataset directory path')
parser.add_argument('--label_path', default= "/home/zhex/data/electronic_tag/labels.txt", help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")

parser.add_argument('--net', default="slim",
                    help="The network architecture ,optional(RFB , slim)")
# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default="models_tag/slim-Epoch-48-Loss-1.727860439208246.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=96, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=100, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=1, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models_tag/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--log_dir', default='./models_tag/logs',
                    help='log dir')
parser.add_argument('--cuda_index', default="0", type=str,
                    help='Choose cuda index.If you have 4 GPUs, you can set it like 0,1,2,3')
parser.add_argument('--power', default=2, type=int,help='poly lr pow')
parser.add_argument('--overlap_threshold', default=0.35, type=float,help='overlap_threshold')
parser.add_argument('--optimizer_type', default="Adam", type=str,help='optimizer_type')
parser.add_argument('--input_size', default=320, type=int,
                    help='define network input size,default optional value 128/160/320/384/480/640/1280')
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

input_img_size = args.input_size  # define input size ,default optional(128/160/320/384/480/640/1280)
logging.info("input size :{}".format(input_img_size))

define_img_size(input_img_size)  # must put define_img_size() before 'import fd_config'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


def lr_poly(base_lr, iter):
    return base_lr * ((1 - float(iter) / args.num_epochs) ** (args.power))


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.lr, i_iter)
    optimizer.param_groups[0]['lr'] = lr


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            print(".", flush=True)
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )

            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()

    logging.info(args)
    if args.net == 'slim':
        create_net = create_mb_tiny_fd
        config = fd_config
    elif args.net == 'RFB':
        create_net = create_Mb_Tiny_RFB_fd
        config = fd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, args.overlap_threshold)
    test_transform = TestTransform(config.image_size, config.image_mean_test, config.image_std)

    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    train_dataset = TXTDataset(args.datasets, is_train=True, transform=train_transform,target_transform=target_transform)
    val_dataset = TXTDataset(args.datasets, is_train=False, transform=test_transform,target_transform=target_transform,)
    train_loader = DataLoader(train_dataset, args.batch_size,num_workers=args.num_workers,shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size,num_workers=args.num_workers,shuffle=False)
    num_classes = 2
    # create net
    net = create_net(num_classes)

    if torch.cuda.device_count() >= 1:
        cuda_index_list = [int(v.strip()) for v in args.cuda_index.split(",")]
        net = nn.DataParallel(net, device_ids=cuda_index_list)
        logging.info("use gpu :{}".format(cuda_index_list))

    min_loss = -10000.0
    last_epoch = 49 #-1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr

    params = [
        {'params': net.module.base_net.parameters(), 'lr': base_net_lr},
        {'params': itertools.chain(
            net.module.source_layer_add_ons.parameters(),
            net.module.extras.parameters()
        ), 'lr': extra_layers_lr},
        {'params': itertools.chain(
            net.module.regression_headers.parameters(),
            net.module.classification_headers.parameters()
        )}
    ]
    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.module.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)
    criterion = MultiboxLoss(config.priors, neg_pos_ratio=3,center_variance=0.1, size_variance=0.2, device=DEVICE)
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
        logging.info("use Adam optimizer")
    else:
        logging.fatal(f"Unsupported optimizer: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")
    if args.optimizer_type != "Adam":
        if args.scheduler == 'multi-step':
            logging.info("Uses MultiStepLR scheduler.")
            milestones = [int(v.strip()) for v in args.milestones.split(",")]
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=last_epoch)
        elif args.scheduler == 'cosine':
            logging.info("Uses CosineAnnealingLR scheduler.")
            scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
        elif args.scheduler == 'poly':
            logging.info("Uses PolyLR scheduler.")
        else:
            logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
            parser.print_help(sys.stderr)
            sys.exit(1)

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in tqdm(range(last_epoch + 1, args.num_epochs)):
        if args.optimizer_type != "Adam":
            if args.scheduler != "poly":
                if epoch != 0:
                    scheduler.step()
        train(train_loader, net, criterion, optimizer,device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)

        if args.scheduler == "poly":
            adjust_learning_rate(optimizer, epoch)
        logging.info("lr rate :{}".format(optimizer.param_groups[0]['lr']))

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            logging.info("lr rate :{}".format(optimizer.param_groups[0]['lr']))
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.module.save(model_path)
            logging.info(f"Saved model {model_path}")