
import argparse
import warnings
import os
import random
import numpy as np
import time
import datetime
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
import itertools

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from data.dataloader import Dataloader
from models.ProtoNet import ProtoNet
from utils.loss import prototypical_loss
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=100, help='Number of max epochs in training (default: 100)')
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--workers', type=int, default=8, help='Number of workers in dataset loader (default: 4)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size in training (default: 32)')
parser.add_argument('--lr', default=0.001)
parser.add_argument('--lr-scheduler-step', type=float, default=20)
# episode
parser.add_argument('--iterations', type=int, default=100, help='number of episodes per epoch[default=100]')
parser.add_argument('--classes-per-it-train', type=int, default=60, help='number of random classes per episode for training')
parser.add_argument('--num-support-train', type=int, default=5, help='number of samples per class to use as support')
parser.add_argument('--num-query-train', type=int, default=5)

parser.add_argument('--classes-per-it-valid', type=int, default=5, help='number of random classes per episode for training')
parser.add_argument('--num-support-valid', type=int, default=5, help='number of samples per class to use as support')
parser.add_argument('--num-query-valid', type=int, default=15)
# Path
parser.add_argument('--dataset-path', type=str, default='..' + os.sep + 'dataset')
parser.add_argument('--log-path', type=str, default='..' + os.sep + 'output')

parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")
parser.add_argument('--evaluate', '-e', default=False, action='store_true')
# Distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

os.makedirs('./saved_models', exist_ok=True) # save Model.pth
# os.makedirs('./output', exist_ok=True) # save log

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # STFT 인자
    summary = SummaryWriter()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Models & DataParallel Setting
    model = ProtoNet()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            print("this")
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    else:
        model = torch.nn.DataParallel(model).cuda()

    # Criterion & Optimizer & Scheduler
    criterion = prototypical_loss
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                gamma=0.5,
                                                step_size=args.lr_scheduler_step)

    # Load Model if you have
    best_acc = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is not None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']

            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # dataset & dataloader
    train_dataloader = Dataloader(args=args, mode='train') # Dataset 포함
    valid_dataloader = Dataloader(args=args, mode='val')
    test_dataloader = Dataloader(args=args, mode='test')

    if args.evaluate:
        test(args, test_dataloader, model, criterion)
        return

    # Train & Test
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for epoch in range(args.start_epoch, args.epochs):

        train(args, train_dataloader, model, optimizer, scheduler, criterion, summary, epoch,
              train_loss, train_acc)

        avg_loss, avg_acc = validate(args, valid_dataloader, model, criterion, summary, epoch, val_loss, val_acc)

        if avg_acc >= best_acc:
            print("Found better validated model")
            best_acc = avg_acc

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                torch.save({
                    'epoch': epoch+1,
                    'best_acc': best_acc,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, "saved_models/checkpoint_%d.pth" % (epoch+1))


def train(args, train_dataloader, model, optimizer, scheduler, criterion, summary, epoch,
          train_loss, train_acc):
    model.train()
    tr_iter = iter(train_dataloader)
    end = time.time()
    idx = 0

    for batch in tqdm(tr_iter): # todo batch = way5 * shot5 = 25
        optimizer.zero_grad()
        x, y = batch
        x, y = x.cuda(args.gpu, non_blocking=True), y.cuda(args.gpu, non_blocking=True)

        outputs = model(x)
        loss, acc = criterion(input=outputs, target=y,
                              n_support=args.num_support_train)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_acc.append(acc.item())

        niter = epoch * len(train_dataloader) + idx
        summary.add_scalar('Train/Loss', loss.item(), niter)
        summary.add_scalar('Train/acc', acc.item(), niter)

        # if idx % args.print_freq == 0:
        #     print("Loss: %f, acc: %f" %(loss, acc))

    avg_loss = np.mean(train_loss[-args.iterations:])
    avg_acc = np.mean(train_acc[-args.iterations:])
    print("Time Per epoch: ", datetime.timedelta(seconds=time.time() - end))
    print('Epoch {} | Avg Train Loss: {} | Avg Train Acc: {}'.format(epoch, avg_loss, avg_acc))
    scheduler.step()


def validate(args, valid_dataloader, model, criterion, summary, epoch,
             val_loss, val_acc):
    model.eval()
    val_iter = iter(valid_dataloader)
    idx = 0

    for batch in tqdm(val_iter):
        x, y = batch
        x, y = x.cuda(args.gpu, non_blocking=True), y.cuda(args.gpu, non_blocking=True)
        outputs = model(x)
        loss, acc = criterion(input=outputs, target=y,
                              n_support=args.num_support_valid)
        val_loss.append(loss.item())
        val_acc.append(acc.item())

        niter = epoch * len(valid_dataloader) + idx
        summary.add_scalar('Valid/Loss', loss.item(), niter)
        summary.add_scalar('Valid/acc', acc.item(), niter)

    avg_loss = np.mean(val_loss[-args.iterations:])
    avg_acc = np.mean(val_acc[-args.iterations:])
    print('Epoch {} | Avg Valid Loss: {} | Avg Valid Acc: {}'.format(epoch, avg_loss, avg_acc))

    return avg_loss, avg_acc


def test(args, test_dataloader, model, criterion):
    model.eval()
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.cuda(args.gpu, non_blocking=True), y.cuda(non_blocking=True)
            outputs = model(x)
            _, acc = criterion(outputs, target=y,
                               n_support=args.num_support_valid)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc:{}'.format(avg_acc))


if __name__ == '__main__':
    main()
