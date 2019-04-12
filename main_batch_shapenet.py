import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import cv2

import json
import sys
from data_generator.shapenet_data_generator import DataGenerator
from dataset import iDataset

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ShapeNet Batch Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-ncl', '--num_classes', default=20,
                    help='number of classes in the classification task')
parser.add_argument('-ckpt', '--checkpoint_file', default='checkpoint',
					help='checkpoint name to save to')

parser.add_argument('--no_jitter', dest='jitter', action='store_false',
                    help='Option for no color jittering (for iCaRL)')
parser.add_argument('--h_ch', default=0.02, type=float,
                    help='Color jittering : max hue change')
parser.add_argument('--s_ch', default=0.05, type=float,
                    help='Color jittering : max saturation change')
parser.add_argument('--l_ch', default=0.1, type=float,
                    help='Color jittering : max lightness change')

parser.add_argument('--img_size', default=224, type=int,
                    help='Size of images input to the network')
parser.add_argument('--rendered_img_size', default=300, type=int,
                    help='Size of rendered images')
parser.add_argument('--lexp_len', default=100, type=int,
                    help='Number of frames in Learning Exposure')
parser.add_argument('--size_test', default=100, type=int,
                    help='Number of test images per object')

parser.add_argument('--num_instance_per_class', default=25, type=int, 
                    help='Number of instances per class for training')
parser.add_argument('--num_test_instance_per_class', default=15, type=int, 
                    help='Number of test instances per class')
parser.add_argument('--num_val_instance_per_class', default=10, type=int, 
                    help='Number of val instances per class')

parser.set_defaults(jitter=True)


def main():
    global args
    args = parser.parse_args()
    best_acc1 = 0

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

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, 
                                init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, args.num_classes, bias=False)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    #loading mean image; resizing to rendered image size if necessary
    mean_image = np.load('data_generator/shapenet_mean_image.npy')
    #flip mean image BGR to RGB
    mean_image = mean_image[:, :, ::-1]
    mean_image.astype(np.uint8)
    mean_image = cv2.resize(mean_image, (args.rendered_img_size, 
                                         args.rendered_img_size))


    train_instances = {}
    val_instances = {}
    test_instances = {}

    with open('data_generator/shapenet_train_instances.json', 'r') as tm_file:
        train_instances = json.load(tm_file)
        for cl in train_instances:
            tmp_list = []
            for synset, modelID in train_instances[cl]:
                tmp_list.append(modelID)
            
            val_instances[cl] = np.random.choice(tmp_list, 
                                                 args.num_val_instance_per_class,
                                                 replace=False)
            
            train_instances[cl] = np.random.choice(list(set(tmp_list) 
                                                        - set(val_instances[cl])), 
                                                   args.num_instance_per_class,
                                                   replace=False)

    with open('data_generator/shapenet_test_instances.json') as tm_file:
        test_instances = json.load(tm_file)
        for cl in test_instances:
            tmp_list = []
            for synset, modelID in test_instances[cl]:
                tmp_list.append(modelID)
            test_instances[cl] = tmp_list

    classes = [cl for cl in train_instances]
    classes.sort() # So the order doesn't change and hence neither do the labels
    class_map = {cl:i for i,cl in enumerate(classes)}
    
    train_dgs = [[DataGenerator(category_name=cl, 
                                instance_name=instance, 
                                n_frames=args.lexp_len, 
                                size_test=args.size_test,
                                resolution=args.rendered_img_size, 
                                job='train') 
                  for instance in train_instances[cl]] 
                  for cl in classes]

    val_dgs = [[DataGenerator(category_name=cl, 
                              instance_name=instance, 
                              n_frames=args.lexp_len, 
                              size_test=args.size_test,
                              resolution=args.rendered_img_size, 
                              job='train') 
                  for instance in val_instances[cl]] 
                  for cl in classes]

    test_dgs = [[DataGenerator(category_name=cl, 
                               instance_name=instance, 
                               n_frames=args.lexp_len, 
                               size_test=args.size_test,
                               resolution=args.rendered_img_size, 
                               job='test') 
                  for instance in test_instances[cl]] 
                  for cl in classes]
    
    max_train_data_size = (2 * args.num_classes 
                           * args.lexp_len 
                           * args.num_instance_per_class)
    max_val_data_size = (args.num_classes 
                         * args.lexp_len 
                         * args.num_val_instance_per_class)
    max_test_data_size = (args.num_classes 
                          * args.size_test 
                          * args.num_test_instance_per_class)
    
    # Correct settings for initializing iDataset
    args.algo = 'icarl'
    args.jitter = True

    if args.evaluate:
        print('Loading Test Data')
        val_set = iDataset(args, mean_image, test_dgs, max_test_data_size, 
                           classes, class_map, 'test')
    else:
        print('Loading Training Data')
        train_set = iDataset(args, mean_image, train_dgs, max_train_data_size, 
                             classes, class_map, 'batch_train')
        print('Loading Val Data')
        val_set = iDataset(args, mean_image, val_dgs, max_val_data_size, 
                           classes, class_map, 'batch_val')

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.checkpoint_file)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (indices, input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (indices, input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5 = top5))

        print(' * Acc@1 {top1.avg:.3f} '
              .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint'):
    torch.save(state, filename+'.pth.tar')
    if is_best:
        shutil.copyfile(filename, filename+'-model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()