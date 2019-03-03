# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import math
import time
import glob
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from sklearn import metrics
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from util import AverageMeter, load_model
from eval_linear import accuracy


parser = argparse.ArgumentParser()
parser.add_argument('--vocdir', type=str, required=False, default='', help='pascal voc 2007 dataset')
parser.add_argument('--split', type=str, required=False, default='train', choices=['train', 'trainval'], help='training split')
parser.add_argument('--model', type=str, required=False, default='',
                    help='evaluate this model')
parser.add_argument('--nit', type=int, default=80000, help='Number of training iterations')
parser.add_argument('--fc6_8', type=int, default=1, help='If true, train only the final classifier')
parser.add_argument('--train_batchnorm', type=int, default=0, help='If true, train batch-norm layer parameters')
parser.add_argument('--eval_random_crops', type=int, default=1, help='If true, eval on 10 random crops, otherwise eval on 10 fixed crops')
parser.add_argument('--stepsize', type=int, default=5000, help='Decay step')
parser.add_argument('--lr', type=float, required=False, default=0.003, help='learning rate')
parser.add_argument('--wd', type=float, required=False, default=1e-6, help='weight decay')
parser.add_argument('--min_scale', type=float, required=False, default=0.1, help='scale')
parser.add_argument('--max_scale', type=float, required=False, default=0.5, help='scale')
parser.add_argument('--seed', type=int, default=31, help='random seed')

def main():
    args = parser.parse_args()    
    print(args)

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # create model and move it to gpu
    model = load_model(args.model)
    model.top_layer = nn.Linear(model.top_layer.weight.size(1), 20)
    model.cuda()
    cudnn.benchmark = True

    # what partition of the data to use
    if args.split == 'train':
        args.test = 'val'
    elif args.split == 'trainval':
        args.test = 'test'
    # data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = VOC2007_dataset(args.vocdir, split=args.split, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(args.min_scale, args.max_scale), ratio=(1, 1)),
            transforms.ToTensor(),
            normalize,
         ]))

    loader = torch.utils.data.DataLoader(dataset,
         batch_size=16, shuffle=False,
         num_workers=24, pin_memory=True)
    print('PASCAL VOC 2007 ' + args.split + ' dataset loaded')

    # re initialize classifier
    for y, m in enumerate(model.classifier.modules()):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.fill_(0.1)
    model.top_layer.bias.data.fill_(0.1)

    if args.fc6_8:
       # freeze some layers 
        for param in model.features.parameters():
            param.requires_grad = False
        # unfreeze batchnorm scaling
        if args.train_batchnorm:
            for layer in model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    for param in layer.parameters():
                        param.requires_grad = True

    # set optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.wd,
    )

    criterion = nn.BCEWithLogitsLoss(reduction='none')

    print('Start training')
    it = 0
    losses = AverageMeter()
    while it < args.nit:
        it = train(
            loader,
            model,
            optimizer,
            criterion,
            args.fc6_8,
            losses,
            it=it,
            total_iterations=args.nit,
            stepsize=args.stepsize,
        )

    print('Evaluation')
    if args.eval_random_crops:
        transform_eval = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(args.min_scale, args.max_scale), ratio=(1, 1)), 
            transforms.ToTensor(),
            normalize,
        ]
    else:
        transform_eval = [
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))
        ] 

    print('Train set')
    train_dataset = VOC2007_dataset(args.vocdir, split=args.split, transform=transforms.Compose(transform_eval))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=24, 
        pin_memory=True,
    )
    evaluate(train_loader, model, args.eval_random_crops)

    print('Test set')
    test_dataset = VOC2007_dataset(args.vocdir, split=args.test, transform=transforms.Compose(transform_eval))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=24, 
        pin_memory=True,
    )
    evaluate(test_loader, model, args.eval_random_crops)


def evaluate(loader, model, eval_random_crops):
    model.eval()
    gts = []
    scr = []
    for crop in range(9 * eval_random_crops + 1):
        for i, (input, target) in enumerate(loader):
            # move input to gpu and optionally reshape it
            if len(input.size()) == 5:
                bs, ncrops, c, h, w = input.size()
                input = input.view(-1, c, h, w)
            input = input.cuda(non_blocking=True)

            # forward pass without grad computation
            with torch.no_grad():
                output = model(input)
            if crop < 1 :
                    scr.append(torch.sum(output, 0, keepdim=True).cpu().numpy())
                    gts.append(target)
            else:
                    scr[i] += output.cpu().numpy()
    gts = np.concatenate(gts, axis=0).T
    scr = np.concatenate(scr, axis=0).T
    aps = []
    for i in range(20):
        # Subtract eps from score to make AP work for tied scores
        ap = metrics.average_precision_score(gts[i][gts[i]<=1], scr[i][gts[i]<=1]-1e-5*gts[i][gts[i]<=1])
        aps.append( ap )
    print(np.mean(aps), '  ', ' '.join(['%0.2f'%a for a in aps]))


def train(loader, model, optimizer, criterion, fc6_8, losses, it=0, total_iterations=None, stepsize=None, verbose=True):
    # to log
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    current_iteration = it

    # use dropout for the MLP
    model.train()
    # in the batch norms always use global statistics
    model.features.eval()

    for (input, target) in loader:
        # measure data loading time
        data_time.update(time.time() - end)
        
        # adjust learning rate
        if current_iteration != 0 and current_iteration % stepsize == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
                print('iter {0} learning rate is {1}'.format(current_iteration, param_group['lr']))

        # move input to gpu
        input = input.cuda(non_blocking=True)

        # forward pass with or without grad computation
        output = model(input)

        target = target.float().cuda()
        mask = (target == 255)
        loss = torch.sum(criterion(output, target).masked_fill_(mask, 0)) / target.size(0)

        # backward 
        optimizer.zero_grad()
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        # and weights update
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if verbose is True and current_iteration % 25 == 0:
            print('Iteration[{0}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   current_iteration, batch_time=batch_time,
                   data_time=data_time, loss=losses))
        current_iteration = current_iteration + 1
        if total_iterations is not None and current_iteration == total_iterations:
            break
    return current_iteration


class VOC2007_dataset(torch.utils.data.Dataset):
    def __init__(self, voc_dir, split='train', transform=None):
        # Find the image sets
        image_set_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
        image_sets = glob.glob(os.path.join(image_set_dir, '*_' + split + '.txt'))
        assert len(image_sets) == 20
        # Read the labels
        self.n_labels = len(image_sets)
        images = defaultdict(lambda:-np.ones(self.n_labels, dtype=np.uint8)) 
        for k, s in enumerate(sorted(image_sets)):
            for l in open(s, 'r'):
                name, lbl = l.strip().split()
                lbl = int(lbl)
                # Switch the ignore label and 0 label (in VOC -1: not present, 0: ignore)
                if lbl < 0:
                    lbl = 0
                elif lbl == 0:
                    lbl = 255
                images[os.path.join(voc_dir, 'JPEGImages', name + '.jpg')][k] = lbl
        self.images = [(k, images[k]) for k in images.keys()]
        np.random.shuffle(self.images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i][0])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.images[i][1]

if __name__ == '__main__':
    main()

