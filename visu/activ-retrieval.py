# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
from shutil import copyfile
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.insert(0, '..')
from util import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='Retrieve images with maximal activations')
    parser.add_argument('--data', type=str, help='path to dataset')
    parser.add_argument('--model', type=str, help='Model')
    parser.add_argument('--conv', type=int, default=1, help='convolutional layer')
    parser.add_argument('--exp', type=str, default='', help='path to res')
    parser.add_argument('--count', type=int, default=9, help='save this many images')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    return parser.parse_args()


def main(args):
    # create repo
    repo = os.path.join(args.exp, 'conv' + str(args.conv))
    if not os.path.isdir(repo):
        os.makedirs(repo)

    # build model
    model = load_model(args.model)
    model.cuda()
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    #load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    # dataset
    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                             num_workers=args.workers)

    # keys are filters and value are arrays with activation scores for the whole dataset
    layers_activations = {}
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        activations = forward(model, args.conv, input_var)

        if i == 0:
            layers_activations = {filt: np.zeros(len(dataset)) for filt in activations}
        if i < len(dataloader) - 1:
            e_idx = (i + 1) * 256
        else:
            e_idx = len(dataset)
        s_idx = i * 256
        for filt in activations:
            layers_activations[filt][s_idx: e_idx] = activations[filt].cpu().data.numpy()

        if i % 100 == 0:
            print('{0}/{1}'.format(i, len(dataloader)))

    # save top N images for each filter
    for filt in layers_activations:
        repofilter = os.path.join(repo, filt)
        if not os.path.isdir(repofilter):
            os.mkdir(repofilter)
        top = np.argsort(layers_activations[filt])[::-1]
        if args.count > 0:
            top = top[:args.count]

        for pos, img in enumerate(top):
            src, _ = dataset.imgs[img]
            copyfile(src, os.path.join(repofilter, "{}_{}".format(pos, src.split('/')[-1])))


def forward(model, my_layer, x):
    if model.sobel is not None:
        x = model.sobel(x)
    layer = 1
    res = {}
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
            if isinstance(m, nn.ReLU):
                if layer == my_layer:
                    for channel in range(int(x.size()[1])):
                        key = 'layer' + str(layer) + '-channel' + str(channel)
                        res[key] = torch.squeeze(x.mean(3).mean(2))[:, channel]
                    return res
                layer = layer + 1
    return res


if __name__ == '__main__':
    args = parse_args()
    main(args)
