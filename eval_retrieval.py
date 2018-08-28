# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
from collections import OrderedDict
import os
import pickle
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
import torchvision
from torch.autograd import Variable

from util import load_model


class ImageHelper:
    def __init__(self, S, L, transforms):
        self.S = S
        self.L = L
        self.transforms = transforms

    def load_and_prepare_image(self, fname, roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im = Image.open(fname)
        im_size_hw = np.array((im.size[1], im.size[0]))
        if self.S == -1:
            ratio = 1.0
        elif self.S == -2:
            if np.max(im_size_hw) > 124:
                ratio = 1024.0/np.max(im_size_hw)
            else:
                ratio = -1
        else:
            ratio = float(self.S)/np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = self.transforms(im.resize((new_size[1], new_size[0]), Image.BILINEAR))
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            # ROI format is (xmin,ymin,xmax,ymax)
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[:, roi[1]:roi[3], roi[0]:roi[2]]
        return im_resized

    def get_rmac_region_coordinates(self, H, W, L):
        # Almost verbatim from Tolias et al Matlab implementation.
        # Could be heavily pythonized, but really not worth it...
        # Desired overlap of neighboring regions
        ovr = 0.4
        # Possible regions for the long dimension
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)

        b = (np.maximum(H, W) - w) / (steps - 1)
        # steps(idx) regions for long dimension. The +1 comes from Matlab
        # 1-indexing...
        idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1

        # Region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx
        elif H > W:
            Hd = idx

        regions_xywh = []
        for l in range(1, L+1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            # Center coordinates
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            # Center coordinates
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

            for i_ in cenH:
                for j_ in cenW:
                    regions_xywh.append([j_, i_, wl, wl])

        # Round the regions. Careful with the borders!
        for i in range(len(regions_xywh)):
            for j in range(4):
                regions_xywh[i][j] = int(round(regions_xywh[i][j]))
            if regions_xywh[i][0] + regions_xywh[i][2] > W:
                regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
            if regions_xywh[i][1] + regions_xywh[i][3] > H:
                regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
        return np.array(regions_xywh)


class PCA(object):
    '''
    Fits and applies PCA whitening
    '''
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        mean = X.mean(axis=0)
        X -= mean
        self.mean = Variable(torch.from_numpy(mean).view(1, -1))
        Xcov = np.dot(X.T, X)
        d, V = np.linalg.eigh(Xcov)

        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            print("%d / %d singular values are 0" % (n_0, d.size))
            d[d < eps] = eps
        totenergy = d.sum()
        idx = np.argsort(d)[::-1][:self.n_components]
        d = d[idx]
        V = V[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        D = np.diag(1. / np.sqrt(d))
        self.DVt = Variable(torch.from_numpy(np.dot(D, V.T)))

    def to_cuda(self):
        self.mean = self.mean.cuda()
        self.DVt = self.DVt.cuda()

    def apply(self, X):
        X = X - self.mean
        num = torch.mm(self.DVt, X.transpose(0, 1)).transpose(0, 1)
        # L2 normalize on output
        return num


class Dataset:
    def __init__(self, path, eval_binary_path):
        self.path = path
        self.eval_binary_path = eval_binary_path
        # Some images from the Paris dataset are corrupted. Standard practice is
        # to ignore them
        self.blacklisted = set(["paris_louvre_000136",
                            "paris_louvre_000146",
                            "paris_moulinrouge_000422",
                            "paris_museedorsay_001059",
                            "paris_notredame_000188",
                            "paris_pantheon_000284",
                            "paris_pantheon_000960",
                            "paris_pantheon_000974",
                            "paris_pompidou_000195",
                            "paris_pompidou_000196",
                            "paris_pompidou_000201",
                            "paris_pompidou_000467",
                            "paris_pompidou_000640",
                            "paris_sacrecoeur_000299",
                            "paris_sacrecoeur_000330",
                            "paris_sacrecoeur_000353",
                            "paris_triomphe_000662",
                            "paris_triomphe_000833",
                            "paris_triomphe_000863",
                            "paris_triomphe_000867"])
        self.load()

    def load(self):
        # Load the dataset GT
        self.lab_root = '{0}/lab/'.format(self.path)
        self.img_root = '{0}/jpg/'.format(self.path)
        lab_filenames = np.sort(os.listdir(self.lab_root))
        # Get the filenames without the extension
        self.img_filenames = [e[:-4] for e in np.sort(os.listdir(self.img_root))
                              if e[:-4] not in self.blacklisted]

        # Parse the label files. Some challenges as filenames do not correspond
        # exactly to query names. Go through all the labels to:
        # i) map names to filenames and vice versa
        # ii) get the relevant regions of interest of the queries,
        # iii) get the indexes of the dataset images that are queries
        # iv) get the relevants / non-relevants list
        self.relevants = {}
        self.junk = {}
        self.non_relevants = {}

        self.filename_to_name = {}
        self.name_to_filename = OrderedDict()
        self.q_roi = {}
        for e in lab_filenames:
            if e.endswith('_query.txt'):
                q_name = e[:-len('_query.txt')]
                q_data = open("{0}/{1}".format(self.lab_root, e)).readline().split(" ")
                q_filename = q_data[0][5:] if q_data[0].startswith('oxc1_') else q_data[0]
                self.filename_to_name[q_filename] = q_name
                self.name_to_filename[q_name] = q_filename
                good = set([e.strip() for e in open("{0}/{1}_ok.txt".format(self.lab_root, q_name))])
                good = good.union(set([e.strip() for e in open("{0}/{1}_good.txt".format(self.lab_root, q_name))]))
                junk = set([e.strip() for e in open("{0}/{1}_junk.txt".format(self.lab_root, q_name))])
                good_plus_junk = good.union(junk)
                self.relevants[q_name] = [i for i in range(len(self.img_filenames))
                                          if self.img_filenames[i] in good]
                self.junk[q_name] = [i for i in range(len(self.img_filenames))
                                     if self.img_filenames[i] in junk]
                self.non_relevants[q_name] = [i for i in range(len(self.img_filenames))
                                              if self.img_filenames[i] not in good_plus_junk]
                self.q_roi[q_name] = np.array([float(q) for q in q_data[1:]], dtype=np.float32)
                #np.array(map(float, q_data[1:]), dtype=np.float32)

        self.q_names = self.name_to_filename.keys()
        self.q_index = np.array([self.img_filenames.index(self.name_to_filename[qn])
                                 for qn in self.q_names])
        self.N_images = len(self.img_filenames)
        self.N_queries = len(self.q_index)

    def score(self, sim, temp_dir, eval_bin):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        idx = np.argsort(sim, axis=1)[:, ::-1]
        maps = [self.score_rnk_partial(i, idx[i], temp_dir, eval_bin)
                for i in range(len(self.q_names))]
        for i in range(len(self.q_names)):
            print("{0}: {1:.2f}".format(self.q_names[i], 100 * maps[i]))
        print(20 * "-")
        print("Mean: {0:.2f}".format(100 * np.mean(maps)))

    def score_rnk_partial(self, i, idx, temp_dir, eval_bin):
        rnk = np.array(self.img_filenames)[idx]
        with open("{0}/{1}.rnk".format(temp_dir, self.q_names[i]), 'w') as f:
            f.write("\n".join(rnk)+"\n")
        cmd = "{0} {1}{2} {3}/{4}.rnk".format(eval_bin, self.lab_root, self.q_names[i], temp_dir, self.q_names[i])
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        map_ = float(p.stdout.readlines()[0])
        p.wait()
        return map_

    def get_filename(self, i):
        return os.path.normpath("{0}/{1}.jpg".format(self.img_root,
                                                     self.img_filenames[i]))

    def get_query_filename(self, i):
        return os.path.normpath("{0}/{1}.jpg".format(self.img_root,
                                                     self.img_filenames[self.q_index[i]]))

    def get_query_roi(self, i):
        return self.q_roi[self.q_names[i]]


def ensure_directory_exists(fname):
    dirname = fname[:fname.rfind('/')]
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def normalize_L2(a, dim):
    norms = torch.sqrt(torch.sum(a**2, dim=dim, keepdim=True))
    return a / norms


def rmac(features, rmac_levels, pca=None):
    nim, nc, xd, yd = features.size()

    rmac_regions = image_helper.get_rmac_region_coordinates(xd, yd, rmac_levels)
    rmac_regions = rmac_regions.astype(np.int)
    nr = len(rmac_regions)

    rmac_descriptors = []
    for x0, y0, w, h in rmac_regions:
        desc = features[:, :, y0:y0 + h, x0:x0 + w]
        desc = torch.max(desc, 2, keepdim=True)[0]
        desc = torch.max(desc, 3, keepdim=True)[0]
        # insert an additional dimension for the cat to work
        rmac_descriptors.append(desc.view(-1, 1, nc))

    rmac_descriptors = torch.cat(rmac_descriptors, 1)

    rmac_descriptors = normalize_L2(rmac_descriptors, 2)

    if pca is None:
        return rmac_descriptors

    # PCA + whitening
    npca = pca.n_components
    rmac_descriptors = pca.apply(rmac_descriptors.view(nr * nim, nc))
    rmac_descriptors = normalize_L2(rmac_descriptors, 1)

    rmac_descriptors = rmac_descriptors.view(nim, nr, npca)

    # Sum aggregation and L2-normalization
    rmac_descriptors = torch.sum(rmac_descriptors, 1)
    rmac_descriptors = normalize_L2(rmac_descriptors, 1)
    return rmac_descriptors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Oxford / Paris')
    parser.add_argument('--S', type=int, default=1024,
                        help='Resize larger side of image to S pixels (e.g. 800)')
    parser.add_argument('--L', type=int, default=3,
                        help='Use L spatial levels (e.g. 3)')
    parser.add_argument('--n_pca', type=int, default=512,
                        help='output dimension of PCA')
    parser.add_argument('--model', type=str, default='pretrained',
                        help='Model from which RMAC is computed')
    parser.add_argument('--dataset', type=str, required=True,
                        help='path to dataset')
    parser.add_argument('--dataset_name', type=str, default='Oxford',
                        choices=['Oxford', 'Paris'], help='Dataset name')
    parser.add_argument('--stage', type=str, default='extract_train',
                        choices=['extract_train', 'train_pca', 'db_features',
                        'q_features', 'eval'], help='what action to perform ')
    parser.add_argument('--eval_binary', type=str, required=True,
                        help='Path to the compute_ap binary to evaluate Oxford / Paris')
    parser.add_argument('--temp_dir', type=str, default='',
                        help='Path to a temporary directory to store features and scores')
    parser.add_argument('--multires', dest='multires', action='store_true',
                        help='Enable multiresolution features')
    parser.add_argument('--aqe', type=int, required=False,
                        help='Average query expansion with k neighbors')
    parser.add_argument('--dbe', type=int, required=False,
                        help='Database expansion with k neighbors')

    parser.set_defaults(multires=False)
    args = parser.parse_args()

    # Load the dataset and the image helper
    print "Prepare the dataset from ", args.dataset
    dataset = Dataset(args.dataset, args.eval_binary)

    ensure_directory_exists(args.temp_dir + '/')

    if args.stage in ('extract_train', 'db_features', 'q_features'):

        if args.model == 'pretrained':
            print("loading supervised pretrained VGG-16")
            net = torchvision.models.vgg16_bn(pretrained=True)
        else:
            net = load_model(args.model)

        transforms_comp = []
        features_layers = list(net.features.children())[:-1]
        net.features = torch.nn.Sequential(*features_layers)
        transforms_comp.extend([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])

        transforms = torchvision.transforms.Compose(transforms_comp)

        print("moving to GPU")
        net.cuda()
        net.eval()
        print("  done")

        print("initialize image helper")
        image_helper = ImageHelper(args.S, args.L, transforms)


    if args.stage == 'extract_train':
        print("extract regions for training")
        # extract at a single scale
        S = args.S
        image_helper.S = S
        N_dataset = dataset.N_images
        def process_image(i):
            print(i),
            sys.stdout.flush()
            fname_out = "{0}/{1}_S{2}_L{3}_regions/{4}.npy".format(args.temp_dir, args.dataset_name, S, args.L, i)

            ensure_directory_exists(fname_out)
            I = image_helper.load_and_prepare_image(dataset.get_filename(i), roi=None)
            v = torch.autograd.Variable(I.unsqueeze(0))
            vc = v.cuda()
            if hasattr(net, 'sobel') and net.sobel is not None:
                vc = net.sobel(vc)
            activation_map = net.features(vc).cpu()

            rmac_descriptors = rmac(activation_map, args.L)
            np.save(fname_out, rmac_descriptors.data.numpy())

        map(process_image, range(dataset.N_images))

    elif args.stage == 'train_pca':
        # load training vectors
        train_x = []
        for i in range(10000):
            fname_in = "{0}/{1}_S{2}_L{3}_regions/{4}.npy".format(args.temp_dir, args.dataset_name, args.S, args.L, i)
            if not os.path.exists(fname_in):
                break
            x = np.load(fname_in)
            train_x.append(x)

        print("loaded %d train vectors" % len(train_x))

        train_x = np.vstack([x.reshape(-1, x.shape[-1]) for x in train_x])
        print("   size", train_x.shape)

        pca = PCA(args.n_pca)
        pca.fit(train_x)
        pcaname = '%s/%s_S%d_PCA.pickle' % (args.temp_dir, args.dataset_name, args.S)

        print("writing", pcaname)
        pickle.dump(pca, open(pcaname, 'w'), -1)

    elif args.stage == 'db_features' or args.stage == 'q_features':
        # for tests on Paris, use Oxford PCA, and vice-versa
        pcaname = '%s/%s_S%d_PCA.pickle' % (
            args.temp_dir, 'Paris' if args.dataset_name == 'Oxford' else 'Oxford', args.S)
        print("loading PCA from", pcaname)
        pca = pickle.load(open(pcaname, 'r'))

        print("Compute features")
        # extract at a single scale
        S = args.S
        image_helper.S = S
        N_dataset = dataset.N_images

        def process_image(fname_in, roi, fname_out):
            softmax = torch.nn.Softmax().cuda()
            I = image_helper.load_and_prepare_image(fname_in, roi=roi)
            v = torch.autograd.Variable(I.unsqueeze(0))
            vc = v.cuda()
            if hasattr(net, 'sobel') and net.sobel is not None:
                vc = net.sobel(vc)
            activation_map = net.features(vc).cpu()
            descriptors = rmac(activation_map, args.L, pca=pca)
            np.save(fname_out, descriptors.data.numpy())

        if args.stage == 'db_features':
            for i in range(dataset.N_images):
                fname_in = dataset.get_filename(i)
                fname_out = "{0}/{1}_S{2}_L{3}_db/{4}.npy".format(args.temp_dir, args.dataset_name, S, args.L, i)
                ensure_directory_exists(fname_out)
                print(i),
                sys.stdout.flush()
                process_image(fname_in, None, fname_out)

        elif args.stage == 'q_features':
            for i in range(dataset.N_queries):
                fname_in = dataset.get_query_filename(i)
                roi = dataset.get_query_roi(i)
                fname_out = "{0}/{1}_S{2}_L{3}_q/{4}.npy".format(args.temp_dir, args.dataset_name, S, args.L, i)
                ensure_directory_exists(fname_out)
                print(i),
                sys.stdout.flush()
                process_image(fname_in, roi, fname_out)

    elif args.stage == 'eval':
        S = args.S

        print("load query features")
        features_queries = []
        for i in range(dataset.N_queries):
            fname = "{0}/{1}_S{2}_L{3}_q/{4}.npy".format(args.temp_dir, args.dataset_name, S, args.L, i)
            features_queries.append(np.load(fname))
        features_queries = np.vstack(features_queries)

        print("  size", features_queries.shape)

        print("load database features")
        features_dataset = []
        for i in range(dataset.N_images):
            fname = "{0}/{1}_S{2}_L{3}_db/{4}.npy".format(args.temp_dir, args.dataset_name, S, args.L, i)
            features_dataset.append(np.load(fname))
        features_dataset = np.vstack(features_dataset)
        print("  size", features_dataset.shape)

        # Compute similarity
        sim = features_queries.dot(features_dataset.T)

        # Score
        dataset.score(sim, args.temp_dir, args.eval_binary)
