#!/bin/bash
#
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

VOC="/private/home/bojanowski/data/VOCdevkit/VOC2007"
CAFFE="/private/home/bojanowski/code/unsup-eval-pascal/voc-classification/caffe"

# download code for pascal classification
mkdir -p third-parties
if [ ! -d third-parties/voc-classification ]; then
  git clone https://github.com/philkr/voc-classification.git third-parties/voc-classification
fi

# user config
USERCONFIG=third-parties/voc-classification/src/user_config.py
/bin/cat <<EOM >$USERCONFIG
from os import path
# Path to caffe
CAFFE_DIR = "${CAFFE}"
# Path to the VOC 2007 or 2012 directory
VOC_DIR = "${VOC}"
EOM

# change stepsize in train_cls.py
sed -i -e "s/stepsize=10000/stepsize=20000/g" third-parties/voc-classification/src/train_cls.py
sed -i -e "s/stepsize=5000/stepsize=20000/g" third-parties/voc-classification/src/train_cls.py

# run transfer
MODELROOT="${HOME}/deepcluster_models"
PROTO="${MODELROOT}/alexnet/model.prototxt"
MODEL="${MODELROOT}/alexnet/model.caffemodel"
EXP="${HOME}/deepcluster_exp/pascal_all"
LR=0.001
BSZ=16

mkdir -p ${EXP}

python third-parties/voc-classification/src/train_cls.py ${PROTO} ${MODEL} --output ${EXP}/ \
  --clip ThresholdBackward28 --train-from ConvNdBackward5 \
  --random-from DropoutBackward23 --gpu 0 --no-mean \
  -lr ${LR} -bs ${BSZ} -nit 150000 2>&1 | tee ${EXP}/output.txt
