# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

VOC='/private/home/bojanowski/data/VOCdevkit/VOC2007'
CAFFE='/private/home/bojanowski/code/unsup-eval-pascal/voc-classification/caffe'

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
CAFFE_DIR = '${CAFFE}'
# Path to the VOC 2007 or 2012 directory
VOC_DIR = '${VOC}'
EOM

# change stepsize in train_cls.py
sed -i -e 's/stepsize=10000/stepsize=5000/g' third-parties/voc-classification/src/train_cls.py
sed -i -e 's/stepsize=20000/stepsize=5000/g' third-parties/voc-classification/src/train_cls.py

# run transfer
PROTO="/private/home/mathilde/model-to-release/alexnet/model.prototxt"
MODEL="/private/home/mathilde/model-to-release/alexnet/model.caffemodel"
LR=0.003
BSZ=16
EXP=""

mkdir -p ${EXP}

python third-parties/voc-classification/src/train_cls.py ${PROTO} ${MODEL} --output ${EXP}/ \
--clip ThresholdBackward28 --train-from DropoutBackward23 \
--random-from DropoutBackward23 --gpu 0 --no-mean \
-lr ${LR} -bs ${BSZ} -nit 150000 2>&1 | tee ${EXP}/output.txt
