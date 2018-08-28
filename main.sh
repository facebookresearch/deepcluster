# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/datasets01/imagenet_full_size/061417/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
WORKERS=12
EXP="/private/home/${USER}/test/exp"
PYTHON="/private/home/${USER}/test/conda/bin/python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
