# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/datasets01/imagenet_full_size/061417/"
MODELROOT="${HOME}/deepcluster_models"
MODEL="${MODELROOT}/alexnet/checkpoint.pth.tar"
EXP="${HOME}/deepcluster_exp/linear_classif"

PYTHON="${HOME}/test/conda/bin/python"

mkdir -p ${EXP}

${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --conv 3 --lr 0.01 \
  --wd -7 --tencrops --verbose --exp ${EXP} --workers 12
