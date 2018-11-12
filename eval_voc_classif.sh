# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

VOCDIR=""
MODELROOT="${HOME}/deepcluster_models"
MODEL="${MODELROOT}/alexnet/checkpoint.pth.tar"

PYTHON="${HOME}/test/conda/bin/python"

# with training the batch norm
# 72.0 mAP
$PYTHON eval_voc_classif.py --vocdir $VOCDIR --model $MODEL --split trainval --fc6_8 1 --train_batchnorm 1

# without training the batch norm
# 70.4 mAP
$PYTHON eval_voc_classif.py --vocdir $VOCDIR --model $MODEL --split trainval --fc6_8 1 --train_batchnorm 0
