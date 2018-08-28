# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

MODEL='/private/home/mathilde/model-to-release/alexnet/checkpoint.pth.tar'
ARCH='vgg16'
EXP='/private/home/mathilde/temp'
CONV=6

python gradient_ascent.py --model ${MODEL} --exp ${EXP} --conv ${CONV} --arch ${ARCH}
