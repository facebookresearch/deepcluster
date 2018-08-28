# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash


# This source is adapted from the "deep_retrieval" package that comes with
# Deep Image Retrieval: Learning global representations for image search. A. Gordo, J. Almazan, J. Revaud, and D. Larlus. In ECCV, 2016
# The original source is not accessible anymore, but other people shared the code, see eg. https://github.com/figitaki/deep-retrieval
# follow the instructions on that github repo to download the data, compile the evaluation package, and set the path to the resulting directory below:

#DATASETS='./datasets'

# load pytorch model from here
MODEL='/private/home/mathilde/model-to-release/vgg16/checkpoint.pth.tar'

# this is to obtain the supervised performance
#MODEL='pretrained'

TEMP='/private/home/mathilde/temp'

# should be compiled as part of the dataset preparation
EVALBINARY="$DATASETS/evaluation/compute_ap"
EVAL='Paris'
PCA='Oxford'
DATASETEVAL="$DATASETS/$EVAL"
DATASETPCA="$DATASETS/$PCA"

python eval_retrieval.py --model ${MODEL} --eval_binary ${EVALBINARY} --temp_dir ${TEMP} --dataset ${DATASETPCA} --dataset_name ${PCA} --stage extract_train
python eval_retrieval.py --model ${MODEL} --eval_binary ${EVALBINARY} --temp_dir ${TEMP} --dataset ${DATASETPCA} --dataset_name ${PCA} --stage train_pca
python eval_retrieval.py --model ${MODEL} --eval_binary ${EVALBINARY} --temp_dir ${TEMP} --dataset ${DATASETEVAL} --dataset_name ${EVAL} --stage q_features
python eval_retrieval.py --model ${MODEL} --eval_binary ${EVALBINARY} --temp_dir ${TEMP} --dataset ${DATASETEVAL} --dataset_name ${EVAL} --stage db_features
python eval_retrieval.py --model ${MODEL} --eval_binary ${EVALBINARY} --temp_dir ${TEMP} --dataset ${DATASETEVAL} --dataset_name ${EVAL} --stage eval
