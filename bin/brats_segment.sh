#!/usr/bin/env bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8


export CUDA_VISIBLE_DEVICES=0
export USE_WANDB=0
export USE_ALCHEMY=0
export USE_NEPTUNE=0

RUN_CONFIG=brats_segment.yml

model_name=resnet34
fold=0
#data=TC
for data in TC E WT; do
  log_name=${model_name}-ss-crop-$data-256-$fold
  LOGDIR=/logs/papers/segmentation/brats/200e/${log_name}/
  catalyst-dl run \
      --config=./configs/${RUN_CONFIG} \
      --logdir=$LOGDIR \
      --out_dir=$LOGDIR:str \
      --stages/data_params/fold=${fold}:int \
      --stages/data_params/data=${data}:str \
      --verbose
done