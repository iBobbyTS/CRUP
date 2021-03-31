#!/usr/bin/env sh

python train.py \
  -dataset /Users/ibobby/Dataset/hdr \
  -low 256 \
  -full 512 \
  -lr 1e-4 \
  -bs 4 \
  -ff /Users/ibobby/root/bin
