#!/bin/bash
zip -r ../checkpoints/init-train.zip init-train \
 -x "init-train/checkpoints/*.model" \
 -x "init-train/eval/*" \
 -x "init-train/results/*"


zip -r ../checkpoints/qbc-work.zip qbc-work \
 -x "qbc-work/.ipynb_checkpoints/*.model" \
 -x "qbc-work/results/*" \
 -x "qbc-work/structures/candidates.*" \
 -x "qbc-work/structures/train-*" \
 -x "qbc-work/log/*" \
 -x "qbc-work/eval/train.*"\
 -x "qbc-work/train-iter.extxyz"

zip -r ../checkpoints/random-train.zip random-train \
 -x "init-train/checkpoints/*.model" \
 -x "init-train/results/*"

zip -r ../checkpoints/eigen-qbc-work.zip eigen-qbc-work \
  -x "eigen-qbc-work/config/*" \
  -x "eigen-qbc-work/aims/*" \
  -x "eigen-qbc-work/results/*" \
  -x "eigen-qbc-work/structures/*" \
  -x "eigen-qbc-work/checkpoints/*" \
  -x "eigen-qbc-work/eval/*model*" \
  -x "eigen-qbc-work/log/*" \
  -x "eigen-qbc-work/candidates.start.extxyz"

