#!/bin/bash
set -eu

GPUS=$1
BASE_LR=0.02 #0.01
IMS_PER_BATCH=32 #4
MAX_ITER=1400000 #1350000 #900000
TRAINSET=O365_Open800k_VI

JOB_NAME=mfrcnn_Bs$IMS_PER_BATCH-iter_$MAX_ITER-lr_$BASE_LR
OUTPUT_DIR=/home/chnxi/Models/GOD/$TRAINSET/$JOB_NAME/

STAMP=$(date +"%Y%m%d_%H%M%S")

echo $JOB_NAME

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi

export NGPUS=4
CUDA_VISIBLE_DEVICES="$GPUS"  python -W ignore::DeprecationWarning -m torch.distributed.launch --nproc_per_node=$NGPUS \
    ./tools/train_net.py --config-file "./configs/GOD/e2e_faster_rcnn_R_50_FPN_1x.yaml" \
    DATASETS.TRAIN "('GOD_O365_Open800k_train', 'GOD_VisualIntent_train', )" \
    MODEL.ROI_BOX_HEAD.NUM_CLASSES 847 \
    SOLVER.IMS_PER_BATCH $IMS_PER_BATCH \
    SOLVER.BASE_LR $BASE_LR \
    SOLVER.MAX_ITER  $MAX_ITER \
    OUTPUT_DIR $OUTPUT_DIR \
    | tee -a $OUTPUT_DIR/log_$STAMP.txt