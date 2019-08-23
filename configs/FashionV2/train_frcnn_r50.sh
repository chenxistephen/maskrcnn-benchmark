#!/bin/bash
set -eu

export NGPUS=8
python -W ignore::DeprecationWarning -m torch.distributed.launch --nproc_per_node=$NGPUS \
    ./tools/train_net.py --config-file "./configs/FashionV2/e2e_faster_rcnn_R_50_FPN_1x.yaml" \
    MODEL.ROI_BOX_HEAD.NUM_CLASSES 83 \
    SOLVER.IMS_PER_BATCH 16