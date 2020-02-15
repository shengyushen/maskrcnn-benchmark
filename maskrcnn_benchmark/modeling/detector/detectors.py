# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN

# /root/ssy/ssynew/maskrcnn-benchmark/maskrcnn_benchmark/modeling/detector/generalized_rcnn.py
_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    # /root/ssy/training/object_detection/pytorch/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
