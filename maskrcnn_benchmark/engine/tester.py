import torch

import os

from maskrcnn_benchmark.data.build import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.comm import synchronize

def test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    # SSY discard unused second value==0
    #data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    data_loaders_val, _ = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    print("ssy21")
    results = []
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        print("SSY output_folder "+str(output_folder))
        print("SSY dataset_name "+str(dataset_name))
        #print("SSY data_loader_val "+str(data_loader_val))
        # /root/ssy/maskrcnn-benchmark/maskrcnn_benchmark/engine/inference.py
        result = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            # /root/ssy/training/object_detection/pytorch/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        #print("SSY inference result "+str(result))
        print("ssy45")
        synchronize()
        results.append(result)
    return results

