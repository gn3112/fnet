from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.engine import DefaultTrainer, EvalHook
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import ImageList, BoxMode
from pycocotools.mask import decode

import json 
import os
import cv2
import numpy as np
import random
from utils import show_bbox, overlap_feature

import torch 
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor

FILE_PATH = os.getcwd()
DATASET_PATH = os.path.join(FILE_PATH, "fruit_dataset2")
N_SAMPLES = 5

def get_fruit_dicts():
    record = []
    samples_dict_file = [f for f in os.listdir(DATASET_PATH) if f.endswith('.txt')]
    for i in range(N_SAMPLES):
        sample_dict_file = os.path.join(DATASET_PATH, samples_dict_file[i])
        with open(sample_dict_file) as f:
            sample_dict = json.load(f)
        sample_dict["file_name"] = sample_dict["file_name"].replace("Users","Users")
        for obj in sample_dict["annotations"]:
            mask = decode(eval(obj["segmentation"]))
            _, contours, _ = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                                    cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []

            for contour in contours:
                contour = contour.flatten().tolist()
                if len(contour) > 4:
                    segmentation.append(contour)

            obj["segmentation"] = segmentation
            obj["bbox_mode"] = BoxMode.XYWH_ABS

        record.append(sample_dict)
    
    return record

def func():
    # Build dataloader
    loader = build_detection_test_loader(cfg, "fruit_valid", mapper=sample)



if __name__ == "__main__": 
    with open('class_idx.txt') as f:
        class_idx = json.load(f)

    # Smaller dataset for now
    class_idx = {name: i for i, name in enumerate(['Tomato', 'Orange', 'Lemon', 'Apple', 'Banana', 'Avocado'])} 

    dataset_dicts = get_fruit_dicts()

    for d in ["train","test"]:
        DatasetCatalog.register("fruit_{}".format(d), lambda d=d: get_fruit_dicts())
        MetadataCatalog.get("fruit_{}".format(d)).set(thing_classes=list(class_idx.keys()))

    fruit_metadata = MetadataCatalog.get("fruit_test").evaluator_type
    print(fruit_metadata)

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("fruit_train",)
    cfg.DATASETS.TEST = ("fruit_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 30    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_idx)  
    cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 1.4]]
    cfg.TEST.EVAL_PERIOD = 2
    cfg.OUTPUT_DIR = "test"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # eval_hook = EvalHook(period, func)




    trainer = DefaultTrainer(cfg) 
    # trainer.register_hooks(eval_hook)
    # trainer.register_hooks([EvalHook(,)])
    trainer.resume_or_load(resume=False)
    trainer.train()