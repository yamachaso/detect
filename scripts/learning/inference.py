import os
from argparse import ArgumentParser
from pathlib import Path
from time import time

import cv2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from entities.predictor import Predictor
from modules.const import CONFIGS_PATH, DATASETS_PATH, OUTPUTS_PATH

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--weight", "-w", default=f"{OUTPUTS_PATH}/2022_08_04_07_40/model_final.pth")
    parser.add_argument("--device", "-d", default="cuda:0")
    parser.add_argument("--save", action='store_false')
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(f"{CONFIGS_PATH}/config.yaml")
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = args.weight

    dataset_name = "cabbage_val"
    register_coco_instances(dataset_name,
                            {"thing_classes": ["cabbage"]},
                            f"{DATASETS_PATH}/val/COCO_Cabbage_val.json",
                            f"{DATASETS_PATH}/val/images")
    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    print("device :", cfg.MODEL.DEVICE)
    print("weight :", cfg.MODEL.WEIGHTS)
    print("-" * 40)

    predictor = Predictor(cfg)

    save_dir = f"{OUTPUTS_PATH}/images"
    os.makedirs(save_dir, exist_ok=True)
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        start = time()
        outputs = predictor.predict(img)
        end = time()

        out = outputs.draw_instances(img[:, :, ::-1])
        fname = Path(d["file_name"]).stem + ".png"
        print(f"{end - start:.4f}", fname)

        if args.save:
            cv2.imwrite(
                f"{save_dir}/{fname}",
                out.get_image()[:, :, ::-1])
