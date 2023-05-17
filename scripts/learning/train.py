import os
from datetime import datetime

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from modules.const import CONFIGS_PATH, DATASETS_PATH, OUTPUTS_PATH

for d in ["train", "val"]:
    name = f"cabbage_{d}"
    register_coco_instances(name, {"thing_classes": ["cabbage"]},
                            f"{DATASETS_PATH}/{d}/COCO_Cabbage_{d}.json",
                            f"{DATASETS_PATH}/{d}/images")

metadata = MetadataCatalog.get("cabbage_train")
dataset_dicts = DatasetCatalog.get("cabbage_train")

cfg = get_cfg()
cfg.merge_from_file(f"{CONFIGS_PATH}/config.yaml")
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.OUTPUT_DIR = f"{OUTPUTS_PATH}/{datetime.now().strftime('%Y_%m_%d_%H_%M')}"


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, distributed=False, output_dir=cfg.OUTPUT_DIR)


trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
