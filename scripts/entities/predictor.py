from typing import Any, Tuple, TypedDict, Union

import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes
from detectron2.structures import Instances as RawInstances
from detectron2.utils.visualizer import ColorMode, Visualizer
from modules.utils import smirnov_grubbs
from torch import Tensor

from entities.image import BinaryMask


class Instances(RawInstances):
    """predict完了して各値がセットされた後のInstances(type指定用)"""

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        super().__init__(image_size, **kwargs)
        self.boxes: Boxes
        self.pred_masks: Tensor
        self.scores: Tensor
        self.pred_classes: Tensor

    # 戻り値の型を上書きするためにオーバーライド
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        super().to(*args, **kwargs)


class OutputsDictType(TypedDict):
    instances: Instances


class PredictResult:
    """detecton2のpredictの出力を扱いやすい形にパースする"""

    def __init__(self, outputs_dict: OutputsDictType, device: str = "cpu"):
        self._instances: Instances = outputs_dict['instances'].to(device)

        self.scores: np.ndarray = self._instances.scores.numpy()
        self.labels: np.ndarray = self._instances.pred_classes.numpy()

        mask_array: np.ndarray = self._instances.pred_masks.numpy().astype("uint8")

        # rotated_bboxの形式は(center, weight, height, angle)の方がよい？
        # radiusも返すべき？
        # contourはどうやってｍｓｇに渡す？
        masks = []
        contours = []
        centers = []
        bboxes = []
        areas = []
        for each_mask_array in mask_array:
            each_mask = BinaryMask(each_mask_array)
            closing_mask = cv2.morphologyEx(
                each_mask.mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
            masks.append(closing_mask)
            contours.append(each_mask.contour)
            centers.append(each_mask.get_center())
            bboxes.append(each_mask.get_rotated_bbox())
            areas.append(each_mask.get_area())

        # NMSで除去できない不良インスタンスの除去
        outlier_indexes = smirnov_grubbs(areas, 0.05)
        print(outlier_indexes)
        valid_indexes = [i for i in range(
            mask_array.shape[0]) if i not in outlier_indexes]

        self.instances = self._instances[valid_indexes]
        self.num_instances = len(self.instances)
        self.masks = np.array(masks)[valid_indexes]
        self.contours = np.array(contours)[valid_indexes]
        self.centers = np.array(centers)[valid_indexes]
        self.bboxes = np.array(bboxes)[valid_indexes]
        self.areas = np.array(areas)[valid_indexes]

    def draw_instances(self, img, metadata={}, scale=0.5, instance_mode=ColorMode.IMAGE_BW, targets: Union[list, np.ndarray, None] = None):
        v = Visualizer(
            img,
            metadata=metadata,
            scale=scale,
            # remove the colors of unsegmented pixels.
            # This option is only available for segmentation models
            instance_mode=instance_mode
        )
        instances = self.instances if targets is None else self.instances[targets]
        return v.draw_instance_predictions(instances).get_image()[:, :, ::-1]


# これは別の場所に置くべきでは
class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, img):
        outputs = self.predictor(img)
        parsed_outputs = PredictResult(outputs)
        return parsed_outputs
