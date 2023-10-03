# %%
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
   
        for i in range(mask_array.shape[0]):
            if self.scores[i] < 0.95:
                outlier_indexes.append(i)
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
        self.mask_array_shape = (20, 30)

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

from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from detectron2.config import get_cfg
# from entities.predictor import Predictor
from modules.const import CONFIGS_PATH, OUTPUTS_PATH, SAMPLES_PATH
from modules.grasp import GraspDetector
from modules.image import (compute_optimal_depth_thresh,
                           extract_flont_instance_indexes,
                           extract_flont_mask_with_thresh, merge_mask,
                           transform_ddi)
from modules.visualize import convert_rgb_to_3dgray, get_color_by_score
from utils import RealsenseBagHandler, imshow

config_path = f"{CONFIGS_PATH}/config.yaml"
weight_path = f"{OUTPUTS_PATH}/mask_rcnn/model_final.pth"
device = "cuda:0"

print(config_path)
print(weight_path)
print(device)

cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = weight_path
cfg.MODEL.DEVICE = device

predictor = Predictor(cfg)
path = glob(f"{SAMPLES_PATH}/realsense_viewer_bags/*")[0]
handler = RealsenseBagHandler(path, 640, 480, 30)

img, depth = handler.get_images()
print(img.dtype, depth.dtype)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[1].imshow(depth, cmap="binary")
cv2.imwrite('after_Lena.jpg', img)

res = predictor.predict(img)
# print(res.scores)
seg = res.draw_instances(img[:, :, ::-1])
imshow(seg)

print(res.num_instances)
print(res.mask_array_shape)

# %%
print(len(res.instances))
print(res.instances)
print(res.num_instances)

# %%
masks = res.masks
merged_mask = merge_mask(masks)
# depthの欠損箇所があれば全体マスクから除外
valid_mask = np.where(merged_mask * depth > 0, 255, 0).astype("uint8")
ddi = transform_ddi(np.where(valid_mask > 0, depth, depth[depth > 0].mean()), 5)
opt_depth_th = compute_optimal_depth_thresh(depth, valid_mask, n=3)
print("opt_depth_th :", opt_depth_th)
# この段階でのflont_maskはコンテナ含んだり、インスタンスの見切れなどもそんざいする
raw_flont_mask = extract_flont_mask_with_thresh(depth, opt_depth_th, n=3)
raw_flont_img = cv2.bitwise_and(img, img, mask=raw_flont_mask)
fig, axes = plt.subplots(1, 3)
axes[0].imshow(merged_mask)
axes[1].imshow(ddi)
axes[2].imshow(raw_flont_img)
# %% depth filteringの結果とインスタンスセグメンテーションの結果のマージ (背景除去 & マスク拡大)
# flont_mask = refine_flont_mask(raw_flont_mask, masks, 0.8)
flont_indexes = extract_flont_instance_indexes(raw_flont_mask, masks, 0.8)
flont_mask = merge_mask(masks[flont_indexes])
flont_img = cv2.bitwise_and(img, img, mask=flont_mask)
fig, axes = plt.subplots(1, 3)
axes[0].imshow(img)
axes[1].imshow(flont_mask)
axes[2].imshow(flont_img)


# %%
finger_num = 4
hand_radius_mm = 150
finger_radius_mm = 1
unit_angle = 15
frame_size = img.shape[:2]
fp = handler.fp
elements_th = 0
el_insertion_th = 0
el_contact_th = 0
el_bw_depth_th = 0
detector = GraspDetector(finger_num=finger_num, hand_radius_mm=hand_radius_mm,
                         finger_radius_mm=finger_radius_mm,
                         unit_angle=unit_angle, frame_size=frame_size, fp=fp,
                         elements_th=elements_th, 
                         el_insertion_th=el_insertion_th, el_contact_th=el_contact_th,
                         el_bw_depth_th=el_bw_depth_th)
# %%


def get_radius_for_augment(bbox):
    side_h = np.linalg.norm((bbox[0] - bbox[1])) / 2
    side_v = np.linalg.norm((bbox[0] - bbox[2])) / 2
    short_side = min(side_h, side_v)

    return int(short_side // 2)


gray_3c = convert_rgb_to_3dgray(img)
reversed_flont_mask = cv2.bitwise_not(flont_mask)
base_img = cv2.bitwise_and(img, img, mask=flont_mask) + \
    cv2.bitwise_and(gray_3c, gray_3c, mask=reversed_flont_mask)

top_cnd_num = 1
cnd_img_1 = base_img.copy()
cnd_img_2 = base_img.copy()
flont_indexes_set = set(flont_indexes)
for i in range(res.num_instances):
    label = str(res.labels[i])
    score = res.scores[i]
    center = res.centers[i]
    mask = res.masks[i]
    contour = res.contours[i]
    bbox = res.bboxes[i]

    is_flont = i in flont_indexes_set

    if is_flont:
        radius_for_augment = get_radius_for_augment(bbox)
        candidates = detector.detect(center, depth, contour, radius_for_augment=radius_for_augment)
        scores = [cnd.total_score for cnd in candidates]
        sorted_indexes = np.argsort(scores)[::-1]
        # for cnd in candidates:
        for i in sorted_indexes[:top_cnd_num]:
            cnd = candidates[i]
            color = get_color_by_score(cnd.total_score)
            if cnd.is_framein:
                cnd.draw(cnd_img_1, color, line_thickness=1)
                if cnd.is_valid:
                    # print(best_cnd.center_d, [el.insertion_point_d for el in best_cnd.elements])
                    cnd.draw(cnd_img_2, color, line_thickness=1)
                    cv2.circle(cnd_img_2, cnd.center, 3, (0, 255, 0), -1)

    color = (255, 100, 0) if is_flont else (0, 100, 255)
    for target_img in (cnd_img_1, cnd_img_2):
        cv2.drawContours(target_img, [contour], -1, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(target_img, center, 3, (0, 0, 255), -1)

imshow(cnd_img_1)
imshow(cnd_img_2)

# %%
imshow(res.masks[2])
test_image = np.zeros((res.masks[5].shape[0], res.masks[5].shape[1], 3))
test_image[:,:,0] = res.masks[2]
test_image[:,:,1] = res.masks[2]
test_image[:,:,2] = res.masks[2]
test_image = test_image * 255
# test_image = cv2.cvtColor(res.masks[5], cv2.COLOR_GRAY2RGB)
# test_image=res.masks[5]
imshow(test_image)
ellipse = cv2.fitEllipse(res.contours[2])
test_image= cv2.ellipse(test_image,ellipse,(0,255,0),2)
imshow(test_image)
# plt.imshow(test_image)
# %%
