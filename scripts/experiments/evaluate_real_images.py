# %%
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from detectron2.config import get_cfg
from entities.predictor import Predictor
from modules.const import CONFIGS_PATH, OUTPUTS_PATH, SAMPLES_PATH
from modules.grasp import GraspDetector
from modules.image import (compute_optimal_depth_thresh,
                           extract_flont_instance_indexes,
                           extract_flont_mask_with_thresh, merge_mask,
                           transform_ddi)
from modules.visualize import convert_rgb_to_3dgray, get_color_by_score
from utils import RealsenseBagHandler, imshow

# %%
config_path = f"{CONFIGS_PATH}/config.yaml"
weight_path = f"{OUTPUTS_PATH}/2022_10_16_08_01/model_final.pth"
device = "cuda:0"

cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = weight_path
cfg.MODEL.DEVICE = device

predictor = Predictor(cfg)
# %%
path = glob(f"{SAMPLES_PATH}/realsense_viewer_bags/*")[0]
handler = RealsenseBagHandler(path, 640, 480, 30)

img, depth = handler.get_images()
print(img.dtype, depth.dtype)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[1].imshow(depth, cmap="binary")

depth.shape
# %%
res = predictor.predict(img)
seg = res.draw_instances(img[:, :, ::-1])
imshow(seg)

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
center_diff_th = 0
el_insertion_th = 0
el_contact_th = 0
el_bw_depth_th = 0
detector = GraspDetector(finger_num=finger_num, hand_radius_mm=hand_radius_mm,
                         finger_radius_mm=finger_radius_mm,
                         unit_angle=unit_angle, frame_size=frame_size, fp=fp,
                         elements_th=elements_th, center_diff_th=center_diff_th,
                         el_insertion_th=el_insertion_th, el_contact_th=el_contact_th,
                         el_bw_depth_th=el_bw_depth_th,
                         augment_anchors=True)
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
