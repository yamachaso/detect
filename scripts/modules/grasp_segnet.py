from typing import List, Tuple

import cv2
import numpy as np

from modules.image import extract_depth_between_two_points
from modules.type import Image, ImagePointUV, ImagePointUVD, Mm, Px

import rospy
from detect.msg import Instance


import numpy as np
import torch
from modules.segnet.segnet import SegNetBasicVer2
from modules.const import WIGHTS_DIR, SEGNET_DATASETS_PATH
from modules.segnet.utils.dataloader import one_img_getitem
import cv2
import matplotlib.pyplot as plt
import colorsys


class SegnetInference:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rootpath = f"{SEGNET_DATASETS_PATH}/train"

        self.color_mean = (0.50587555, 0.55623757, 0.4438057)
        self.color_std = (0.19117119, 0.17481992, 0.18417963)
        self.input_size = (160, 120)

        self.state_dict = torch.load(f"{WIGHTS_DIR}/segnetbasic_10000.pth")
        self.net = SegNetBasicVer2()
        self.net = self.net.to(self.device)
        self.net.load_state_dict(self.state_dict)
        self.net.eval()


    def predict(self, img):
        img = one_img_getitem(img, self.input_size, self.color_mean, self.color_std)
        img = img.to(self.device)
        outputs, outputs_color = self.net(img)
        y = outputs  # AuxLoss側は無視
        y_c = outputs_color
        # print(y)
        y = y[0].cpu().detach().numpy().transpose((1, 2, 0))
        y = np.squeeze(y, -1) # (120, 160, 1)から(120, 160)に次元削減
        y_c = y_c[0].cpu().detach().numpy().transpose((1, 2, 0))
        return y, y_c

class GraspCandidate:
    def __init__(self, hand_radius_px, finger_radius_px, angle, depth,
                 center, insertion_points):
        self.hand_radius_px = hand_radius_px
        self.finger_radius_px = finger_radius_px
        self.angle = angle
        self.center = center
        self.center_d = depth[center[1], center[0]]
        self.insertion_points = insertion_points

        self.is_valid = True
        self.is_framein = False
        self.shifted_center = None
        self.total_score = 1 # TODO
        self.debug_infos = []


    def get_center_uv(self) -> ImagePointUV:
        return self.center
    
    def get_insertion_points_uv(self) -> List[ImagePointUV]:
        return self.insertion_points




class GraspDetector:
    # TODO: hand_radiusの追加
    def __init__(self, finger_num: int, hand_radius_mm: Mm, finger_radius_mm: Mm, unit_angle: int,
                 frame_size: Tuple[int, int], fp: float,
                 elements_th: float = 0., center_diff_th: float = 0.,
                 el_insertion_th: float = 0.5, el_contact_th: float = 0.5, el_bw_depth_th: float = 0.,
                 augment_anchors: bool = False, angle_for_augment: int = 15):
        self.finger_num = finger_num
        self.unit_angle = unit_angle  # 生成される把持候補の回転の刻み角
        self.elements_th = elements_th
        self.center_diff_th = center_diff_th
        self.el_insertion_th = el_insertion_th
        self.el_contact_th = el_contact_th
        self.el_bw_depth_th = el_bw_depth_th
        self.augment_anchors = augment_anchors
        self.angle_for_augment = angle_for_augment

        # NOTE: mm, pxの変換は深度、解像度、受光素子のサイズに依存するのでdetect時に変換
        self.hand_radius_mm = hand_radius_mm
        self.finger_radius_mm = finger_radius_mm

        self.h, self.w = frame_size
        self.fp = fp

        self.base_angle = 360 // self.finger_num  # ハンドの指間の角度 (等間隔前提)
        self.candidate_element_num = 360 // self.unit_angle
        self.candidate_num = self.base_angle // self.unit_angle

        self.base_rmat = self._compute_rmat(self.base_angle)
        self.unit_rmat = self._compute_rmat(self.unit_angle)

        self.segnet = SegnetInference()

    def _compute_rmat(self, angle):
        """
        回転行列を生成
        """
        cos_rad, sin_rad = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        return np.array([[cos_rad, -sin_rad], [sin_rad, cos_rad]])

    def _convert_mm_to_px(self, v_mm: Mm, d: Mm) -> Px:
        v_px = (v_mm / d) * self.fp  # (v_mm / 1000) * self.fp / (d / 1000)
        return v_px

    def _compute_rotated_points(self, center: ImagePointUV, base_v: np.ndarray, angle: int):
        # anchorsの計算時は事前にrmatは計算
        rmat = self._compute_rmat(angle)
        finger_v = base_v
        rotated_points = []
        for _ in range(360 // angle):
            rotated_points.append(
                tuple(np.int0(np.round(center + finger_v))))
            finger_v = np.dot(finger_v, rmat)
        return rotated_points

    def compute_insertion_points(self, center: ImagePointUV, base_finger_v: np.ndarray):
        return self._compute_rotated_points(center, base_finger_v, self.base_angle)

    def detect(self, img: Image, depth: Image, instance_msg: Instance) -> List[GraspCandidate]:
        # 単位変換
        center = np.array(instance_msg.center)
        center_d = depth[center[1], center[0]]
        # center_dが欠損すると0 divisionになるので注意
        hand_radius_px = self._convert_mm_to_px(self.hand_radius_mm, center_d)
        finger_radius_px = self._convert_mm_to_px(self.finger_radius_mm, center_d)
        # ベクトルははじめの角度求めるとかで関数内部で計算してもいいかも
        unit_vector = np.array([0, -1])
        base_finger_v = unit_vector * hand_radius_px  # 単位ベクトル x ハンド半径

        y, y_c = self.segnet.predict(img)
        mask = y * instance_msg.mask

        sum_value = np.array([0.0, 0.0, 0.0])
        cnt = np.array([0.0, 0.0, 0.0])
        for i in range(120):
            for j in range(160):
                sum_value += y_c[i][j] * mask[i][j]
                cnt += mask[i][j]
        sum_value /= cnt
        hi, si, vi = colorsys.rgb_to_hsv(sum_value[0], sum_value[1], sum_value[2])
        #     print(hi, si, vi)
        anglei = hi * 90
        print(f'推論 : {anglei}')

        finger_v = np.dot(base_finger_v, self._compute_rmat(anglei)) # TODO 角度
        insertion_points = self.compute_insertion_points(center, finger_v)
        candidate = GraspCandidate(hand_radius_px=hand_radius_px, finger_radius_px=finger_radius_px, angle=anglei,
                                depth=depth, center=center, insertion_points=insertion_points)

        return candidate
