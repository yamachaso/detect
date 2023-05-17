#!/usr/bin/env python3
from typing import Tuple

import numpy as np
from cv_bridge import CvBridge
from detect.msg import Instance, RotatedBoundingBox
from entities.predictor import PredictResult
from modules.ros.utils import PointProjector, numpy2multiarray
from std_msgs.msg import Int32MultiArray


class InstanceHandler:
    bridge = CvBridge()

    @classmethod
    def from_predict_result(cls, instances: PredictResult, index: int) -> Instance:
        return Instance(
            label=str(instances.labels[index]),
            score=instances.scores[index],
            bbox=RotatedBoundingBox(*instances.bboxes[index]),
            center=instances.centers[index],
            area=instances.areas[index],
            mask=cls.bridge.cv2_to_imgmsg(instances.masks[index]),
            contour=numpy2multiarray(
                Int32MultiArray, instances.contours[index])
        )


class RotatedBoundingBoxHandler:
    def __init__(self, msg: RotatedBoundingBox):
        self.msg = msg

        self.ul = np.array(self.msg.upper_left, dtype=np.int32)
        self.ur = np.array(self.msg.upper_right, dtype=np.int32)
        self.ll = np.array(self.msg.lower_left, dtype=np.int32)
        self.lr = np.array(self.msg.lower_right, dtype=np.int32)

    def tolist(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        return (self.msg.upper_left, self.msg.upper_right, self.msg.lower_right, self.msg.lower_left)

    def get_sides_2d(self) -> Tuple[float, float]:
        """ bboxの短辺と長辺の画像平面上での長さ[pixel]を算出する"""
        side_h = np.linalg.norm((self.ul - self.ur)) / 2
        side_v = np.linalg.norm((self.ul - self.ll)) / 2
        short_side = min(side_h, side_v)
        long_side = max(side_h, side_v)

        return short_side, long_side

    def get_sides_3d(self, projector: PointProjector, depth: np.ndarray) -> Tuple[float, float]:
        """ bboxの短辺と長辺の3次元空間上での長さ[mm]を算出する"""
        # z方向は同じとして考えてmm単位の長さ出すべきだった
        # あるいはpx->mm変換行う？
        # short_side_3d = projector.get_length_between_2d_points(
        #     self.ul, self.ur, depth)
        # long_side_3d = projector.get_length_between_2d_points(
        #     self.ul, self.ll, depth)
        short_side_3d = projector.get_flat_length_between_2d_points(
            self.ul, self.ur, depth)
        long_side_3d = projector.get_flat_length_between_2d_points(
            self.ul, self.ll, depth)

        return short_side_3d, long_side_3d
