from typing import List, Tuple, Union

import cv2
import numpy as np
from modules.image import gen_color_palette
from torch import Tensor


class IndexedMask(np.ndarray):
    """0: background, 1~n: classes"""
    def __new__(cls, masks: Union[np.ndarray, List, Tuple]):
        masks = np.asarray(masks, dtype=np.uint8)
        assert masks.ndim == 3
        n, h, w = masks.shape
        # cast ndarray to IndexedMask
        self = np.zeros((h, w)).view(cls)
        # 要検証： masksが深度降順であることを前提にしている
        for i, mask in enumerate(masks):
            self[mask != 0] = i + 1

        self.n = n
        self.palette = gen_color_palette(n)
        return self

    def to_rgb(self):
        img = np.zeros((*self.shape, 3), dtype=np.uint8)
        for i in range(self.n):
            img[self == i + 1] = self.palette[i]
        return img


class BinaryMask:
    """
    mask: 1 or 255, (n, h, w)
    contour: used to calculate other values
    """

    def __init__(self, mask: Union[np.ndarray, Tensor]):
        self.mask: np.ndarray = np.asarray(mask, dtype=np.uint8)
        self.contour = self._get_contour()

    # private
    def _get_contour(self):
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            contour = max(contours, key=lambda x: cv2.contourArea(x))
        else:
            contour = contours[0]

        return contour

    def get_center(self):
        mu = cv2.moments(self.contour)
        center = np.array([int(mu["m10"] / mu["m00"]), int(mu["m01"] / mu["m00"])])
        return np.int0(np.rint(center))

    def get_rotated_bbox(self):
        # (upper_left, upper_right, lower_right, lower_left)
        return np.int0(np.rint(cv2.boxPoints(cv2.minAreaRect(self.contour))))

    def get_area(self):
        return cv2.contourArea(self.contour)
