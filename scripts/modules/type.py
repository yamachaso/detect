from typing import Tuple
from numpy import ndarray

Px = float  # mmから変換した場合などは少数値になる
Mm = float
Image = ndarray
ImagePointUV = Tuple[int, int]  # [px, px, mm]
ImagePointUVD = Tuple[ImagePointUV, Mm]  # [px, px, mm]