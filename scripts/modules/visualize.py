import cv2
import numpy as np


def draw_bbox(img, box, color=(0, 255, 0), **kwargs):
    img = cv2.drawContours(img, [np.array(box)], 0, color, 2, **kwargs)
    return img


def draw_candidate(img, p1, p2, color=(0, 0, 255), is_target=False, target_thickness=2, **kwargs):
    thickness = target_thickness if is_target else 1
    img = cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA, **kwargs)
    return img


def draw_candidates(img, candidates, target_index=None, **kwargs):
    for i, (p1, p2) in enumerate(candidates):
        is_target = i == target_index
        img = draw_candidate(img, p1, p2, is_target, **kwargs)
    return img


def draw_candidates_and_boxes(img, candidates_list, boxes, target_indexes=None, gray=False, **kwargs):
    if gray:
        img = convert_rgb_to_3dgray(img)
    if not target_indexes:
        target_indexes = [None] * len(candidates_list)
    for candidates, box, target_index in zip(candidates_list, boxes, target_indexes):
        img = draw_bbox(img, box, **kwargs)
        img = draw_candidates(img, candidates,
                              target_index=target_index, **kwargs)
    return img


def convert_1dgray_to_3dgray(gray):
    gray_1d = gray[:, :, np.newaxis] if len(gray.shape) == 2 else gray
    gray_3d = cv2.merge([gray_1d] * 3)
    return gray_3d


def convert_rgb_to_3dgray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3d = convert_1dgray_to_3dgray(gray)
    return gray_3d


def get_color_by_score(score):
    coef = (1 - score)
    color = (255, 255 * coef, 255 * coef)
    return color
