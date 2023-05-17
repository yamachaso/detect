from colorsys import hsv_to_rgb

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from modules.const import UINT16MAX


def gen_color_palette(n):
    hsv_array = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    rgb_array = np.array(list(
        map(lambda x: [int(v * 255) for v in hsv_to_rgb(*x)], hsv_array)), dtype=np.uint8)
    return rgb_array


def transform_ddi(depth, n):
    mask = np.ones((n, n)).astype('uint8')  # erodeで使用するmaskはuint8
    # mask[n//2, n//2] = 0
    mask[1:-1, 1:-1] = 0  # 外周部以外は０に
    depth_min = cv2.erode(depth, mask, iterations=1)  # 最小値フィルタリング
    ddi = np.abs(depth.astype('int32') -
                 depth_min.astype('int32')).astype('uint16')
    return ddi


def compute_thresh_by_histgram(hist, min_v, max_v, n):
    h_list = []
    for i in range(min_v, max_v + 1):
        t1 = np.sum(hist[i - n:i + n + 1])
        t2 = np.sum(hist[i - n * 2:i - n])
        t3 = np.sum(hist[i + n + 1:i + n * 2 + 1])
        res = t1 - t2 - t3
        h_list.append(res)
    sorted_h = np.argsort(h_list)  # argsortはデフォルト昇順
    optimal_thresh = sorted_h[-1] + min_v
    return optimal_thresh


def compute_optimal_ddi_thresh(ddi, whole_mask, n):
    ddi_hist_wo_mask = cv2.calcHist([ddi], channels=[0], mask=whole_mask, histSize=[
        UINT16MAX], ranges=[0, UINT16MAX - 1])
    ddi_values_on_mask = ddi[whole_mask > 0]
    min_ddi, max_ddi = ddi_values_on_mask.min(), ddi_values_on_mask.max()
    optimal_ddi_thresh = compute_thresh_by_histgram(
        ddi_hist_wo_mask, min_ddi, max_ddi, n)
    return optimal_ddi_thresh


def compute_optimal_depth_thresh(depth, whole_mask, n):
    """前景抽出のためのdepthしきい値を算出する"""
    # depthからddiへの変換（マスクの範囲外は平均値で補完）
    ddi = transform_ddi(np.where(whole_mask > 0, depth,
                        depth[depth > 0].mean()), n)
    # ddiヒストグラムからddiしきい値を算出（物体のエッジに相当）
    optimal_ddi_thresh = compute_optimal_ddi_thresh(ddi, whole_mask, n)
    # ddiしきい値が一定以上のdepthを取得（物体境界に該当）
    edge_mask = np.where(ddi > optimal_ddi_thresh, 255, 0).astype("uint8")
    edge_depth = np.where(edge_mask > 0, depth, 0)
    # エッジ部分のdepthヒストグラムから前傾抽出に適したdepthしきい値を算出
    depth_hist_wo_mask = cv2.calcHist([depth], channels=[0],
                                      mask=edge_mask, histSize=[UINT16MAX],
                                      ranges=[0, UINT16MAX - 1])
    min_depth, max_depth = edge_depth.min(), edge_depth.max()
    optimal_depth_thresh = compute_thresh_by_histgram(
        depth_hist_wo_mask, min_depth, max_depth, n)
    rounded_optimal_depth_thresh = np.int0(np.round(optimal_depth_thresh))

    return rounded_optimal_depth_thresh


def refine_flont_mask(whole_flont_mask, instance_masks, thresh=0.8):
    res_mask = np.zeros_like(whole_flont_mask)
    for mask in instance_masks:
        overlay = np.where(whole_flont_mask > 0, mask, 0)
        score = len(overlay[overlay > 0]) / len(mask[mask > 0])
        if score <= thresh:
            continue
        res_mask += mask
    res_mask = np.where(res_mask > 0, 255, 0).astype("uint8")
    return res_mask


def merge_mask(instance_masks):
    return np.where(np.sum(instance_masks, axis=0) > 0, 255, 0).astype("uint8")


def extract_flont_instance_indexes(whole_flont_mask, instance_masks, thresh=0.8):
    flont_indexes = []
    for i, mask in enumerate(instance_masks):
        overlay = np.where(whole_flont_mask > 0, mask, 0)
        score = len(overlay[overlay > 0]) / len(mask[mask > 0])
        if score > thresh:
            flont_indexes.append(i)
    return flont_indexes


def extract_flont_mask_with_thresh(depth, thresh, n):
    # flont_mask = np.where(depth <= thresh, whole_mask, 0).astype("uint8")
    flont_mask = np.where(depth <= thresh, 255, 0).astype("uint8")
    # 欠損ピクセルの補完
    closing_flont_mask = cv2.morphologyEx(
        flont_mask, cv2.MORPH_CLOSE, np.ones((n, n), np.uint8))
    # 膨張によりはみ出したピクセルの除去
    # final_flont_mask = np.where(whole_mask > 0, closing_flont_mask, 0)
    final_flont_mask = closing_flont_mask

    return final_flont_mask


def extract_flont_img(img, depth, whole_mask, n):
    optimal_depth_thresh = compute_optimal_depth_thresh(depth, whole_mask, n)
    flont_mask = extract_flont_mask_with_thresh(depth, optimal_depth_thresh, n)
    result_img = cv2.bitwise_and(img, img, mask=flont_mask)

    return result_img


def extract_depth_between_two_points(depth, p1, p2, mode="nearest", order=1):
    n = np.int0(np.round(np.linalg.norm(np.array(p1) - np.array(p2), ord=2)))
    h, w = np.linspace(p1[1], p2[1], n), np.linspace(p1[0], p2[0], n)
    res = map_coordinates(depth, np.vstack((h, w)), mode=mode, order=order)
    return res
