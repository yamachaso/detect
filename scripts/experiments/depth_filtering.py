# %%
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from modules.const import SAMPLES_PATH
from modules.image import extract_flont_img as new_extract_flont_img
from scipy.signal import argrelmax, argrelmin, savgol_filter
from utils import imshow, load_py2_pickle

# %%
path_list = sorted(glob(f"{SAMPLES_PATH}/saved_data/*"))
path = path_list[0]

data = load_py2_pickle(path)

img = data["img"]
depth = data["depth"]
objects = data["objects"]

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[1].imshow(depth, cmap="binary")

# %%
global_min_depth, global_max_depth = depth.min(), depth.max()

merged_mask = np.where(np.sum([obj["mask"] for obj in objects], axis=0) > 0, 255, 0).astype("uint8")
values_on_merged_mask = depth[merged_mask > 0]
objects_min_depth, objects_max_depth = values_on_merged_mask.min(), values_on_merged_mask.max()
print("global  range:", global_min_depth, global_max_depth)
print("objects range:", objects_min_depth, objects_max_depth)

# %%
uint16_max = 2**16
hist_values = cv2.calcHist([depth], channels=[0], mask=merged_mask, histSize=[uint16_max], ranges=[0, uint16_max - 1])
# hist_values = cv2.calcHist([depth], channels=[0], histSize=[uint16_max], ranges=[0, uint16_max - 1])
print(hist_values.shape)
plt.plot(hist_values)
# %%
hist_values_with_mask = cv2.calcHist([depth], channels=[0], mask=merged_mask, histSize=[uint16_max], ranges=[0, uint16_max - 1])
hist_values_without_mask = cv2.calcHist([depth], channels=[0], mask=None, histSize=[uint16_max], ranges=[0, uint16_max - 1])
fig, axes = plt.subplots(2, 2)
# maskなしヒストグラム
axes[0][0].plot(hist_values_without_mask)
axes[0][0].set_xlim(global_min_depth, global_max_depth)
axes[0][1].plot(hist_values_without_mask)
axes[0][1].set_xlim(objects_min_depth, objects_max_depth)
# maskありヒストグラム
axes[1][0].plot(hist_values_with_mask)
axes[1][0].set_xlim(global_min_depth, global_max_depth)
axes[1][1].plot(hist_values_with_mask)
axes[1][1].set_xlim(objects_min_depth, objects_max_depth)
fig.show()

# %%
# optimal_depth_threshの計算はmaskありhistから計算する
hist = hist_values_with_mask
min_v = objects_min_depth
max_v = objects_max_depth
n = 3

h_list = np.array([])
for i in range(min_v, max_v + 1):
    t1 = np.sum(hist[i - n:i + n + 1])
    t2 = np.sum(hist[i - n * 2:i - n])
    t3 = np.sum(hist[i + n + 1:i + n * 2 + 1])
    res = t1 - t2 - t3
    h_list = np.append(h_list, res)
sorted_h = np.argsort(h_list) + min_v  # argsortはデフォルト昇順
optimal_depth_thresh = sorted_h[-1]

plt.xlim(min_v, max_v)
plt.plot(hist)
plt.vlines(optimal_depth_thresh, 0, hist.max(), color="red")

# %%
print(optimal_depth_thresh)
flont_mask = np.where(depth <= optimal_depth_thresh, 255, 0).astype("uint8")
flont_img = cv2.bitwise_and(img, img, mask=flont_mask)
imshow(flont_img)
# %%
# ヒストグラム平滑化して極値をみつけてもいいかも
filtered_y = savgol_filter(hist[:, 0], 10, 1)
plt.plot(filtered_y)
plt.xlim(min_v, max_v)
for x in argrelmax(filtered_y):
    print(x)
    plt.plot(x, filtered_y[x], "ro", color="orange")
for x in argrelmin(filtered_y):
    print(x)
    plt.plot(x, filtered_y[x], "ro", color="pink")
# %%
flont_mask = np.where(depth <= 990, 255, 0).astype("uint8")
flont_img = cv2.bitwise_and(img, img, mask=flont_mask)
imshow(flont_img)

# %%
# depthのヒストグラムそのままではだめ
plt.xlim(min_v, max_v)
plt.plot(hist.max() - hist)
# %%


def trasnform_ddi(depth, n):
    mask = np.ones((n, n)).astype('uint8')  # erodeで使用するmaskはuint8
    # mask[n//2, n//2] = 0
    mask[1:-1, 1:-1] = 0  # 外周部以外は０に
    depth_min = cv2.erode(depth, mask, iterations=1)  # 最小値フィルタリング
    ddi = np.abs(depth.astype('int32') -
                 depth_min.astype('int32')).astype('uint16')
    return ddi


ddi = trasnform_ddi(depth, 5)
imshow(ddi)
# %%
global_min_ddi, global_max_ddi = ddi.min(), ddi.max()
ddi_values_on_merged_mask = ddi[merged_mask > 0]
objects_min_ddi, objects_max_ddi = ddi_values_on_merged_mask.min(), ddi_values_on_merged_mask.max()
print("global  range:", global_min_ddi, global_max_ddi)
print("objects range:", objects_min_ddi, objects_max_ddi)

hist_values_with_mask_ddi = cv2.calcHist([ddi], channels=[0], mask=merged_mask, histSize=[uint16_max], ranges=[0, uint16_max - 1])
hist_values_without_mask_ddi = cv2.calcHist([ddi], channels=[0], mask=None, histSize=[uint16_max], ranges=[0, uint16_max - 1])
fig, axes = plt.subplots(2, 2)
# maskなしヒストグラム
axes[0][0].plot(hist_values_without_mask_ddi)
axes[0][0].set_xlim(global_min_ddi, global_max_ddi)
axes[0][1].plot(hist_values_without_mask_ddi)
axes[0][1].set_xlim(objects_min_ddi, objects_max_ddi)
# maskありヒストグラム
axes[1][0].plot(hist_values_with_mask_ddi)
axes[1][0].set_xlim(global_min_ddi, global_max_ddi)
axes[1][1].plot(hist_values_with_mask_ddi)
axes[1][1].set_xlim(objects_min_ddi, objects_max_ddi)
fig.show()


# %%
hist = hist_values_without_mask_ddi
min_v = objects_min_ddi
max_v = objects_max_ddi
n = 3

h_list = np.array([])
for i in range(min_v, max_v + 1):
    t1 = np.sum(hist[i - n:i + n + 1])
    t2 = np.sum(hist[i - n * 2:i - n])
    t3 = np.sum(hist[i + n + 1:i + n * 2 + 1])
    res = t1 - t2 - t3
    h_list = np.append(h_list, res)
sorted_h = np.argsort(h_list)  # argsortはデフォルト昇順
optimal_ddi_thresh = sorted_h[-1] + min_v
print(optimal_ddi_thresh)

plt.xlim(min_v, max_v)
plt.plot(hist)
plt.vlines(optimal_ddi_thresh, 0, hist.max(), color="red")

# %%
# マスク内のピクセル欠損がやや気になるか？
optimal_depth_thresh_2 = np.mean(depth[ddi <= optimal_ddi_thresh])
print(optimal_depth_thresh_2)
flont_mask = np.where(depth <= optimal_depth_thresh_2, merged_mask, 0).astype("uint8")
# ピクセル欠損の補完
closing_flont_mask = cv2.morphologyEx(flont_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
# 冗長かもしれないが膨張によってはみ出たピクセルの除去
final_flont_mask = np.where(merged_mask > 0, closing_flont_mask, 0)

# 白い箇所が最終的なマスク
imshow(np.dstack([flont_mask, closing_flont_mask, final_flont_mask]))

# %%
flont_img = cv2.bitwise_and(img, img, mask=final_flont_mask)
imshow(flont_img)
# %%


def compute_optimal_depth_thresh_using_ddi(depth, n):
    # ddiヒストグラムからddiしきい値を算出（物体のエッジに相当）
    ddi = trasnform_ddi(depth, n)
    hist_without_mask = cv2.calcHist([ddi], channels=[0], mask=None, histSize=[uint16_max], ranges=[0, uint16_max - 1])
    min_ddi, max_ddi = ddi.min(), ddi.max()
    h_list = []
    for i in range(min_ddi, max_ddi + 1):
        t1 = np.sum(hist_without_mask[i - n:i + n + 1])
        t2 = np.sum(hist_without_mask[i - n * 2:i - n])
        t3 = np.sum(hist_without_mask[i + n + 1:i + n * 2 + 1])
        res = t1 - t2 - t3
        h_list.append(res)
    sorted_h = np.argsort(h_list)  # argsortはデフォルト昇順
    optimal_ddi_thresh = sorted_h[-1] + min_ddi
    # ddiしきい値をdepthしきい値に変換
    optimal_depth_thresh = np.mean(depth[ddi <= optimal_ddi_thresh])

    return optimal_depth_thresh


def extract_flont_img(img, mask, n):
    optimal_depth_thresh = compute_optimal_depth_thresh_using_ddi(depth, n)
    flont_mask = np.where(depth <= optimal_depth_thresh, mask, 0).astype("uint8")
    # 欠損ピクセルの補完
    closing_flont_mask = cv2.morphologyEx(flont_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # 膨張によりはみ出したピクセルの除去
    final_flont_mask = np.where(mask > 0, closing_flont_mask, 0)
    result_img = cv2.bitwise_and(img, img, mask=final_flont_mask)

    return result_img


fig, axes = plt.subplots(1, 2)
axes[0].imshow(extract_flont_img(img, merged_mask, 5))
axes[1].imshow(extract_flont_img(depth, merged_mask, 5))


# %%
imshow(new_extract_flont_img(img, depth, merged_mask, 5))
# %%
""" depth threshのヒストグラムから直接もとめた閾値から前景画像を得る手法の改良版 """
# インスタンスセグメンテーションの結果への依存度が高い
optimal_depth_thresh = 989
merged_mask = np.zeros_like(depth)
for obj in objects:
    mask = obj["mask"]
    masked_depth = depth[mask > 0]
    valid_num = len(masked_depth[masked_depth <= optimal_depth_thresh])
    ratio = valid_num / len(masked_depth)
    print(valid_num, ratio)
    if ratio > 0.3:
        merged_mask += mask
merged_mask = np.where(merged_mask > 0, 255, 0).astype("uint8")
imshow(merged_mask)
imshow(cv2.bitwise_and(img, img, mask=merged_mask))
# %%
