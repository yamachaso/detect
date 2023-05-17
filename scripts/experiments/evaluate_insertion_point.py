# %%
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from modules.const import SAMPLES_PATH
from modules.visualize import get_color_by_score
from utils import imshow, load_py2_pickle

# %%
path_list = sorted(glob(f"{SAMPLES_PATH}/saved_data/*"))
print(os.getcwd())
print(path_list)
# %%
path = path_list[0]
print(path)
data = load_py2_pickle(path)

img = data["img"]
depth = data["depth"]
objects = data["objects"]

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[1].imshow(depth, cmap="binary")
# %%
normalized_depth = (depth.copy() - depth.min()) / (depth.max() - depth.min())

grad_img = ((1 - normalized_depth[:, :, np.newaxis]) * 255).astype("uint8")
heatmap_img = cv2.applyColorMap(grad_img, cv2.COLORMAP_JET)

base_img = cv2.addWeighted(img, 1, heatmap_img, 0.2, 0)
imshow(base_img)

# %%
masks = []
contours = []
centers = []
candidates_list = []
for obj in objects:
    masks.append(obj["mask"])
    contours.append(obj["contour"])
    centers.append(obj["center"])
    candidates_list.append(obj["candidates"])

# %%
merged_mask = np.where(np.sum(masks, axis=0) > 0, 255, 0)
imshow(merged_mask)
# %%
contour_img = base_img.copy()
cv2.drawContours(contour_img, contours, -1, (255, 255, 0), -1)
alpha = 0.1
overlay_img = cv2.addWeighted(base_img, 1 - alpha, contour_img, alpha, 0)
cv2.drawContours(overlay_img, contours, -1, (255, 255, 0), 2)
imshow(overlay_img)
# %%
candidate_img = overlay_img.copy()
target_index = 3
target_candidate_index = 1

target_candidates = candidates_list[target_index]
target_center = centers[target_index]
for i, points in enumerate(target_candidates):
    is_target_candidate = target_candidate_index == i
    color = (240, 160, 80) if is_target_candidate else (0, 0, 255)
    thickness = 2 if is_target_candidate else 1
    for edge in points:
        cv2.line(candidate_img, target_center, np.int0(edge), color, thickness, cv2.LINE_AA)

for i in range(len(centers)):
    center = centers[i]
    cv2.circle(candidate_img, center, 3, (255, 0, 0), -1)
    cv2.putText(candidate_img, str(i), (center[0] + 5, center[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)


imshow(candidate_img)
# %%
# about the third instance
flatten_points = sum(target_candidates, [])
print(len(flatten_points))
print(flatten_points)

depth_list = [depth[uv[::-1]] for uv in flatten_points]
print(depth_list)
fig, ax = plt.subplots()
ax.set_xlim(0, len(flatten_points) - 1)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.plot(depth_list)
# %%
# insertion pointのdepthによる判定
deep_points = [flatten_points[i] for i in range(len(flatten_points)) if depth_list[i] > 1060]
print(len(deep_points))

candidate_img_2 = candidate_img.copy()
for pt in deep_points:
    cv2.circle(candidate_img_2, pt, 3, (0, 255, 0), -1)
    cv2.circle(candidate_img_2, pt, 10, (0, 100, 100), 1, cv2.LINE_AA)
imshow(candidate_img_2)

# %%
print(target_center)
print(target_candidates)
target_candidate = target_candidates[target_candidate_index]
print(target_candidate)

edge_xy = target_candidate[0][::-1]
radius = 10
# 画面端だとエラーでそう
x_slice = slice(edge_xy[0] - radius, edge_xy[0] + radius + 1)
y_slice = slice(edge_xy[1] - radius, edge_xy[1] + radius + 1)
cropped_img = candidate_img_2[x_slice, y_slice]
imshow(cropped_img)
# %%
cropped_depth = depth[x_slice, y_slice]
finger_mask = np.zeros_like(cropped_depth, dtype=np.uint8)
cv2.circle(finger_mask, (cropped_depth.shape[0] // 2, cropped_depth.shape[1] // 2), radius, 255, -1)
imshow(finger_mask)
depth_values_in_mask = cropped_depth[finger_mask == 255]
print("unique values", np.unique(depth_values_in_mask))
print("mean value", np.mean(depth_values_in_mask))

# %%


def compute_depth_profile_in_finger_area(depth, pt_xy, radius):
    x_slice = slice(pt_xy[0] - radius, pt_xy[0] + radius + 1)
    y_slice = slice(pt_xy[1] - radius, pt_xy[1] + radius + 1)
    cropped_depth = depth[x_slice, y_slice]
    finger_mask = np.zeros_like(cropped_depth, dtype=np.uint8)
    cv2.circle(finger_mask, (cropped_depth.shape[0] // 2, cropped_depth.shape[1] // 2), radius, 255, -1)
    depth_values_in_mask = cropped_depth[finger_mask == 255]
    return int(np.min(depth_values_in_mask)), int(np.max(depth_values_in_mask)), int(np.mean(depth_values_in_mask))


print(compute_depth_profile_in_finger_area(depth, edge_xy[::-1], 10))
# %%
# merged_contour = sum(contours, [])
# cv2.boundingRect(contours[0])
h_indexes, w_indexes = np.where(merged_mask > 0)
objects_bbox_x_min, objects_bbox_x_max = w_indexes.min(), w_indexes.max()
objects_bbox_y_min, objects_bbox_y_max = h_indexes.min(), h_indexes.max()
objects_bbox_ul = (objects_bbox_x_min, objects_bbox_y_min)
objects_bbox_lr = (objects_bbox_x_max, objects_bbox_y_max)
print(objects_bbox_ul, objects_bbox_lr)

tmp_img = img.copy()
cv2.rectangle(tmp_img, objects_bbox_ul, objects_bbox_lr, (255, 0, 0), 2, cv2.LINE_AA)
imshow(tmp_img)

# %%
radius = 10
cnt = 0
mean_depth_list = []
scores = []

# objects_min_depthとobjects_max_depthはdepth filteringにも使う
# objects_min_depth = depth[objects_bbox_x_min:objects_bbox_x_max + 1, objects_bbox_y_min:objects_bbox_y_max + 1].min()

target_mask = masks[target_index]
instance_min_depth = depth[target_mask > 0].min()  # mask is boolean array
objects_max_depth = depth[objects_bbox_x_min:objects_bbox_x_max + 1, objects_bbox_y_min:objects_bbox_y_max + 1].max()
print(instance_min_depth, objects_max_depth)

for points in target_candidates:
    for u, v in points:
        min_depth, max_depth, mean_depth = compute_depth_profile_in_finger_area(depth, (v, u), radius)
        mean_depth_list.append(mean_depth)
        # 正規化しないと最大値、最小値の間の差が小さい -> ２乗
        # ただcandidate個別の最小値、最大値をつかってmin-max正規化してしまうとcandidate間での基準が変わってしまうのでは？
        # →　一旦ここではcandidate内での比較にとどめておきつつ、オリジナルな値を保存しておいてあとでひかくとかでもいいかも
        # 先に全体マスク領域+アルファ(explode)の領域から仮の最大・最小をもとめてもいいかも
        # 逆にcandidateごとにしたほうがdepthに偏りあるとき対応できるのでは？（片側だけ残ってる場合とか）
        # TOFIX: 最大・最小の差がないときに最も大きな値となってしまう → centerのdepthを最小値にすればいい気がする
        score = ((mean_depth - instance_min_depth) / (objects_max_depth - instance_min_depth + 1e-6)) ** 2
        scores.append(score)
        cnt += 1

print(mean_depth_list)
print(scores)

# 指の範囲も含めたinsertion pointの深さ判定
# 実際にはスコアとして計算
deep_points_2 = [flatten_points[i] for i in range(len(flatten_points)) if mean_depth_list[i] >= 1076]

candidate_img_3 = candidate_img.copy()
for pt in deep_points_2:
    cv2.circle(candidate_img_3, pt, 3, (0, 255, 0), -1)
    cv2.circle(candidate_img_3, pt, 10, (0, 100, 100), 1, cv2.LINE_AA)
imshow(candidate_img_3)
# %%
# 指の範囲も含めたinsertion pointの深さ判定
print(scores)
deep_points_3 = [flatten_points[i] for i in range(len(flatten_points)) if scores[i] >= 0.95]

candidate_img_4 = candidate_img.copy()
for pt in deep_points_3:
    cv2.circle(candidate_img_4, pt, 3, (0, 255, 0), -1)
    cv2.circle(candidate_img_4, pt, 10, (0, 100, 100), 1, cv2.LINE_AA)
imshow(candidate_img_4)
# %%
# 理想的にはinsertion pointとcontact pointの組み合わせからなるパーツで評価したほうがいいが、
# 組み合わせ角度が一定の場合は先に組み合わせ時のスコア評価してフィルタリングしたほうが計算減ってよいかも
# → そもinsertion pointのスコアが低い場合contact point計算しなくてよい
print(target_candidates)
finger_num = len(target_candidate)
candidates_scores = [np.prod(scores[i:i + 4]) for i in range(0, len(scores), finger_num)]
print(candidates_scores)

best_candidate = target_candidates[np.argmax(candidates_scores)]
print(best_candidate)

candidate_img_5 = candidate_img.copy()
for edge in best_candidate:
    cv2.line(candidate_img_5, target_center, np.int0(edge), (255, 0, 0), thickness, cv2.LINE_AA)

imshow(candidate_img_5)

# %%


def insertion_point_score(min_depth, max_depth, mean_depth):
    return ((mean_depth - min_depth) / (max_depth - instance_min_depth + 1e-6)) ** 2


def evaluate_single_insertion_point(depth, pt_xy, radius, min_depth, max_depth):
    _, _, mean_depth = compute_depth_profile_in_finger_area(depth, pt_xy, radius)
    score = insertion_point_score(min_depth, max_depth, mean_depth)
    return score


def evaluate_insertion_points(depth, candidates, radius, min_depth, max_depth):
    scores = []
    for points in candidates:
        for u, v in points:
            score = evaluate_single_insertion_point(depth, (v, u), radius, min_depth, max_depth)
            scores.append(score)

    return scores


def evaluate_single_insertion_points_set(insertion_point_scores):
    """ 同じのcandidateに所属するinsertion_pointのスコアの積 """
    return np.prod(insertion_point_scores)


def evaluate_insertion_points_set(depth, candidates, radius, min_depth, max_depth):
    scores = evaluate_insertion_points(depth, candidates, radius, min_depth, max_depth)
    finger_num = len(candidates[0])
    candidates_scores = [evaluate_single_insertion_points_set(scores[i:i + finger_num]) for i in range(0, len(scores), finger_num)]

    return candidates_scores


# %%
# candidates_list全体に対する評価と可視化
candidate_img_6 = img.copy()
for i, obj in enumerate(objects):
    candidates = obj["candidates"]
    mask = obj["mask"]
    center = obj["center"]
    instance_min_depth = depth[mask > 0].min()
    scores = evaluate_insertion_points_set(depth, candidates, radius, instance_min_depth, objects_max_depth)
    best_index = np.argmax(scores)
    best_score = scores[best_index]
    best_candidate = candidates[best_index]
    print(i, instance_min_depth, best_score)
    color = get_color_by_score(best_score)

    for edge in best_candidate:
        cv2.line(candidate_img_6, center, np.int0(edge), color, 2, cv2.LINE_AA)

for i, center in enumerate(centers):
    cv2.putText(candidate_img_6, f"{i}: {best_score:.1f}", (center[0] + 5, center[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)

imshow(candidate_img_6)

# %%
