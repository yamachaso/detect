# %%
import cProfile
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from modules.const import SAMPLES_PATH
from scipy.ndimage import map_coordinates
from utils import imshow, load_py2_pickle

# %%
img_shape = (1000, 1000)
base_img = np.zeros(img_shape, dtype=np.uint8)
center = (50, 80)
circle_img = base_img.copy()
cv2.circle(circle_img, center, 10, 255, thickness=-1)
plt.imshow(circle_img, cmap="gray")
# %%
contours, hierarchy = cv2.findContours(circle_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
x, y, w, h = cv2.boundingRect(contours[0])
upper_left_point = np.array((x, y))
print(x, y, w, h)
print(contours)
shifted_contours = [contour - upper_left_point for contour in contours]
cropped_base_img = base_img[y:y + h, x:x + w]
cnt_img = cropped_base_img.copy()
cv2.drawContours(cnt_img, shifted_contours, -1, 255)
plt.imshow(cnt_img, cmap="gray")
# %%
line_img = cropped_base_img.copy()
line_pt1 = center
line_pt2 = (80, 20)
shifted_line_pt1 = center - upper_left_point
shifted_line_pt2 = (80, 20) - upper_left_point
line_img = cv2.line(line_img, tuple(shifted_line_pt1), tuple(shifted_line_pt2), 255, 1)
plt.imshow(line_img, cmap="gray")

# %%
bitwise_img = cropped_base_img.copy()
cv2.bitwise_and(cnt_img, line_img, bitwise_img)
plt.imshow(bitwise_img, cmap="gray")
intersection = [(x, y) for x, y in zip(*np.where(bitwise_img > 0))][0]
original_intersection = tuple(intersection + upper_left_point)
print("intersection:", intersection)
print("original intersection:", original_intersection)

# %%


def compute_intersection_between_contour_and_line(img_shape, contour, line_pt1_xy, line_pt2_xy):
    """
    輪郭と線分の交点座標を取得する
    TODO: 線分ごとに描画と論理積をとり非効率なので改善方法要検討
    """
    blank_img = np.zeros(img_shape)
    # クロップ前に計算したcontourをクロップ後の画像座標に変換し描画
    cnt_img = blank_img.copy()
    cv2.drawContours(cnt_img, [contour], -1, 255, 1, lineType=cv2.LINE_AA)
    # クロップ前に計算したlineをクロップ後の画像座標に変換し描画
    line_img = blank_img.copy()
    # 斜めの場合、ピクセルが重ならない場合あるのでlineはthicknessを２にして平均をとる
    line_img = cv2.line(line_img, line_pt1_xy, line_pt2_xy, 255, 2, lineType=cv2.LINE_AA)
    # バイナリ画像(cnt_img, line_img)のbitwiseを用いて、contourとlineの交点を検出
    bitwise_img = blank_img.copy()
    cv2.bitwise_and(cnt_img, line_img, bitwise_img)

    intersections = [(w, h) for h, w in zip(*np.where(bitwise_img > 0))]  # hw to xy
    mean_intersection = np.int0(np.round(np.mean(intersections, axis=0)))
    return mean_intersection


# %%
intersection = compute_intersection_between_contour_and_line(base_img.shape[:2], contours[0], line_pt1, line_pt2)
print("intersection:", intersection)
# %%
# timeitやcProfileの引数にndarray形式が使えない、tupleやlistで渡すと今度は関数内部でエラー
# n = 1000
# cProfile.run(f"compute_intersection_between_contour_and_line({img_shape}, [[[50, 40]]], {line_pt1}, {line_pt2})")
# timeit(f"compute_intersection_between_contour_and_line({img_shape}, contour=[[[50, 40]]], line_pt1={line_pt1}, line_pt2={line_pt2})", number=n)


# ラップするとうまくいく
def wrapper():
    compute_intersection_between_contour_and_line(base_img.shape[:2], contours[0], line_pt1, line_pt2)


# (100, 100)で0.01sec, (1000, 1000)で0.03sec
cProfile.run("wrapper()")


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
contact_img = img.copy()
img_shape = contact_img.shape[:2]
for obj in objects:
    contour = obj["contour"]
    candidates = obj["candidates"]
    center = obj["center"]
    for points in candidates:
        for edge in points:
            intersection = compute_intersection_between_contour_and_line(img_shape, contour, center, edge)
            print("intersection:", intersection)
            cv2.line(contact_img, center, edge, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(contact_img, intersection, 3, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    break

imshow(contact_img)
# %%
target_index = 3
obj = objects[target_index]
contour = obj["contour"]
mask = obj["mask"]
candidates = obj["candidates"]
center = obj["center"]
instance_min_depth = depth[mask > 0].min()

contact_img_2 = img.copy()
for points in candidates:
    for edge in points:
        intersection = compute_intersection_between_contour_and_line(img_shape, contour, center, edge)
        cv2.line(contact_img_2, center, edge, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(contact_img_2, intersection, 3, (255, 0, 0), 1, lineType=cv2.LINE_AA)

cropped_contact_img_2 = contact_img_2[center[1] - 80:center[1] + 80, center[0] - 80:center[0] + 80]
imshow(cropped_contact_img_2)

# %%
# 交点の座標から指の半径分edge側にシフトした点をcontact pointとする
finger_radius = 3
intersection = compute_intersection_between_contour_and_line(img_shape, contour, center, edge)
direction_v = np.array(edge) - np.array(center)
unit_direction_v = direction_v / np.linalg.norm(direction_v, ord=2)
print("unit_direction_v", unit_direction_v)
# 移動後座標 = 移動元座標 + 方向ベクトル x 移動量(指半径[pixel])
contact_point = np.int0(np.round(intersection + unit_direction_v * finger_radius))
print("contact_point", contact_point)

shifted_contact_img_2 = img.copy()
cv2.line(shifted_contact_img_2, center, edge, (0, 0, 255), 1, lineType=cv2.LINE_AA)
cv2.circle(shifted_contact_img_2, edge, 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)
cv2.circle(shifted_contact_img_2, intersection, finger_radius, (255, 0, 0), 1, lineType=cv2.LINE_AA)
cv2.circle(shifted_contact_img_2, contact_point, finger_radius, (0, 255, 0), 1, lineType=cv2.LINE_AA)

cropped_shifted_contact_img_2 = shifted_contact_img_2[center[1] - 80:center[1] + 80, center[0] - 80:center[0] + 80]
imshow(cropped_shifted_contact_img_2)

# %%


def compute_contact_point(contour, center, edge, finger_radius):
    x, y, w, h = cv2.boundingRect(contour)
    upper_left_point = np.array((x, y))
    shifted_contour = contour - upper_left_point
    shifted_center, shifted_edge = [tuple(pt - upper_left_point) for pt in (center, edge)]

    shifted_intersection = compute_intersection_between_contour_and_line((h, w), shifted_contour, shifted_center, shifted_edge)
    intersection = tuple(shifted_intersection + upper_left_point)

    direction_v = np.array(edge) - np.array(center)
    unit_direction_v = direction_v / np.linalg.norm(direction_v, ord=2)
    # 移動後座標 = 移動元座標 + 方向ベクトル x 移動量(指半径[pixel])
    contact_point = np.int0(np.round(intersection + unit_direction_v * finger_radius))

    return contact_point


print(compute_contact_point(contour, center, edge, finger_radius))
# %%
contact_img_3 = img.copy()
for points in candidates:
    for edge in points:

        contact_point = compute_contact_point(contour, center, edge, finger_radius)
        cv2.line(contact_img_3, center, edge, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(contact_img_3, contact_point, 3, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.circle(contact_img_3, edge, 3, (255, 0, 0), 1, lineType=cv2.LINE_AA)

cropped_contact_img_3 = contact_img_3[center[1] - 80:center[1] + 80, center[0] - 80:center[0] + 80]
imshow(cropped_contact_img_3)

# %%
contact_img_4 = img.copy()
for obj in objects:
    contour = obj["contour"]
    mask = obj["mask"]
    candidates = obj["candidates"]
    center = obj["center"]
    instance_min_depth = depth[mask > 0].min()
    for points in candidates:
        for edge in points:
            contact_point = compute_contact_point(contour, center, edge, finger_radius)
            cv2.line(contact_img_4, center, edge, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(contact_img_4, contact_point, 3, (0, 255, 0), 1, lineType=cv2.LINE_AA)
            cv2.circle(contact_img_4, edge, 3, (255, 0, 0), 1, lineType=cv2.LINE_AA)
imshow(contact_img_4)

# %%
# ipとcpの間のdepthを調べる


def extract_depth_between_two_points(depth, p1, p2):
    n = np.int0(np.round(np.linalg.norm(np.array(p1) - np.array(p2), ord=2)))
    h, w = np.linspace(p1[0], p2[0], n), np.linspace(p1[1], p2[1], n)
    res = map_coordinates(depth, np.vstack((h, w)))
    return res


fig, axes = plt.subplots(1, 2)
hist_1 = extract_depth_between_two_points(depth, center, edge)
hist_2 = extract_depth_between_two_points(depth, contact_point, edge)
axes[0].plot(hist_1)
axes[1].plot(hist_2)

max_depth_bw_pts_1 = np.max(hist_1)
max_depth_bw_pts_2 = np.max(hist_2)
mean_depth_bw_pts_1 = np.mean(hist_1)
mean_depth_bw_pts_2 = np.mean(hist_2)
bw_pts_score_1 = (mean_depth_bw_pts_1 - instance_min_depth) / (max_depth_bw_pts_1 - instance_min_depth)
bw_pts_score_2 = (mean_depth_bw_pts_2 - instance_min_depth) / (max_depth_bw_pts_2 - instance_min_depth)
print("max depth:", max_depth_bw_pts_1, max_depth_bw_pts_2)
print("mean depth:", mean_depth_bw_pts_1, mean_depth_bw_pts_2)
print("score:", bw_pts_score_1, bw_pts_score_2)


def compute_bw_depth_score(depth, contact_point, insertion_point, min_depth):
    values = extract_depth_between_two_points(depth, contact_point, insertion_point)
    max_depth = values.max()
    valid_values = values[values > 0]
    mean_depth = np.mean(valid_values) if len(valid_values) > 0 else 0
    score = (mean_depth - min_depth) / (max_depth - min_depth)
    return score


print(compute_bw_depth_score(depth, center, edge, instance_min_depth))


# %%
