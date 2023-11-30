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
def compute_intersection_between_contour_and_line(img_shape, center, contour):
    """
    輪郭と線分の交点座標を取得する
    TODO: 線分ごとに描画と論理積をとり非効率なので改善方法要検討
    """
    (x,y),radius = cv2.minEnclosingCircle(contour)
    radius = int(radius * 2)

    blank_img = np.zeros(img_shape)
    # クロップ前に計算したcontourをクロップ後の画像座標に変換し描画
    cnt_img = blank_img.copy()
    cv2.drawContours(cnt_img, [contour], -1, 255, 1, lineType=cv2.LINE_AA)

    intersection_list = []

    for theta_d in range(0, 360, 15):
        theta_r = np.rad2deg(theta_d)
        u = int(radius * np.cos(theta_r) + center[0])
        v = int(radius * np.sin(theta_r) + center[1])

        # クロップ前に計算したlineをクロップ後の画像座標に変換し描画
        line_img = blank_img.copy()
        # 斜めの場合、ピクセルが重ならない場合あるのでlineはthicknessを２にして平均をとる
        line_img = cv2.line(line_img, center, (u, v), 255, 2, lineType=cv2.LINE_AA)
        # バイナリ画像(cnt_img, line_img)のbitwiseを用いて、contourとlineの交点を検出
        bitwise_img = blank_img.copy()
        cv2.bitwise_and(cnt_img, line_img, bitwise_img)

        print(*np.where(bitwise_img > 0))

        intersections = [(w, h) for h, w in zip(*np.where(bitwise_img > 0))]  # hw to xy
        mean_intersection = np.int0(np.round(np.mean(intersections, axis=0)))
        intersection_list.append(mean_intersection)

    return intersection_list


#%%
def compute_contours_list(img_shape, center, contour):
    """
    輪郭と線分の交点座標を取得する
    TODO: 線分ごとに描画と論理積をとり非効率なので改善方法要検討
    """
    (x,y),radius = cv2.minEnclosingCircle(contour)
    radius = int(radius * 2)

    blank_img = np.zeros(img_shape)
    # クロップ前に計算したcontourをクロップ後の画像座標に変換し描画
    cnt_img = blank_img.copy()
    cv2.drawContours(cnt_img, [contour], -1, 255, 1, lineType=cv2.LINE_AA)

    contours_list = []


    for theta_d in range(0, 360, 15):
        theta_r = np.rad2deg(theta_d)
        u = int(radius * np.cos(theta_r) + center[0])
        v = int(radius * np.sin(theta_r) + center[1])

        # クロップ前に計算したlineをクロップ後の画像座標に変換し描画
        line_img = blank_img.copy()
        # 斜めの場合、ピクセルが重ならない場合あるのでlineはthicknessを２にして平均をとる
        line_img = cv2.line(line_img, center, (u, v), 255, 2, lineType=cv2.LINE_AA)
        # バイナリ画像(cnt_img, line_img)のbitwiseを用いて、contourとlineの交点を検出
        bitwise_img = blank_img.copy()
        cv2.bitwise_and(cnt_img, line_img, bitwise_img)

        intersections = [(w, h) for h, w in zip(*np.where(bitwise_img > 0))]  # hw to xy
        mean_intersection = np.int0(np.round(np.mean(intersections, axis=0)))
        contours_list.append(mean_intersection)

    return contours_list


# %%
contact_img = img.copy()
imshow(contact_img)
img_shape = contact_img.shape[:2]

index = 0

contour = objects[index]["contour"]
candidates = objects[index]["candidates"]
center = objects[index]["center"]

intersection_list = compute_contours_list(img_shape, center, contour)
# print("intersection:", intersection)
for intersection in intersection_list:
    cv2.circle(contact_img, intersection, 3, (255, 0, 0), 1, lineType=cv2.LINE_AA)

imshow(contact_img)