# %%
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from detectron2.config import get_cfg
from entities.predictor import Predictor
from modules.const import CONFIGS_PATH, OUTPUTS_PATH, SAMPLES_PATH
from modules.grasp import GraspDetector
from modules.image import (compute_optimal_depth_thresh,
                           extract_flont_instance_indexes,
                           extract_flont_mask_with_thresh, merge_mask,
                           transform_ddi)
from modules.visualize import convert_rgb_to_3dgray, get_color_by_score
from utils import RealsenseBagHandler, imshow

# %%
config_path = f"{CONFIGS_PATH}/config.yaml"
weight_path = f"{OUTPUTS_PATH}/mask_rcnn/model_final.pth"
device = "cuda:0"

print(config_path)
print(weight_path)
print(device)

cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = weight_path
cfg.MODEL.DEVICE = device

predictor = Predictor(cfg)
# %%

path = f'{OUTPUTS_PATH}/tmp/2023-09-15-05-00-29/color/1.jpg'
# path = f'{SAMPLES_PATH}/real_images/color/color000.jpg'
img = cv2.imread(path)
print(img.shape)
imshow(img)

# %%
res = predictor.predict(img)
seg = res.draw_instances(img[:, :, ::-1])
imshow(seg)


# %%
height, widht, channel = img.shape
center_h, center_w = height // 2, widht // 2
# %%

point = (center_w, center_h)
index = -1
for i, contour in enumerate(res.contours):
    # 点が輪郭の内側の場合は +1、輪郭の境界線上の場合は 0、輪郭の外側の場合は -1 を返す
    if cv2.pointPolygonTest(contour, point, False) >= 0:
        index = i
        # 特定の点を含む輪郭を見つけた場合の処理を行います
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)  # 輪郭を描画する例
        break
imshow(img)
# %%


ellipse = cv2.fitEllipse(res.contours[index])
print(type(ellipse))
print(ellipse)
cv2.ellipse(img,ellipse,(255,0,0),2)
# cv2.circle(img, (296, 223), 3, (0, 0, 255), 5, lineType=cv2.LINE_AA)

imshow(img)


# %%
import numpy as np
from scipy import optimize

def generate_ellipse(c, ab, angle):
    cx, cy = c
    a, b = ab
    a /= 2
    b /= 2
    angle *= -1
    # 楕円のパラメータを計算
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))

    # 楕円の方程式を返す
    def ellipse_equation(x, y):
        x_rotated = cos_angle * (x - cx) - sin_angle * (y - cy)
        y_rotated = sin_angle * (x - cx) + cos_angle * (y - cy)
        return (x_rotated**2 / a**2) + (y_rotated**2 / b**2) - 1

    return ellipse_equation

# 解きたい関数をリストで戻す
def func(x, t, ellipse):
    px, py, r = x
    ellipse_func = generate_ellipse(ellipse[0], ellipse[1], ellipse[2])
    equations = [
        ellipse_func(px + r * np.cos(t), py - r * np.sin(t)),
        ellipse_func(px + r * np.cos(t + np.pi * 2 / 3), py - r * np.sin(t + np.pi * 2 / 3)),
        ellipse_func(px + r * np.cos(t - np.pi * 2 / 3), py - r * np.sin(t - np.pi * 2 / 3)), 
    ]
    return equations

# 制約を設定
def constraint_func(x):
    return x[2] - 50.0  # r >= 0.001


# 初期値
cons = (
    {'type': 'ineq', 'fun': constraint_func}
)

theta = 3.1415926535 * 2 / 3 / 24 * 9

# 最適化を実行
initial_guess = [ellipse[0][0], ellipse[0][1], 200.0]
result = optimize.minimize(lambda x: np.sum(np.array(func(x, theta, ellipse))**2), initial_guess, constraints=cons, method="SLSQP")
print(result)

# ref : https://qiita.com/imaizume/items/44896c8e1dd0bcbacdd5

# %%
px = result.x[0]
py = result.x[1]
r = result.x[2]

# img_tmp = img.copy()
cv2.circle(img_tmp, (int(px + r * np.cos(theta)), int(py - r * np.sin(theta))), 3, (255, 0, 255), 5, lineType=cv2.LINE_AA)
cv2.circle(img_tmp, (int(px + r * np.cos(theta + np.pi * 2 / 3)), int(py - r * np.sin(theta + np.pi * 2 / 3))), 3, (255, 0, 255), 5, lineType=cv2.LINE_AA)
cv2.circle(img_tmp, (int(px + r * np.cos(theta - np.pi * 2 / 3)), int(py - r * np.sin(theta - np.pi * 2 / 3))), 3, (255, 0, 255), 5, lineType=cv2.LINE_AA)
imshow(img_tmp)


# %%
from typing import List, Tuple
from modules.type import Image, ImagePointUV, ImagePointUVD, Mm, Px
import numpy as np
from scipy import optimize

class InsertionCalculator:
    # TODO: hand_radiusの追加
    def __init__(self, finger_num: int, hand_radius_mm: Mm, finger_radius_mm: Mm, unit_angle: int,
                 frame_size: Tuple[int, int], fp: float):
        self.finger_num = finger_num
        self.unit_angle = unit_angle  # 生成される把持候補の回転の刻み角

        # NOTE: mm, pxの変換は深度、解像度、受光素子のサイズに依存するのでdetect時に変換
        self.hand_radius_mm = hand_radius_mm
        self.finger_radius_mm = finger_radius_mm

        self.h, self.w = frame_size
        self.fp = fp

        self.base_angle = 360 // self.finger_num  # ハンドの指間の角度 (等間隔前提)
        self.candidate_num = self.base_angle // self.unit_angle


    def _convert_mm_to_px(self, v_mm: Mm, d: Mm) -> Px:
        v_px = (v_mm / d) * self.fp  # (v_mm / 1000) * self.fp / (d / 1000)
        return v_px
    
    def _convert_px_to_mm(self, v_px: Px, d: Mm) -> Mm:
        v_mm = (v_px / self.fp) * d
        return v_mm
    
    def get_target_index(self, contours):
        center_w = self.w / 2
        center_h = self.h / 2
        point = (center_w, center_h)
        index = -1
        for i, contour in enumerate(contours):
            # 点が輪郭の内側の場合は +1、輪郭の境界線上の場合は 0、輪郭の外側の場合は -1 を返す
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                index = i
                break
        return index
    
    def get_insertion_points(self, ellipse, theta, hand_radius_px, finger_radius_px):
        def generate_ellipse(c, ab, angle):
            cx, cy = c
            a, b = ab
            a /= 2
            b /= 2
            angle *= -1
            a += finger_radius_px
            # print("radio : ", a / finger_radius_px)
            b += finger_radius_px
            # 楕円のパラメータを計算
            cos_angle = np.cos(np.radians(angle))
            sin_angle = np.sin(np.radians(angle))

            # 楕円の方程式を返す
            def ellipse_equation(x, y):
                x_rotated = cos_angle * (x - cx) - sin_angle * (y - cy)
                y_rotated = sin_angle * (x - cx) + cos_angle * (y - cy)
                return (x_rotated**2 / a**2) + (y_rotated**2 / b**2) - 1

            return ellipse_equation

        # 解きたい関数をリストで戻す
        def func(x, t, ellipse):
            px, py, r = x
            ellipse_func = generate_ellipse(ellipse[0], ellipse[1], ellipse[2])

            equations = []
            for i in range(self.finger_num):
                equations.append(
                    ellipse_func(px + r * np.cos(t + np.radians(i*self.base_angle)), py - r * np.sin(t + np.radians(i*self.base_angle)))
                )

            return equations

        # 制約を設定
        def constraint_func(x):
            return x[2] - 10.0  #TMP r >= 10.0
        
            # 初期値
        cons = (
            {'type': 'ineq', 'fun': constraint_func}
        )

        # 最適化を実行
        initial_guess = [ellipse[0][0], ellipse[0][1], hand_radius_px]
        result = optimize.minimize(lambda x: np.sum(np.array(func(x, theta, ellipse))**2), initial_guess, constraints=cons, method="SLSQP")
        return result.x

    def is_in_image(self, h, w, i, j) -> bool:
        if i < 0 or i >= h or j < 0 or j >= w:
            return False
        return True

    def get_xytr_list(self, depth: Image, contours: np.ndarray, centers):
        # center_dが欠損すると0 divisionになるので注意
        index = self.get_target_index(contours)
        center = centers[index]
        center_d = depth[center[1], center[0]]

        # TMP 動作確認
        if center_d == 0:
            center_d = 600

        hand_radius_px = self._convert_mm_to_px(self.hand_radius_mm, center_d)
        finger_radius_px = self._convert_mm_to_px(self.finger_radius_mm, center_d)
        # ベクトルははじめの角度求めるとかで関数内部で計算してもいいかも

        ellipse = cv2.fitEllipse(contours[index])

        xytr_list = []
        for i in range(self.candidate_num):
            theta = np.radians(i * self.unit_angle)
            xyr = self.get_insertion_points(ellipse, theta, hand_radius_px, finger_radius_px)
            xytr = np.insert(xyr, 2, theta)
            xytr_list.append(xytr)

        return xytr_list, center_d, finger_radius_px
    
    def point_average_depth(self, hi, wi, depth: Image, finger_radius_px):
        mask = np.zeros_like(depth)
        cv2.circle(mask, (wi, hi), finger_radius_px, 1, thickness=-1)
        mask = mask.astype(np.bool)

        return depth[mask].mean()


    
    def calculate(self, depth: Image, contours: np.ndarray, centers):
        # center_d = depth[self.h // 2, self.w // 2]
        # center_d = 600 # TODO

        xytr_list, center_d, finger_radius_px = self.get_xytr_list(depth, contours, centers)

        max_score = -1
        best_x = -1
        best_y = -1
        best_t = -1
        best_r = -1

        for i in range(self.candidate_num):
            xi, yi, ti, ri = xytr_list[i]
            update = True
            tmp_score = 100000
            for j in range(self.finger_num):
                hi = int(yi - ri * np.sin(ti + np.radians(j*self.base_angle)))
                wi = int(xi + ri * np.cos(ti + np.radians(j*self.base_angle)))
                if not self.is_in_image(self.h, self.w, hi, wi):
                    update = False
                    break
                tmp_score = min(tmp_score, 
                                self.point_average_depth(hi, wi, depth, finger_radius_px))

            if update and tmp_score > max_score:
                best_x = int(xi) 
                best_y = int(yi)
                best_t = ti
                best_r = ri

                max_score = tmp_score        

        print("x, y, t, r", best_x, best_y, best_t, best_r)
        best_r = self._convert_px_to_mm(best_r, center_d) # mm単位に


        return best_x, best_y, best_t, best_r, center_d


    

    def drawResult(self, img, contours, x, y, t, r, d):
        index = self.get_target_index(contours)
        ellipse = cv2.fitEllipse(contours[index])
        cv2.ellipse(img, ellipse,(255,0,0),2)

        r = self._convert_mm_to_px(r, d)

        for i in range(self.finger_num):
            xx = int(x + r * np.cos(t + np.radians(i * self.base_angle)))
            yy = int(y - r * np.sin(t + np.radians(i * self.base_angle)))
            cv2.circle(img, (xx, yy), 15, (255, 0, 255), thickness=-1)

        return img

finger_num = 3
hand_radius_mm = 100
finger_radius_mm = 15

unit_angle = 5
frame_size = (480, 640)
fp = 605

insertion_calculator = InsertionCalculator(finger_num=finger_num, hand_radius_mm=hand_radius_mm,
                                            finger_radius_mm=finger_radius_mm,
                                            unit_angle=unit_angle, frame_size=frame_size, fp=fp)

depth_tmp = np.zeros(img.shape[:2])
ic_ress = insertion_calculator.calculate(depth_tmp, res.contours, res.centers)

x, y, t, r, d = ic_ress
# img_tmp = img.copy()
img_tmp = insertion_calculator.drawResult(img_tmp, res.contours, x, y, t, r, d)
imshow(img_tmp)

# %%
# %%
# import numpy as np
# from matplotlib import pyplot as plt

# def make_circle(a=1, b=1, xy=(0,0), phi=0, n=100):
#     theta = np.arange(0, 2*np.pi, 2*np.pi/n)
#     X = a*np.cos(theta)
#     Y = b*np.sin(theta)
#     data_mat = np.matrix(np.vstack([X, Y]))
#     phi_d = np.deg2rad(phi)
#     rot = np.matrix([[np.cos(phi_d), -np.sin(phi_d)],
#                      [np.sin(phi_d), np.cos(phi_d)]])
#     rot_data = rot*data_mat
#     X = rot_data[0].A
#     Y = rot_data[1].A

#     return X+xy[0], Y+xy[1]

# plt.figure()
# X, Y = make_circle(a=3, b=2, xy=(0,0), phi=0, n=100)
# plt.scatter(X, Y, c="r")
# X, Y = make_circle(a=10, b=4, xy=(2,2), phi=45, n=100)
# plt.scatter(X, Y, c="g")
# X, Y = make_circle(a=20, b=2, xy=(-3,-5), phi=-70, n=100)
# plt.scatter(X, Y, c="b")

# # %%
# def make_circle(a=1, b=1, xy=(0,0), phi=0, n=5):
#     theta = np.arange(0, 2*np.pi, 2*np.pi/n)
#     X = a*np.cos(theta)
#     Y = b*np.sin(theta)
#     data_mat = np.matrix(np.vstack([X, Y]))
#     phi_d = np.deg2rad(phi)
#     rot = np.matrix([[np.cos(phi_d), -np.sin(phi_d)],
#                      [np.sin(phi_d), np.cos(phi_d)]])
#     rot_data = rot*data_mat
#     X = rot_data[0].A
#     Y = rot_data[1].A

#     return X+xy[0], Y+xy[1]

# X, Y = make_circle(a=ellipse[1][0] / 2, b=ellipse[1][1] / 2, xy=ellipse[0], phi=ellipse[2], n=5)

# # %%
# print(X.shape)
# img_tmp = img.copy()
# cv2.circle(img_tmp, (int(ellipse[0][0]), int(ellipse[0][1])), 3, (255, 0, 255), 5, lineType=cv2.LINE_AA)
# for i in range(5):
#     cv2.circle(img_tmp, (int(X[0][i]), int(Y[0][i])), 3, (255, 0, 255), 5, lineType=cv2.LINE_AA)

# imshow(img_tmp)


# # %%
# import sympy as sp
# sp.init_printing()
# sp.var('x, y, a, b, c, d, e, f')
# eq1=sp.Eq(a*x+b*y, e)
# eq2=sp.Eq(c*x+d*y, f)
# sp.solve ([eq1, eq2], [x, y]) 

# # %%
# import sympy as sp
# sp.init_printing()
# sp.var('x, y, a, b, c, d, e, f')
# eq1=sp.Eq(a*x+b*y, e)
# sp.solve ([eq1], [x]) 
# # %%
# sp.var('px, py, r')
# t = 1
# # eq1=sp.Eq(x**2 / 4 + y**2 / 9, 1)
# eq2=sp.Eq((px + r * sp.cos(t)) ** 2 / 4 + (py + r * sp.sin(t)) ** 2 / 9, 1)
# eq3=sp.Eq((px + r * sp.cos(t + sp.pi * 2 / 3)) ** 2 / 4 + (py + r * sp.sin(t + sp.pi * 2 / 3)) ** 2 / 9, 1)
# eq4=sp.Eq((px + r * sp.cos(t - sp.pi * 2 / 3)) ** 2 / 4 + (py + r * sp.sin(t - sp.pi * 2 / 3)) ** 2 / 9, 1)
# sp.solve ([eq2, eq3, eq4]) 

# # %%
# import numpy as np
# from scipy import optimize

# import numpy as np

# def generate_ellipse(c, ab, angle):
#     cx, cy = c
#     a, b = ab
#     a /= 2
#     b /= 2
#     # 楕円のパラメータを計算
#     cos_angle = np.cos(np.radians(angle))
#     sin_angle = np.sin(np.radians(angle))

#     # 楕円の方程式を返す
#     def ellipse_equation(x, y):
#         x_rotated = cos_angle * (x - cx) - sin_angle * (y - cy)
#         y_rotated = sin_angle * (x - cx) + cos_angle * (y - cy)
#         return (x_rotated**2 / a**2) + (y_rotated**2 / b**2) - 1

#     return ellipse_equation

# # 解きたい関数をリストで戻す
# def func(x, t, ellipse):
#     px, py, r = x
#     ellipse_func = generate_ellipse(ellipse[0], ellipse[1], ellipse[2])
#     equations = [
#         ellipse_func(px + r * np.cos(t), py + r * np.sin(t)),
#         ellipse_func(px + r * np.cos(t + np.pi * 2 / 3), py + r * np.sin(t + np.pi * 2 / 3)),
#         ellipse_func(px + r * np.cos(t - np.pi * 2 / 3), py + r * np.sin(t - np.pi * 2 / 3)), 
#     ]
#     return equations

# def constraint_func(x):
#     return x[2] - 50.0  # r >= 0.001

# constraint = optimize.NonlinearConstraint(constraint_func, lb=[-np.inf, -np.inf, 0.001], ub=np.inf)


# # result = optimize.root(func, [0.0, 0.0, 0.0], args=(0.0, ellipse), method="broyden1")
# # result = optimize.root(func, [ellipse[0][0], ellipse[0][1], 200.0], args=(1.0, ellipse), method="broyden1")
# # print(result)


# result = optimize.root(func, [ellipse[0][0], ellipse[0][1], 200.0], args=(1.0, ellipse), method="broyden1", constraints=constraint)
# print(result)
