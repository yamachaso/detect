from typing import List, Tuple

import cv2
import numpy as np
import torch

from modules.ros.utils import PointProjector
from modules.image import extract_depth_between_two_points
from modules.type import Image, ImagePointUV, ImagePointUVD, Mm, Px
from modules.colored_print import *
from modules.ros.action_clients import TFClient
from modules.ros.publishers import ImageMatPublisher2

from scipy import optimize

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose, PoseStamped

class GraspDetector:
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

        self.base_rmat = self._compute_rmat(self.base_angle)
        self.unit_rmat = self._compute_rmat(self.unit_angle)


        self.tf_clients = [
            TFClient("left_camera_color_optical_frame"), 
            TFClient("right_camera_color_optical_frame")
        ]


        self.result_publisher = ImageMatPublisher2("/grasp_detection_server_result", queue_size=10)
        self.w1 = torch.tensor(1.0 / 3, requires_grad=True)
        self.w2 = torch.tensor(1.0 / 3, requires_grad=True)
        self.lr = 1.0e-2


    def _compute_rmat(self, angle):
        """
        回転行列を生成
        """
        cos_rad, sin_rad = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        return np.array([[cos_rad, -sin_rad], [sin_rad, cos_rad]])

    def _convert_mm_to_px(self, v_mm: Mm, d: Mm) -> Px:
        v_px = (v_mm / d) * self.fp  # (v_mm / 1000) * self.fp / (d / 1000)
        return v_px
    
    def _convert_px_to_mm(self, v_px: Px, d: Mm) -> Mm:
        v_mm = (v_px / self.fp) * d
        return v_mm

    def _compute_rotated_points(self, center: ImagePointUV, base_v: np.ndarray, angle: int):
        # anchorsの計算時は事前にrmatは計算
        rmat = self._compute_rmat(angle)
        finger_v = base_v
        rotated_points = []
        for _ in range(360 // angle):
            rotated_points.append(
                tuple(np.int0(np.round(center + finger_v))))
            finger_v = np.dot(finger_v, rmat)
        return rotated_points

    def compute_insertion_points(self, center: ImagePointUV, base_finger_v: np.ndarray):
        return self._compute_rotated_points(center, base_finger_v, self.base_angle)
    
    def distance_point_between_line(self, px, py, x1, y1, x2, y2):
        a = y2 - y1
        b = x1 - x2
        c = -x1 * y2 + x2 * y1
        return np.abs(a * px + b * py + c) / np.sqrt(a * a + b * b)


    def get_min_distance_with_wall(self, center, uvs):
        px, py = center
        (br_x, br_y), (tr_x, tr_y), (tl_x, tl_y), (bl_x, bl_y) = uvs
        # wall_distance = 100000000
        # wall_distance = min(wall_distance, self.distance_point_between_line(px, py, br_x, br_y, tr_x, tr_y))
        # wall_distance = min(wall_distance, self.distance_point_between_line(px, py, tr_x, tr_y, tl_x, tl_y))
        # wall_distance = min(wall_distance, self.distance_point_between_line(px, py, tl_x, tl_y, bl_x, bl_y))
        # wall_distance = min(wall_distance, self.distance_point_between_line(px, py, bl_x, bl_y, br_x, br_y))
        wall_distance = np.array([
            self.distance_point_between_line(px, py, br_x, br_y, tr_x, tr_y),
            self.distance_point_between_line(px, py, tr_x, tr_y, tl_x, tl_y),
            self.distance_point_between_line(px, py, tl_x, tl_y, bl_x, bl_y),
            self.distance_point_between_line(px, py, bl_x, bl_y, br_x, br_y)
        ])
        wall_distance.sort()

        # if wall_distance[1]

        return wall_distance[0]


    def get_corner_coordinate(self, arm_index):
        header = Header()
        header.frame_id = "container_br"
        br_point = self.tf_clients[arm_index].transform_point(header, Point()).point      
        header.frame_id = "container_tr"
        tr_point = self.tf_clients[arm_index].transform_point(header, Point()).point
        header.frame_id = "container_tl"
        tl_point = self.tf_clients[arm_index].transform_point(header, Point()).point
        header.frame_id = "container_bl"
        bl_point = self.tf_clients[arm_index].transform_point(header, Point()).point
        
        return (br_point.x, br_point.y), (tr_point.x, tr_point.y), (tl_point.x, tl_point.y), (bl_point.x, bl_point.y)


    def detect(self, arm_index, img: Image, depth: Image, projector: PointProjector, centers, contours, masks, exclusion_list):
        # depth score
        depth_list = np.array([depth[center[1]][center[0]] for center in centers])
        depth_list[depth_list > 1800] = max(depth_list[depth_list < 1800]) # 1.8m以上は明らかにおかしい
        max_depth = depth_list.max()
        min_depth = depth_list.min()
        depth_score = 1 - (depth_list - min_depth) / (max_depth - min_depth + 0.0001)

        # ellipse score
        ellipse_list = [cv2.fitEllipse(contour) for contour in contours]
        ellipse_masks = [cv2.ellipse(np.zeros_like(depth),ellipse, 1, -1).astype(np.bool) for ellipse in ellipse_list]
        original_masks = [mask.astype(np.bool) for mask in masks]
        ellipse_iou = []
        for mask, emask in zip(original_masks, ellipse_masks):
            ellipse_iou.append(np.sum(mask * emask) / np.sum(mask | emask)) # IoU
        max_iou = max(ellipse_iou)
        min_iou = min(ellipse_iou)
        ellipse_score = (ellipse_iou - min_iou) / (max_iou - min_iou + 0.0001)

        # angle score
        ratio_list = np.array([min(el[1][0], el[1][1]) / max(el[1][0], el[1][1]) for el in ellipse_list])
        max_ratio = max(ratio_list)
        min_raito = min(ratio_list)
        angle_score = (ratio_list - min_raito) / (max_ratio - min_raito + 0.0001)

        # area score
        area_list = np.array([cv2.contourArea(contour) for contour in contours])
        max_area = max(area_list)
        min_area = min(area_list)
        area_score = (area_list - min_area) / (max_area - min_area + 0.0001)

        # wall score
        cor_coos = self.get_corner_coordinate(arm_index)

        printg("depth_list : {}".format(depth_list))

        uvs_list = []
        for distance in depth_list:
            z = distance / 1000 # mm -> m
            uvs = [projector.camera_to_screen(x, y, z) for x, y in cor_coos]
            uvs_list.append(uvs)

        wall_list = np.array([self.get_min_distance_with_wall(center, uvs) for (center, uvs) in zip(centers, uvs_list)])

        def sigmoid(x):
            a = 100 # キャベツの直径(px単位)
            return 1-1.0 / (1.0 + np.exp((x  - a) / a * 5 ))
        wall_score = sigmoid(wall_list)



        dc_pub = ImageMatPublisher2("/depth_score", queue_size=10)
        fc_pub = ImageMatPublisher2("/friction_score", queue_size=10)
        ac_pub = ImageMatPublisher2("/angle_score", queue_size=10)
        wc_pub = ImageMatPublisher2("/wall_score", queue_size=10)
        arc_pub = ImageMatPublisher2("/area_score", queue_size=10)

        dc_img, fc_img, ac_img, wc_img, arc_img =  img.copy(), img.copy(), img.copy(), img.copy(), img.copy()
        for i in range(len(depth_score)):
           cv2.putText(dc_img, f"{depth_score[i]:.2f}", (centers[i][0] - 10, centers[i][1] + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
           cv2.putText(fc_img, f"{ellipse_score[i]:.2f}", (centers[i][0] - 10, centers[i][1] + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
           cv2.putText(ac_img, f"{angle_score[i]:.2f}", (centers[i][0] - 10, centers[i][1] + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
           cv2.putText(wc_img, f"{wall_score[i]:.2f}", (centers[i][0] - 10, centers[i][1] + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
           cv2.putText(arc_img, f"{area_score[i]:.2f}", (centers[i][0] - 10, centers[i][1] + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        #    cv2.putText(tc_img, f"{final_score[i]:.2f}", (centers[i][0] - 10, centers[i][1] + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        dc_pub.publish(dc_img)
        fc_pub.publish(fc_img)
        ac_pub.publish(ac_img)
        wc_pub.publish(wc_img)
        arc_pub.publish(arc_img)
        # tc_pub.publish(tc_img)


        # # 単位変換
        # depth_list = [depth[center[1]][center[0]] for center in centers]
        # max_depth = max(depth_list)
        # min_depth = min(depth_list)
        # depth_score = 1 - (depth_list - min_depth) / (max_depth - min_depth + 0.0001)

        # ellipse_list = [cv2.fitEllipse(contour) for contour in contours]
        # ellipse_masks = [cv2.ellipse(np.zeros_like(depth),ellipse, 1, -1).astype(np.bool) for ellipse in ellipse_list]
        # original_masks = [mask.astype(np.bool) for mask in masks]
        # ellipse_iou = []
        # for mask, emask in zip(original_masks, ellipse_masks):
        #     ellipse_iou.append(np.sum(mask * emask) / np.sum(mask | emask)) # IoU
        # max_iou = max(ellipse_iou)
        # min_iou = min(ellipse_iou)
        # ellipse_score = (ellipse_iou - min_iou) / (max_iou - min_iou + 0.0001)


        # cor_coos = self.get_corner_coordinate(arm_index)

        # printg("cor_coos : {}".format(cor_coos))

        # uvs_list = []
        # for distance in depth_list:
        #     z = distance / 1000 # mm -> m
        #     uvs = [projector.camera_to_screen(x, y, z) for x, y in cor_coos]
        #     uvs_list.append(uvs)
        
        # printy("center : {}, uvs : {}".format(centers[0], uvs_list[0]))

        (u1, v1), (u2, v2), (u3, v3), (u4, v4) = uvs_list[0]
        cv2.line(img, (int(u1), int(v1)), (int(u2), int(v2)), 255, 2, lineType=cv2.LINE_AA)
        cv2.line(img, (int(u2), int(v2)), (int(u3), int(v3)), 255, 2, lineType=cv2.LINE_AA)
        cv2.line(img, (int(u3), int(v3)), (int(u4), int(v4)), 255, 2, lineType=cv2.LINE_AA)
        cv2.line(img, (int(u4), int(v4)), (int(u1), int(v1)), 255, 2, lineType=cv2.LINE_AA)

        # wall_list = [self.get_min_distance_with_wall(center, uvs) for (center, uvs) in zip(centers, uvs_list)]
        # max_wall = max(wall_list)
        # min_wall = min(wall_list)
        # wall_score = (wall_list - min_wall) / (max_wall - min_wall + 0.0001)

        # TMP!!!!!
        # final_score = depth_score * 0.2 + ellipse_score * 0.2 + wall_score * 1.0
        final_score = depth_score * 1.0 + ellipse_score * 0.0 + wall_score * 0.8 + angle_score * 0.0 + area_score * 0.5

        printg(wall_score)

        for i in range(len(final_score)):
           cv2.putText(img, f"{final_score[i]:.2f}", 
                       (centers[i][0] - 5, centers[i][1] + 5), 
                       cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 1)
        

        while final_score.size > 0:
            target_index = np.argmax(final_score)
            if len(exclusion_list) == 0 or np.all(np.linalg.norm(centers[target_index] - exclusion_list, axis = 1) > 20): #TMP
                break
            else:
                final_score = np.delete(final_score, target_index)
                centers = np.delete(centers, target_index, axis=0) # 2次元配列はaxis設定しないと1次元化されてしまう
                contours = np.delete(contours, target_index, axis=0) # 2次元配列はaxis設定しないと1次元化されてしまう
                ellipse_list = np.delete(ellipse_list, target_index, axis=0) # 同上
                printy("Inappropreate target")

        # cv2.putText(img, f"{final_score[target_index]:.2f}", (centers[target_index][0] + 5, centers[target_index][1] + 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.ellipse(img, ellipse_list[target_index], (0, 255, 255), 3)
        return target_index, img, max(final_score), centers, contours


    def detect_training(self, arm_index, img: Image, depth: Image, projector: PointProjector, centers, contours, masks, exclusion_list):
        # depth score
        depth_list = np.array([depth[center[1]][center[0]] for center in centers])
        depth_list[depth_list > 1800] = max(depth_list[depth_list < 1800]) # 1.8m以上は明らかにおかしい
        max_depth = depth_list.max()
        min_depth = depth_list.min()
        depth_score = 1 - (depth_list - min_depth) / (max_depth - min_depth + 0.0001)

        # ellipse score
        ellipse_list = [cv2.fitEllipse(contour) for contour in contours]
        ellipse_masks = [cv2.ellipse(np.zeros_like(depth),ellipse, 1, -1).astype(np.bool) for ellipse in ellipse_list]
        original_masks = [mask.astype(np.bool) for mask in masks]
        ellipse_iou = []
        for mask, emask in zip(original_masks, ellipse_masks):
            ellipse_iou.append(np.sum(mask * emask) / np.sum(mask | emask)) # IoU
        max_iou = max(ellipse_iou)
        min_iou = min(ellipse_iou)
        ellipse_score = (ellipse_iou - min_iou) / (max_iou - min_iou + 0.0001)

        # wall score
        cor_coos = self.get_corner_coordinate(arm_index)

        printg("depth_list : {}".format(depth_list))

        uvs_list = []
        for distance in depth_list:
            z = distance / 1000 # mm -> m
            uvs = [projector.camera_to_screen(x, y, z) for x, y in cor_coos]
            uvs_list.append(uvs)

        wall_list = np.array([self.get_min_distance_with_wall(center, uvs) for (center, uvs) in zip(centers, uvs_list)])

        def sigmoid(x):
            a = 100 # キャベツの直径(px単位)
            return 1-1.0 / (1.0 + np.exp((x  - a) / a * 5 ))
        wall_score = sigmoid(wall_list)

        # max_wall = max(wall_list)
        # min_wall = min(wall_list)
        # wall_score = (wall_list - min_wall) / (max_wall - min_wall + 0.0001)

        for i in range(len(centers)):
           cv2.putText(img, f"{i}", 
                       (centers[i][0] - 5, centers[i][1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
           
           
        self.result_publisher.publish(img)

        index = int(input("Enter the best cabbage index you think : "))

        # ref: https://rightcode.co.jp/blog/information-technology/pytorch-automatic-differential-linear-regression

        depth_score = torch.from_numpy(depth_score)
        ellipse_score = torch.from_numpy(ellipse_score)
        wall_score = torch.from_numpy(wall_score)
        final_score = depth_score * self.w1 + ellipse_score * self.w2 + wall_score * (1 - self.w1 - self.w2)

        printb("dep: {}, ell: {}, wal: {}".format(depth_score[index],
                                                  ellipse_score[index],
                                                  wall_score[index]))
        
        print(torch.mean(torch.cat([depth_score[0:index], depth_score[index:]])))
        print(torch.mean(torch.cat([ellipse_score[0:index], ellipse_score[index:]])))
        print(torch.mean(torch.cat([wall_score[0:index], wall_score[index:]])))


        print("final score : ", final_score[index])
        print("depth score : ", depth_score[index])
        print("ellipse score : ", ellipse_score[index])
        print("wall score: ", wall_score[index])
        y = final_score[index] - torch.mean(torch.cat([final_score[0:index], final_score[index:]]))

        y.backward()
        print("y : ", y)
        print("grad : ", self.w1.grad, self.w2.grad)

        with torch.no_grad():
            self.w1 += self.w1.grad * self.lr
            self.w2 += self.w2.grad * self.lr
            self.w1.grad.zero_()
            self.w2.grad.zero_()

        print("重み: ", self.w1, self.w2)


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

        self.index = None


    def _convert_mm_to_px(self, v_mm: Mm, d: Mm) -> Px:
        v_px = (v_mm / d) * self.fp  # (v_mm / 1000) * self.fp / (d / 1000)
        return v_px
    
    def _convert_px_to_mm(self, v_px: Px, d: Mm) -> Mm:
        v_mm = (v_px / self.fp) * d
        return v_mm
    
    def get_target_index(self, contours):
        # print(type(contours))
        # print(contours.dtype)
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
    
    def is_depth_valid(self, d):
        if d < 50 or d > 1800: # 50mm以下または1800mm以上はおかしい
            return False
        else:
            return True

    def get_xytr_list(self, depth: Image, contour: np.ndarray, center):
        # center_dが欠損すると0 divisionになるので注意
        # self.index = self.get_target_index(contours)
        # center = centers[self.index]
        center_d = depth[center[1], center[0]]
        if not self.is_depth_valid(center_d):
            center_d = self.point_average_depth(center[1], center[0], depth, 5)


        # TMP 動作確認
        # if center_d == 0:
        #     center_d = 600

        hand_radius_px = self._convert_mm_to_px(self.hand_radius_mm, center_d)
        finger_radius_px = self._convert_mm_to_px(self.finger_radius_mm, center_d)
        # ベクトルははじめの角度求めるとかで関数内部で計算してもいいかも

        ellipse = cv2.fitEllipse(contour)

        xytr_list = []
        for i in range(self.candidate_num):
            theta = np.radians(i * self.unit_angle)
            xyr = self.get_insertion_points(ellipse, theta, hand_radius_px, finger_radius_px)
            xytr = np.insert(xyr, 2, theta)
            xytr_list.append(xytr)

        print("index : ", self.index)
 
        self.cabbage_size_mm = self._convert_px_to_mm(max(ellipse[1][0], ellipse[1][1]), center_d)

        return xytr_list, center_d, finger_radius_px, self.cabbage_size_mm
    
    def point_average_depth(self, hi, wi, depth: Image, finger_radius_px):
        mask = np.zeros_like(depth)
        cv2.circle(mask, (wi, hi), int(finger_radius_px), 1, thickness=-1)
        mask = mask.astype(np.bool)
        mask[depth < 50] = False
        mask[depth > 1800] = False

        return depth[mask].mean()
    
    def calculate(self, depth: Image, contour: np.ndarray, center):
        # center_d = depth[self.h // 2, self.w // 2]
        # center_d = 600 # TODO

        printb("in calculate")
        xytr_list, center_d, finger_radius_px, cabbage_size_mm = self.get_xytr_list(depth, contour, center)

        max_score = -1
        best_x = -1
        best_y = -1
        best_t = -1
        best_r = -1

        printc("CABBAGE SIZE: {}".format(cabbage_size_mm))

        good_xytr_list = []

        for i in range(self.candidate_num):
            xi, yi, ti, ri = xytr_list[i]
            update = True
            worst_depth = 100000
            for j in range(self.finger_num):
                hi = int(yi - ri * np.sin(ti + np.radians(j*self.base_angle)))
                wi = int(xi + ri * np.cos(ti + np.radians(j*self.base_angle)))
                if not self.is_in_image(self.h, self.w, hi, wi):
                    update = False
                    break
                worst_depth = min(worst_depth, 
                                self.point_average_depth(hi, wi, depth, finger_radius_px * 2))
                
            # cabbage_size_mm * 0.6でキャベツの高さをイメージ
            if (worst_depth - center_d) / (cabbage_size_mm * 0.6) > 1:
                good_xytr_list.append([xi, yi, ti, ri])
                    

            if update and worst_depth > max_score:
                best_x = int(xi) 
                best_y = int(yi)
                best_t = ti
                best_r = ri

                max_score = worst_depth        

        if len(good_xytr_list) > 0:
            
            dis2 = 100000
            for xi, yi, ti, ri in good_xytr_list:
                if dis2 > (center[0] - xi)**2 + (center[1] - yi)**2:
                    best_x = int(xi) 
                    best_y = int(yi)
                    best_t = ti
                    best_r = ri
                    dis2 = (center[0] - xi)**2 + (center[1] - yi)**2




        printc("x, y, t, r : {}, {}, {}, {}".format(best_x, best_y, best_t, best_r))


        best_r = self._convert_px_to_mm(best_r, center_d) # mm単位に

        if cabbage_size_mm > 250:
            printg("cabbage too big, maybe multiple cabbages")
            d = -1


        return best_x, best_y, best_t, best_r, center_d


    def compute_cabbage_angle(self, ratio, a):
        # ratioはa ~ 1である必要あり
        ratio = min(ratio, 1.0)
        ratio = max(ratio, a) 
        return np.arccos((2 * ratio * ratio - 1 - a * a) / (1 - a * a)) / 2

    def compute_cabbage_angle_reverse(self, angle, a):
        return np.sqrt(((1 - a*a) * np.cos(2 * angle) + 1 + a*a) / 2)


    def get_major_minor_ratio(self, contour):


        ellipse = cv2.fitEllipse(contour)
        a, b = ellipse[1]
        if a < b:
            a, b = b, a
        return b / a
    
    def get_access_distance(self, contour):
        # if self.index == -1:
        #     return -1
        
        ratio  = self.get_major_minor_ratio(contour)
        a = 0.6 # キャベツの長軸に対する短軸の長さの比
        angle = self.compute_cabbage_angle(ratio, a)

        return (1 - self.compute_cabbage_angle_reverse(np.pi / 2 - angle, a)) * self.cabbage_size_mm / 2


    def drawResult(self, img, contour, x, y, t, r, d):
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(img, ellipse,(255,0,0),2)

        r = self._convert_mm_to_px(r, d)

        for i in range(self.finger_num):
            xx = int(x + r * np.cos(t + np.radians(i * self.base_angle)))
            yy = int(y - r * np.sin(t + np.radians(i * self.base_angle)))
            cv2.circle(img, (xx, yy), 15, (255, 0, 255), thickness=-1)

        return img


# class GraspCandidateElement:
#     # ハンド情報、画像情報、局所的情報（ポイント、値）、しきい値
#     def __init__(self, hand_radius_px: Px, finger_radius_px: Px, depth: Image, contour: np.ndarray,
#                  center: ImagePointUV, insertion_point: ImagePointUV,
#                  insertion_th: float, contact_th: float, bw_depth_th: float):
#         self.hand_radius_px = hand_radius_px
#         self.finger_radius_px = finger_radius_px

#         self.center = center
#         self.center_d = depth[center[1], center[0]]
#         self.insertion_point = insertion_point
#         self.insertion_point_d = None
#         self.intersection_point = None
#         self.intersection_point_d = None
#         self.contact_point = None
#         self.contact_point_d = None
#         self.insertion_score = 0
#         self.contact_score = 0
#         self.bw_depth_score = 0
#         self.total_score = 0
#         self.is_valid_pre = False
#         self.is_valid = False
#         # validnessチェックに引っかかった箇所を追加していく
#         self.debug_infos = []

#         # 詳細なスコアリングの前に明らかに不正な候補は弾く
#         h, w = depth.shape[:2]
#         # centerがフレームインしているのは明らか
#         self.is_framein = self._check_framein(h, w, self.insertion_point)
#         if not self.is_framein:
#             self.debug_infos.append(("framein", self.insertion_point))
#             return

#         self.insertion_point_d = depth[insertion_point[1], insertion_point[0]]
#         self.is_valid_pre = self.is_framein and self._precheck_validness()
#         if not self.is_valid_pre:
#             self.debug_infos.append(
#                 ("precheck", (self.center_d, self.insertion_point_d)))
#             return

#         # TODO: ハンドの開き幅調整可能な場合 insertion point = contact pointとなるので、insertionのスコアはいらない
#         # 挿入点の評価
#         self.intersection_point = self._compute_intersection_point(contour)
#         _, intersection_max_d, intersection_mean_d = self._compute_depth_profile_in_finger_area(
#             depth, self.intersection_point, finger_radius_px)
#         min_d = intersection_mean_d
#         max_d = max(intersection_max_d, self.insertion_point_d)
#         self.insertion_score = self._compute_point_score(
#             depth, min_d, max_d, self.insertion_point)
#         if self.insertion_score < insertion_th:
#             self.debug_infos.append(("insertion_score", self.insertion_score))
#             return
#         # 接触点の計算と評価
#         self.contact_point = self._compute_contact_point(
#             self.intersection_point)
#         self.contact_score = self._compute_point_score(
#             depth, min_d, max_d, self.contact_point)
#         if self.contact_score < contact_th:
#             self.debug_infos.append(("contact_score", self.contact_score))
#             return
#         # 挿入点と接触点の間の障害物の評価
#         self.bw_depth_score = self._compute_bw_depth_score(
#             depth, self.contact_point)
#         if self.bw_depth_score < bw_depth_th:
#             self.debug_infos.append(("bw_depth_score", self.bw_depth_score))
#             return
#         self.total_score = self._compute_total_score()
#         # すべてのスコアが基準を満たしたときのみvalid判定
#         self.is_valid = True

#     def _check_framein(self, h: Px, w: Px, pt: ImagePointUV) -> bool:
#         return not (pt[0] < 0 or pt[1] < 0 or pt[0] >= w or pt[1] >= h)

#     def _precheck_validness(self) -> bool:
#         is_valid_pre = \
#             self._check_depth_existance() and \
#             self._check_depth_difference()
#         return is_valid_pre

#     def _check_depth_existance(self) -> bool:
#         return self.center_d != 0 and self.insertion_point_d != 0

#     def _check_depth_difference(self) -> bool:
#         return self.center_d <= self.insertion_point_d

#     def _compute_intersection_point(self, contour: np.ndarray) -> ImagePointUV:
#         # TODO: 個々のcropはまだ上位に引き上げられそう
#         x, y, w, h = cv2.boundingRect(contour)
#         upper_left_point = np.array((x, y))
#         shifted_contour = contour - upper_left_point
#         shifted_center, shifted_edge = [
#             tuple(pt - upper_left_point) for pt in (self.center, self.insertion_point)]

#         shifted_intersection = self._compute_intersection_between_contour_and_line(
#             (h, w), shifted_contour, shifted_center, shifted_edge)
#         intersection = tuple(shifted_intersection + upper_left_point)
#         return intersection
    
#     def _compute_intersection_between_contour_and_line(self, img_shape, contour, line_pt1_xy, line_pt2_xy):
#         """
#         輪郭と線分の交点座標を取得する
#         TODO: 線分ごとに描画と論理積をとり非効率なので改善方法要検討
#         only here
#         """
#         blank_img = np.zeros(img_shape)
#         # クロップ前に計算したcontourをクロップ後の画像座標に変換し描画
#         cnt_img = blank_img.copy()
#         cv2.drawContours(cnt_img, [contour], -1, 255, 1, lineType=cv2.LINE_AA)
#         # クロップ前に計算したlineをクロップ後の画像座標に変換し描画
#         line_img = blank_img.copy()
#         # 斜めの場合、ピクセルが重ならない場合あるのでlineはthicknessを２にして平均をとる
#         line_img = cv2.line(line_img, line_pt1_xy, line_pt2_xy,
#                             255, 2, lineType=cv2.LINE_AA)
#         # バイナリ画像(cnt_img, line_img)のbitwiseを用いて、contourとlineの交点を検出
#         bitwise_img = blank_img.copy()
#         cv2.bitwise_and(cnt_img, line_img, bitwise_img)

#         intersections = [(w, h)
#                         for h, w in zip(*np.where(bitwise_img > 0))]  # hw to xy
#         if len(intersections) == 0:
#             raise Exception(
#                 "No intersection between a contour and a candidate element, 'hand_radius_mm' or 'fp' may be too small")
#         mean_intersection = np.int0(np.round(np.mean(intersections, axis=0)))
#         return mean_intersection

#     def _compute_contact_point(self, intersection_point: ImagePointUV) -> ImagePointUV:
#         direction_v = np.array(self.insertion_point) - np.array(self.center)
#         unit_direction_v = direction_v / np.linalg.norm(direction_v, ord=2)
#         # 移動後座標 = 移動元座標 + 方向ベクトル x 移動量(指半径[pixel])
#         contact_point = np.int0(
#             np.round(intersection_point + unit_direction_v * self.finger_radius_px))
#         return contact_point

#     def _compute_point_score(self, depth: Image, min_d: Mm, max_d: Mm, pt: ImagePointUV) -> float:
#         # TODO: 引数のmin, maxはターゲットオブジェクト周辺の最小値・最大値
#         _, _, mean_d = self._compute_depth_profile_in_finger_area(
#             depth, pt, self.finger_radius_px)
#         score = max(0, (mean_d - min_d)) / (max_d - min_d + 1e-6)
#         return score

#     def _compute_bw_depth_score(self, depth: Image, contact_point: ImagePointUV) -> float:
#         min_d, max_d, mean_d = self._compute_bw_depth_profile(
#             depth, contact_point, self.insertion_point)
#         # score = max(0, (mean_d - min_d)) / (max_d - min_d + 1e-6)
#         score = 1 - ((mean_d - min_d) / (max_d - min_d + 1e-6))
#         return score
    
#     def _compute_bw_depth_profile(self, depth, contact_point, insertion_point):
#         """
#         contact_pointからinsertion_pointの間での深さの最小値・最大値・平均
#         """
#         values = extract_depth_between_two_points(
#             depth, contact_point, insertion_point)
#         min_depth, max_depth = values.min(), values.max()
#         # 欠損ピクセルの値は除外
#         valid_values = values[values > 0]
#         mean_depth = np.mean(valid_values) if len(valid_values) > 0 else 0
#         return min_depth, max_depth, mean_depth
    
#     def _compute_depth_profile_in_finger_area(self, depth: Image, pt: ImagePointUV, finger_radius_px: Px) -> Tuple[Mm, Mm, Mm]:
#         """
#         ptを中心とする指が差し込まれる範囲における深さの最小値・最大値・平均
#         """
#         h, w = depth.shape[:2]
#         rounded_radius = np.int0(np.round(finger_radius_px))
#         r_slice = slice(max(0, pt[1] - rounded_radius),
#                         min(pt[1] + rounded_radius + 1, h))
#         c_slice = slice(max(0, pt[0] - rounded_radius),
#                         min(pt[0] + rounded_radius + 1, w))
#         cropped_depth = depth[r_slice, c_slice]
#         finger_mask = np.zeros_like(cropped_depth, dtype=np.uint8)
#         cropped_h, cropped_w = cropped_depth.shape[:2]
#         cv2.circle(
#             finger_mask, (cropped_w // 2, cropped_h // 2), rounded_radius, 255, -1)
#         depth_values_in_mask = cropped_depth[finger_mask == 255]
#         return int(np.min(depth_values_in_mask)), int(np.max(depth_values_in_mask)), int(np.mean(depth_values_in_mask))

#     def _compute_total_score(self) -> float:
#         # TODO: ip, cp間のdepthの評価 & 各項の重み付け
#         return self.insertion_score * self.contact_score * self.bw_depth_score

#     def get_points(self):
#         return {"center": self.center, "intersection": self.intersection_point, "contact": self.contact_point, "insertion": self.insertion_point}

#     def get_scores(self):
#         return {"insertion": self.insertion_score, "contact": self.contact_score, "bw_depth": self.bw_depth_score}

#     def draw(self, img: Image, line_color=(0, 0, 0), line_thickness=1, circle_thickness=1, show_circle=True) -> None:
#         cv2.line(img, self.center, self.insertion_point,
#                  line_color, line_thickness, cv2.LINE_AA)
#         if show_circle:
#             rounded_radius = np.int0(np.round(self.finger_radius_px))
#             cv2.circle(img, self.insertion_point, rounded_radius,
#                        (255, 0, 0), circle_thickness, cv2.LINE_AA)
#             cv2.circle(img, self.contact_point, rounded_radius,
#                        (0, 255, 0), circle_thickness, cv2.LINE_AA)

#     # TODO: 重複関数の統合
#     def get_intersection_point_uv(self) -> ImagePointUV:
#         return self.intersection_point

#     def get_contact_point_uv(self) -> ImagePointUV:
#         return self.contact_point

#     def get_insertion_point_uv(self) -> ImagePointUV:
#         return self.insertion_point

#     def get_intersection_point_uvd(self) -> ImagePointUVD:
#         return (self.get_intersection_point_uv(), self.intersection_point_d)

#     def get_contact_point_uvd(self) -> ImagePointUVD:
#         return (self.get_contact_point_uv(), self.contact_point_d)

#     def get_insertion_point_uvd(self) -> ImagePointUVD:
#         return (self.get_insertion_point_uv(), self.insertion_point_d)

# class GraspCandidate:
#     def __init__(self, hand_radius_px, finger_radius_px, angle, depth, contour,
#                  center, insertion_points,
#                  elements_th, el_insertion_th, el_contact_th, el_bw_depth_th):
#         self.hand_radius_px = hand_radius_px
#         self.finger_radius_px = finger_radius_px
#         self.angle = angle
#         # self.bbox_short_radius = bbox_short_radius
#         self.center = center
#         self.center_d = depth[center[1], center[0]]

#         self.is_valid = False
#         self.is_framein = False
#         self.shifted_center = None
#         self.elements_score = 0
#         self.total_score = 0
#         self.debug_infos = []

#         # centerがマスク内になければinvalid
#         # -1: 外側、0: 輪郭上、1: 内側
#         self.is_center_inside_mask = cv2.pointPolygonTest(
#             contour, (int(center[0]), int(center[1])), False) <= 0
#         if self.is_center_inside_mask:
#             self.debug_infos.append(
#                 ("center_inside_mask", self.is_center_inside_mask))
#             return
#         self.elements = [
#             GraspCandidateElement(
#                 hand_radius_px=hand_radius_px, finger_radius_px=finger_radius_px, depth=depth, contour=contour,
#                 center=center, insertion_point=insertion_point,
#                 insertion_th=el_insertion_th, contact_th=el_contact_th, bw_depth_th=el_bw_depth_th
#             ) for insertion_point in insertion_points
#         ]

#         self.is_framein = self._merge_elements_framein()
#         if not self.is_framein:
#             self.debug_infos.append(("framein", self.is_framein))
#             return
#         self.elements_is_valid = self._merge_elements_validness()
#         if not self.elements_is_valid:
#             self.debug_infos.append(("elements_valid", self.elements_is_valid))
#             return
#         # elementの組み合わせ評価
#         self.elements_score = self._compute_elements_score()
#         if self.elements_score < elements_th:
#             self.debug_infos.append(("elements_score", self.elements_score))
#             return
#         # contact pointsの中心とマスクの中心のズレの評価
#         self.shifted_center = self._compute_contact_points_center()
#         # 各スコアの合算
#         self.total_score = self._compute_total_score()
#         # すべてのスコアが基準を満たしたときのみvalid判定
#         self.is_valid = True

#     def _merge_elements_framein(self) -> bool:
#         return np.all([el.is_framein for el in self.elements])

#     def _merge_elements_validness(self) -> bool:
#         return np.all([el.is_valid for el in self.elements])

#     def _compute_contact_points_center(self) -> ImagePointUV:
#         contact_points = self.get_contact_points_uv()
#         return np.int0(np.round(np.mean(contact_points, axis=0)))

#     def _compute_elements_score(self) -> float:
#         element_scores = self.get_element_scores()
#         # return np.prod(element_scores)
#         # return np.mean(element_scores) * (np.min(element_scores) / np.max(element_scores))
        
#         # min_score = np.min(element_scores)
#         # return (np.mean(element_scores) - min_score) / (np.max(element_scores) - min_score + 10e-6)
        
#         # マクシミン的な戦略
#         # スコアの悪い指が含まれない把持候補を選択したいので、各候補の最悪な指のスコアを比較する。
#         return np.min(element_scores) # マクシミン的な戦略

#     def _compute_total_score(self) -> float:
#         return self.elements_score
    
#     def get_center_uv(self) -> ImagePointUV:
#         return self.center

#     def get_intersection_points_uv(self) -> List[ImagePointUV]:
#         return [el.get_intersection_point_uv() for el in self.elements]

#     def get_contact_points_uv(self) -> List[ImagePointUV]:
#         return [el.get_contact_point_uv() for el in self.elements]

#     def get_insertion_points_uv(self) -> List[ImagePointUV]:
#         return [el.get_insertion_point_uv() for el in self.elements]

#     def get_center_uvd(self) -> ImagePointUVD:
#         return (self.center, self.center_d)

#     def get_contact_points_uvd(self) -> List[ImagePointUVD]:
#         return [el.get_contact_point_uvd() for el in self.elements]

#     def get_intersection_points_uvd(self) -> List[ImagePointUVD]:
#         return [el.get_intersection_point_uvd() for el in self.elements]

#     def get_insertion_points_uvd(self) -> List[ImagePointUVD]:
#         return [el.get_insertion_point_uvd() for el in self.elements]

#     def get_element_scores(self) -> List[float]:
#         return [el.total_score for el in self.elements]

#     def get_scores(self):
#         return {"elements": self.elements_score}

#     def draw(self, img, line_color=(0, 0, 0), line_thickness=1, circle_thickness=1, show_circle=True) -> None:
#         for el in self.elements:
#             el.draw(img, line_color, line_thickness,
#                     circle_thickness, show_circle)