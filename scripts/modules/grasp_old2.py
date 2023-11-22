from typing import List, Tuple

import cv2
import numpy as np

from modules.image import extract_depth_between_two_points
from modules.type import Image, ImagePointUV, ImagePointUVD, Mm, Px

from scipy import optimize

import rospy

class GraspCandidateElement:
    # ハンド情報、画像情報、局所的情報（ポイント、値）、しきい値
    def __init__(self, hand_radius_px: Px, finger_radius_px: Px, depth: Image, contour: np.ndarray,
                 center: ImagePointUV, insertion_point: ImagePointUV,
                 insertion_th: float, contact_th: float, bw_depth_th: float):
        self.hand_radius_px = hand_radius_px
        self.finger_radius_px = finger_radius_px

        self.center = center
        self.center_d = depth[center[1], center[0]]
        self.insertion_point = insertion_point
        self.insertion_point_d = None
        self.intersection_point = None
        self.intersection_point_d = None
        self.contact_point = None
        self.contact_point_d = None
        self.insertion_score = 0
        self.contact_score = 0
        self.bw_depth_score = 0
        self.total_score = 0
        self.is_valid_pre = False
        self.is_valid = False
        # validnessチェックに引っかかった箇所を追加していく
        self.debug_infos = []

        # 詳細なスコアリングの前に明らかに不正な候補は弾く
        h, w = depth.shape[:2]
        # centerがフレームインしているのは明らか
        self.is_framein = self._check_framein(h, w, self.insertion_point)
        if not self.is_framein:
            self.debug_infos.append(("framein", self.insertion_point))
            return

        self.insertion_point_d = depth[insertion_point[1], insertion_point[0]]
        self.is_valid_pre = self.is_framein and self._precheck_validness()
        if not self.is_valid_pre:
            self.debug_infos.append(
                ("precheck", (self.center_d, self.insertion_point_d)))
            return

        # TODO: ハンドの開き幅調整可能な場合 insertion point = contact pointとなるので、insertionのスコアはいらない
        # 挿入点の評価
        self.intersection_point = self._compute_intersection_point(contour)
        _, intersection_max_d, intersection_mean_d = self._compute_depth_profile_in_finger_area(
            depth, self.intersection_point, finger_radius_px)
        min_d = intersection_mean_d
        max_d = max(intersection_max_d, self.insertion_point_d)
        self.insertion_score = self._compute_point_score(
            depth, min_d, max_d, self.insertion_point)
        if self.insertion_score < insertion_th:
            self.debug_infos.append(("insertion_score", self.insertion_score))
            return
        # 接触点の計算と評価
        self.contact_point = self._compute_contact_point(
            self.intersection_point)
        self.contact_score = self._compute_point_score(
            depth, min_d, max_d, self.contact_point)
        if self.contact_score < contact_th:
            self.debug_infos.append(("contact_score", self.contact_score))
            return
        # 挿入点と接触点の間の障害物の評価
        self.bw_depth_score = self._compute_bw_depth_score(
            depth, self.contact_point)
        if self.bw_depth_score < bw_depth_th:
            self.debug_infos.append(("bw_depth_score", self.bw_depth_score))
            return
        self.total_score = self._compute_total_score()
        # すべてのスコアが基準を満たしたときのみvalid判定
        self.is_valid = True

    def _check_framein(self, h: Px, w: Px, pt: ImagePointUV) -> bool:
        return not (pt[0] < 0 or pt[1] < 0 or pt[0] >= w or pt[1] >= h)

    def _precheck_validness(self) -> bool:
        is_valid_pre = \
            self._check_depth_existance() and \
            self._check_depth_difference()
        return is_valid_pre

    def _check_depth_existance(self) -> bool:
        return self.center_d != 0 and self.insertion_point_d != 0

    def _check_depth_difference(self) -> bool:
        return self.center_d <= self.insertion_point_d

    def _compute_intersection_point(self, contour: np.ndarray) -> ImagePointUV:
        # TODO: 個々のcropはまだ上位に引き上げられそう
        x, y, w, h = cv2.boundingRect(contour)
        upper_left_point = np.array((x, y))
        shifted_contour = contour - upper_left_point
        shifted_center, shifted_edge = [
            tuple(pt - upper_left_point) for pt in (self.center, self.insertion_point)]

        shifted_intersection = self._compute_intersection_between_contour_and_line(
            (h, w), shifted_contour, shifted_center, shifted_edge)
        intersection = tuple(shifted_intersection + upper_left_point)
        return intersection
    
    def _compute_intersection_between_contour_and_line(self, img_shape, contour, line_pt1_xy, line_pt2_xy):
        """
        輪郭と線分の交点座標を取得する
        TODO: 線分ごとに描画と論理積をとり非効率なので改善方法要検討
        only here
        """
        blank_img = np.zeros(img_shape)
        # クロップ前に計算したcontourをクロップ後の画像座標に変換し描画
        cnt_img = blank_img.copy()
        cv2.drawContours(cnt_img, [contour], -1, 255, 1, lineType=cv2.LINE_AA)
        # クロップ前に計算したlineをクロップ後の画像座標に変換し描画
        line_img = blank_img.copy()
        # 斜めの場合、ピクセルが重ならない場合あるのでlineはthicknessを２にして平均をとる
        line_img = cv2.line(line_img, line_pt1_xy, line_pt2_xy,
                            255, 2, lineType=cv2.LINE_AA)
        # バイナリ画像(cnt_img, line_img)のbitwiseを用いて、contourとlineの交点を検出
        bitwise_img = blank_img.copy()
        cv2.bitwise_and(cnt_img, line_img, bitwise_img)

        intersections = [(w, h)
                        for h, w in zip(*np.where(bitwise_img > 0))]  # hw to xy
        if len(intersections) == 0:
            raise Exception(
                "No intersection between a contour and a candidate element, 'hand_radius_mm' or 'fp' may be too small")
        mean_intersection = np.int0(np.round(np.mean(intersections, axis=0)))
        return mean_intersection

    def _compute_contact_point(self, intersection_point: ImagePointUV) -> ImagePointUV:
        direction_v = np.array(self.insertion_point) - np.array(self.center)
        unit_direction_v = direction_v / np.linalg.norm(direction_v, ord=2)
        # 移動後座標 = 移動元座標 + 方向ベクトル x 移動量(指半径[pixel])
        contact_point = np.int0(
            np.round(intersection_point + unit_direction_v * self.finger_radius_px))
        return contact_point

    def _compute_point_score(self, depth: Image, min_d: Mm, max_d: Mm, pt: ImagePointUV) -> float:
        # TODO: 引数のmin, maxはターゲットオブジェクト周辺の最小値・最大値
        _, _, mean_d = self._compute_depth_profile_in_finger_area(
            depth, pt, self.finger_radius_px)
        score = max(0, (mean_d - min_d)) / (max_d - min_d + 1e-6)
        return score

    def _compute_bw_depth_score(self, depth: Image, contact_point: ImagePointUV) -> float:
        min_d, max_d, mean_d = self._compute_bw_depth_profile(
            depth, contact_point, self.insertion_point)
        # score = max(0, (mean_d - min_d)) / (max_d - min_d + 1e-6)
        score = 1 - ((mean_d - min_d) / (max_d - min_d + 1e-6))
        return score
    
    def _compute_bw_depth_profile(self, depth, contact_point, insertion_point):
        """
        contact_pointからinsertion_pointの間での深さの最小値・最大値・平均
        """
        values = extract_depth_between_two_points(
            depth, contact_point, insertion_point)
        min_depth, max_depth = values.min(), values.max()
        # 欠損ピクセルの値は除外
        valid_values = values[values > 0]
        mean_depth = np.mean(valid_values) if len(valid_values) > 0 else 0
        return min_depth, max_depth, mean_depth
    
    def _compute_depth_profile_in_finger_area(self, depth: Image, pt: ImagePointUV, finger_radius_px: Px) -> Tuple[Mm, Mm, Mm]:
        """
        ptを中心とする指が差し込まれる範囲における深さの最小値・最大値・平均
        """
        h, w = depth.shape[:2]
        rounded_radius = np.int0(np.round(finger_radius_px))
        r_slice = slice(max(0, pt[1] - rounded_radius),
                        min(pt[1] + rounded_radius + 1, h))
        c_slice = slice(max(0, pt[0] - rounded_radius),
                        min(pt[0] + rounded_radius + 1, w))
        cropped_depth = depth[r_slice, c_slice]
        finger_mask = np.zeros_like(cropped_depth, dtype=np.uint8)
        cropped_h, cropped_w = cropped_depth.shape[:2]
        cv2.circle(
            finger_mask, (cropped_w // 2, cropped_h // 2), rounded_radius, 255, -1)
        depth_values_in_mask = cropped_depth[finger_mask == 255]
        return int(np.min(depth_values_in_mask)), int(np.max(depth_values_in_mask)), int(np.mean(depth_values_in_mask))

    def _compute_total_score(self) -> float:
        # TODO: ip, cp間のdepthの評価 & 各項の重み付け
        return self.insertion_score * self.contact_score * self.bw_depth_score

    def get_points(self):
        return {"center": self.center, "intersection": self.intersection_point, "contact": self.contact_point, "insertion": self.insertion_point}

    def get_scores(self):
        return {"insertion": self.insertion_score, "contact": self.contact_score, "bw_depth": self.bw_depth_score}

    def draw(self, img: Image, line_color=(0, 0, 0), line_thickness=1, circle_thickness=1, show_circle=True) -> None:
        cv2.line(img, self.center, self.insertion_point,
                 line_color, line_thickness, cv2.LINE_AA)
        if show_circle:
            rounded_radius = np.int0(np.round(self.finger_radius_px))
            cv2.circle(img, self.insertion_point, rounded_radius,
                       (255, 0, 0), circle_thickness, cv2.LINE_AA)
            cv2.circle(img, self.contact_point, rounded_radius,
                       (0, 255, 0), circle_thickness, cv2.LINE_AA)

    # TODO: 重複関数の統合
    def get_intersection_point_uv(self) -> ImagePointUV:
        return self.intersection_point

    def get_contact_point_uv(self) -> ImagePointUV:
        return self.contact_point

    def get_insertion_point_uv(self) -> ImagePointUV:
        return self.insertion_point

    def get_intersection_point_uvd(self) -> ImagePointUVD:
        return (self.get_intersection_point_uv(), self.intersection_point_d)

    def get_contact_point_uvd(self) -> ImagePointUVD:
        return (self.get_contact_point_uv(), self.contact_point_d)

    def get_insertion_point_uvd(self) -> ImagePointUVD:
        return (self.get_insertion_point_uv(), self.insertion_point_d)


class GraspCandidate:
    def __init__(self, hand_radius_px, finger_radius_px, angle, depth, contour,
                 center, insertion_points,
                 elements_th, el_insertion_th, el_contact_th, el_bw_depth_th):
        self.hand_radius_px = hand_radius_px
        self.finger_radius_px = finger_radius_px
        self.angle = angle
        # self.bbox_short_radius = bbox_short_radius
        self.center = center
        self.center_d = depth[center[1], center[0]]

        self.is_valid = False
        self.is_framein = False
        self.shifted_center = None
        self.elements_score = 0
        self.total_score = 0
        self.debug_infos = []

        # centerがマスク内になければinvalid
        # -1: 外側、0: 輪郭上、1: 内側
        self.is_center_inside_mask = cv2.pointPolygonTest(
            contour, (int(center[0]), int(center[1])), False) <= 0
        if self.is_center_inside_mask:
            self.debug_infos.append(
                ("center_inside_mask", self.is_center_inside_mask))
            return
        self.elements = [
            GraspCandidateElement(
                hand_radius_px=hand_radius_px, finger_radius_px=finger_radius_px, depth=depth, contour=contour,
                center=center, insertion_point=insertion_point,
                insertion_th=el_insertion_th, contact_th=el_contact_th, bw_depth_th=el_bw_depth_th
            ) for insertion_point in insertion_points
        ]

        self.is_framein = self._merge_elements_framein()
        if not self.is_framein:
            self.debug_infos.append(("framein", self.is_framein))
            return
        self.elements_is_valid = self._merge_elements_validness()
        if not self.elements_is_valid:
            self.debug_infos.append(("elements_valid", self.elements_is_valid))
            return
        # elementの組み合わせ評価
        self.elements_score = self._compute_elements_score()
        if self.elements_score < elements_th:
            self.debug_infos.append(("elements_score", self.elements_score))
            return
        # contact pointsの中心とマスクの中心のズレの評価
        self.shifted_center = self._compute_contact_points_center()
        # 各スコアの合算
        self.total_score = self._compute_total_score()
        # すべてのスコアが基準を満たしたときのみvalid判定
        self.is_valid = True

    def _merge_elements_framein(self) -> bool:
        return np.all([el.is_framein for el in self.elements])

    def _merge_elements_validness(self) -> bool:
        return np.all([el.is_valid for el in self.elements])

    def _compute_contact_points_center(self) -> ImagePointUV:
        contact_points = self.get_contact_points_uv()
        return np.int0(np.round(np.mean(contact_points, axis=0)))

    def _compute_elements_score(self) -> float:
        element_scores = self.get_element_scores()
        # return np.prod(element_scores)
        # return np.mean(element_scores) * (np.min(element_scores) / np.max(element_scores))
        
        # min_score = np.min(element_scores)
        # return (np.mean(element_scores) - min_score) / (np.max(element_scores) - min_score + 10e-6)
        
        # マクシミン的な戦略
        # スコアの悪い指が含まれない把持候補を選択したいので、各候補の最悪な指のスコアを比較する。
        return np.min(element_scores) # マクシミン的な戦略

    def _compute_total_score(self) -> float:
        return self.elements_score
    
    def get_center_uv(self) -> ImagePointUV:
        return self.center

    def get_intersection_points_uv(self) -> List[ImagePointUV]:
        return [el.get_intersection_point_uv() for el in self.elements]

    def get_contact_points_uv(self) -> List[ImagePointUV]:
        return [el.get_contact_point_uv() for el in self.elements]

    def get_insertion_points_uv(self) -> List[ImagePointUV]:
        return [el.get_insertion_point_uv() for el in self.elements]

    def get_center_uvd(self) -> ImagePointUVD:
        return (self.center, self.center_d)

    def get_contact_points_uvd(self) -> List[ImagePointUVD]:
        return [el.get_contact_point_uvd() for el in self.elements]

    def get_intersection_points_uvd(self) -> List[ImagePointUVD]:
        return [el.get_intersection_point_uvd() for el in self.elements]

    def get_insertion_points_uvd(self) -> List[ImagePointUVD]:
        return [el.get_insertion_point_uvd() for el in self.elements]

    def get_element_scores(self) -> List[float]:
        return [el.total_score for el in self.elements]

    def get_scores(self):
        return {"elements": self.elements_score}

    def draw(self, img, line_color=(0, 0, 0), line_thickness=1, circle_thickness=1, show_circle=True) -> None:
        for el in self.elements:
            el.draw(img, line_color, line_thickness,
                    circle_thickness, show_circle)


class GraspDetector:
    # TODO: hand_radiusの追加
    def __init__(self, finger_num: int, hand_radius_mm: Mm, finger_radius_mm: Mm, unit_angle: int,
                 frame_size: Tuple[int, int], fp: float,
                 elements_th: float = 0., el_insertion_th: float = 0.5, 
                 el_contact_th: float = 0.5, el_bw_depth_th: float = 0.):
        self.finger_num = finger_num
        self.unit_angle = unit_angle  # 生成される把持候補の回転の刻み角
        self.elements_th = elements_th
        self.el_insertion_th = el_insertion_th
        self.el_contact_th = el_contact_th
        self.el_bw_depth_th = el_bw_depth_th

        # NOTE: mm, pxの変換は深度、解像度、受光素子のサイズに依存するのでdetect時に変換
        self.hand_radius_mm = hand_radius_mm
        self.finger_radius_mm = finger_radius_mm

        self.h, self.w = frame_size
        self.fp = fp

        self.base_angle = 360 // self.finger_num  # ハンドの指間の角度 (等間隔前提)
        self.candidate_element_num = 360 // self.unit_angle
        self.candidate_num = self.base_angle // self.unit_angle

        self.base_rmat = self._compute_rmat(self.base_angle)
        self.unit_rmat = self._compute_rmat(self.unit_angle)

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

    def detect(self, center: ImagePointUV, depth: Image, contour: np.ndarray) -> List[GraspCandidate]:
        # 単位変換
        center_d = depth[center[1], center[0]]
        # center_dが欠損すると0 divisionになるので注意
        hand_radius_px = self._convert_mm_to_px(self.hand_radius_mm, center_d)
        finger_radius_px = self._convert_mm_to_px(
            self.finger_radius_mm, center_d)
        # ベクトルははじめの角度求めるとかで関数内部で計算してもいいかも
        unit_vector = np.array([0, -1])
        base_finger_v = unit_vector * hand_radius_px  # 単位ベクトル x ハンド半径
        candidates = []

        try:
            for i in range(self.candidate_num):
                finger_v = base_finger_v
                insertion_points = self.compute_insertion_points(
                    center, finger_v)
                angle = self.unit_angle * i
                cnd = GraspCandidate(hand_radius_px=hand_radius_px, finger_radius_px=finger_radius_px, angle=angle,
                                        depth=depth, contour=contour, center=center, insertion_points=insertion_points,
                                        elements_th=self.elements_th, el_insertion_th=self.el_insertion_th, 
                                        el_contact_th=self.el_contact_th, el_bw_depth_th=self.el_bw_depth_th
                                        )
                candidates.append(cnd)

                base_finger_v = np.dot(base_finger_v, self.unit_rmat)

        except Exception:
            pass

        return candidates
    

    def is_in_image(self, h, w, i, j) -> bool:
        # not in use now
        if i < 0 or i >= h or j < 0 or j >= w:
            return False
        return True
    

    def drawResult(self, img, depth, x, y, t, r):
        # not in use now
        height, width = depth.shape[0], depth.shape[1]
        center_h, center_w = height // 2, width // 2
        center_d = depth[center_h, center_w]

        print("r mm" , r)
        r = self._convert_mm_to_px(r, center_d)

        print("r px", r)

        point1x, point1y = int(x - r * np.sin(np.deg2rad(t))), int(y + r * np.cos(np.deg2rad(t)))
        point2x, point2y = int(x - r * np.sin(np.deg2rad(t+120))), int(y + r * np.cos(np.deg2rad(t+120)))
        point3x, point3y = int(x - r * np.sin(np.deg2rad(t+240))), int(y + r * np.cos(np.deg2rad(t+240)))

        print("####", point1x, point1y, point2x, point2y, point3x, point3y)


        # cv2.circle(img, (point1x, point1y), 15, (255, 0, 255), thickness=-1)
        # cv2.circle(img, (point2x, point2y), 15, (255, 0, 255), thickness=-1)
        # cv2.circle(img, (point3x, point3y), 15, (255, 0, 255), thickness=-1)

        cv2.circle(img, (point1y, point1x), 15, (255, 0, 255), thickness=-1)
        cv2.circle(img, (point2y, point2x), 15, (255, 0, 255), thickness=-1)
        cv2.circle(img, (point3y, point3x), 15, (255, 0, 255), thickness=-1)

        hand_radius_px = self._convert_mm_to_px(self.hand_radius_mm, center_d)
        hand_radius_px_min = self._convert_mm_to_px(self.hand_radius_mm / 2, center_d)

        for xi in range(-50, 50, 10):
            for yi in range(-50, 50, 10):
                cv2.circle(img, (center_w + yi, center_h + xi), 5, (0, 0, 255), thickness=-1)

        # for ri in np.linspace(hand_radius_px_min, hand_radius_px, 10):
        #     rr = int(ri)
        #     cv2.circle(img, (y, x), rr, (0, 255, 0), thickness=2)

        return img
    

    def calcurate_insertion(self, depth):
        # not in use now
        # 単位変換
        height, width = depth.shape[0], depth.shape[1]

        center_h, center_w = height // 2, width // 2
        center_d = depth[center_h, center_w]

        # center_dが欠損すると0 divisionになるので注意
        hand_radius_px = self._convert_mm_to_px(self.hand_radius_mm, center_d)
        hand_radius_px_min = self._convert_mm_to_px(self.hand_radius_mm / 2, center_d)
        # finger_radius_px = self._convert_mm_to_px(self.finger_radius_mm, center_d)
        # ベクトルははじめの角度求めるとかで関数内部で計算してもいいかも

        max_score = 0
        best_x = -1
        best_y = -1
        best_t = -1
        best_r = -1

        best_xi = -1
        best_yi = -1
        best_ti = -1
        best_ri = -1

        for ti in range(0, 120, 5):
            for xi in range(-50, 50, 10):
                for yi in range(-50, 50, 10):
                    for ri in np.linspace(hand_radius_px_min, hand_radius_px, 10):
                        rr = ri
                        point1x, point1y = center_h + xi - rr * np.sin(np.deg2rad(ti)), center_w + yi + rr * np.cos(np.deg2rad(ti))
                        point2x, point2y = center_h + xi - rr * np.sin(np.deg2rad(ti+120)), center_w + yi + rr * np.cos(np.deg2rad(ti+120))
                        point3x, point3y = center_h + xi - rr * np.sin(np.deg2rad(ti+240)), center_w + yi + rr * np.cos(np.deg2rad(ti+240))

                        if not self.is_in_image(height, width, point1x, point1y):
                            continue
                        if not self.is_in_image(height, width, point2x, point2y):
                            continue
                        if not self.is_in_image(height, width, point3x, point3y):
                            continue

                        # tmp_score = depth[int(point1x)][int(point1y)] + depth[int(point2x)][int(point2y)] + depth[int(point3x)][int(point3y)]
                        tmp_score = min(min(depth[int(point1x)][int(point1y)], depth[int(point2x)][int(point2y)]), depth[int(point3x)][int(point3y)])
                        # print("tmp_score", tmp_score)
                        if tmp_score > max_score:
                            best_x = xi + center_h
                            best_y = yi + center_w
                            best_t = ti
                            best_r = rr

                            best_xi = xi
                            best_yi = yi
                            best_ti = ti
                            best_ri = ri

                            max_score = tmp_score
                            

        print("xi, yi, ti, ri", best_xi, best_yi, best_ti, best_ri)
        best_r = self._convert_px_to_mm(best_r, center_d) # mm単位に


        return best_x, best_y, best_t, best_r


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
        self.index = self.get_target_index(contours)
        center = centers[self.index]
        center_d = depth[center[1], center[0]]

        # TMP 動作確認
        if center_d == 0:
            center_d = 600

        hand_radius_px = self._convert_mm_to_px(self.hand_radius_mm, center_d)
        finger_radius_px = self._convert_mm_to_px(self.finger_radius_mm, center_d)
        # ベクトルははじめの角度求めるとかで関数内部で計算してもいいかも

        ellipse = cv2.fitEllipse(contours[self.index])

        xytr_list = []
        for i in range(self.candidate_num):
            theta = np.radians(i * self.unit_angle)
            xyr = self.get_insertion_points(ellipse, theta, hand_radius_px, finger_radius_px)
            xytr = np.insert(xyr, 2, theta)
            xytr_list.append(xytr)

        print("index : ", self.index)
 
        cabbage_size_mm = self._convert_px_to_mm(max(ellipse[1][0], ellipse[1][1]), center_d)

        # 画像中心にキャベツが見つからなかったとき
        if self.index == -1:
            center_d = -1


        return xytr_list, center, center_d, finger_radius_px, cabbage_size_mm
    
    def point_average_depth(self, hi, wi, depth: Image, finger_radius_px):
        mask = np.zeros_like(depth)
        cv2.circle(mask, (wi, hi), int(finger_radius_px), 1, thickness=-1)
        mask = mask.astype(np.bool)

        return depth[mask].mean()
    
    def calculate(self, depth: Image, contours: np.ndarray, centers):
        # center_d = depth[self.h // 2, self.w // 2]
        # center_d = 600 # TODO

        xytr_list, center, center_d, finger_radius_px, cabbage_size_mm = self.get_xytr_list(depth, contours, centers)

        max_score = -1
        best_x = -1
        best_y = -1
        best_t = -1
        best_r = -1

        print("CABBAGE SIZE", cabbage_size_mm)

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
                                self.point_average_depth(hi, wi, depth, finger_radius_px))
                
            # cabbage_size_mm * 0.6でキャベツの高さをイメージ
            if (worst_depth - center_d) / (cabbage_size_mm * 0.6) > 1:
                good_xytr_list.append([xi, yi, ti, ri])
                    

            if update and worst_depth > max_score:
                best_x = int(xi) 
                best_y = int(yi)
                best_t = ti
                best_r = ri

                max_score = worst_depth        

        rospy.logerr("--------------------------------------------")
        print(len(good_xytr_list))
        if len(good_xytr_list) > 0:
            
            print("good_xytr_list", good_xytr_list)
            dis2 = 100000
            for xi, yi, ti, ri in good_xytr_list:
                if dis2 > (center[0] - xi)**2 + (center[1] - yi)**2:
                    best_x = int(xi) 
                    best_y = int(yi)
                    best_t = ti
                    best_r = ri
                    dis2 = (center[0] - xi)**2 + (center[1] - yi)**2




        print("x, y, t, r", best_x, best_y, best_t, best_r)


        best_r = self._convert_px_to_mm(best_r, center_d) # mm単位に


        return best_x, best_y, best_t, best_r, center_d


    def get_major_minor_ratio(self, contours):


        ellipse = cv2.fitEllipse(contours[self.index])
        a, b = ellipse[1]
        if a < b:
            a, b = b, a
        return b / a
    
    def get_access_distance(self, contours, depth):
        if self.index == -1:
            return -1
        
        ratio  = self.get_major_minor_ratio(contours)
        # TMP
        # 0.6 というのは一番キャベツが立っているときの
        # 長軸と短軸の比として設定している
        return max(ratio - 0.6, 0) / 0.4 * 30 / 1000


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

