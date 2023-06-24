from typing import List, Tuple

import cv2
import numpy as np

from modules.image import extract_depth_between_two_points
from modules.type import Image, ImagePointUV, ImagePointUVD, Mm, Px

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
                 elements_th, center_diff_th, el_insertion_th, el_contact_th, el_bw_depth_th):
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
        min_score = np.min(element_scores)
        return (np.mean(element_scores) - min_score) / (np.max(element_scores) - min_score + 10e-6)

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
                 elements_th: float = 0., center_diff_th: float = 0.,
                 el_insertion_th: float = 0.5, el_contact_th: float = 0.5, el_bw_depth_th: float = 0.):
        self.finger_num = finger_num
        self.unit_angle = unit_angle  # 生成される把持候補の回転の刻み角
        self.elements_th = elements_th
        self.center_diff_th = center_diff_th
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
                                        elements_th=self.elements_th, center_diff_th=self.center_diff_th,
                                        el_insertion_th=self.el_insertion_th, el_contact_th=self.el_contact_th,
                                        el_bw_depth_th=self.el_bw_depth_th
                                        )
                candidates.append(cnd)

                base_finger_v = np.dot(base_finger_v, self.unit_rmat)

        except Exception:
            pass

        return candidates

