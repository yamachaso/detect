#!/usr/bin/env python3
from multiprocessing import Pool
from time import time
from typing import List

import os
from datetime import datetime
import cv2
import numpy as np
import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from detect.msg import (Candidate, Candidates, DetectedObject,
                        GraspDetectionAction, GraspDetectionDebugInfo,
                        GraspDetectionGoal, GraspDetectionResult, PointTuple2D, 
                        CalcurateInsertionAction, CalcurateInsertionGoal, CalcurateInsertionResult)
from geometry_msgs.msg import Point, Pose, PoseStamped
from modules.grasp import GraspDetector, InsertionCalculator
from modules.image import extract_flont_mask_with_thresh, extract_flont_instance_indexes, merge_mask
from modules.visualize import convert_rgb_to_3dgray
from modules.ros.action_clients import (ComputeDepthThresholdClient,
                                        InstanceSegmentationClient, TFClient,
                                        VisualizeClient, ExclusionListClient)
from modules.ros.msg_handlers import RotatedBoundingBoxHandler
from modules.ros.utils import PointProjector, PoseEstimator, multiarray2numpy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from modules.ros.publishers import ImageMatPublisher

from modules.const import CONFIGS_PATH, DATASETS_PATH, OUTPUTS_PATH
from modules.colored_print import *
from scipy import optimize


# def process_instance_segmentation_result_routine(depth, instance_msg, detect_func):
#     instance_center = np.array(instance_msg.center)
#     bbox_handler = RotatedBoundingBoxHandler(instance_msg.bbox)
#     contour = multiarray2numpy(int, np.int32, instance_msg.contour)
#     candidates = detect_func(center=instance_center, depth=depth, contour=contour)
# 
#     return (candidates, instance_center, bbox_handler)
# 
# 
# def create_candidates_msg(original_center, valid_candidates, target_index):
#     return Candidates(candidates=[
#         Candidate(
#             PointTuple2D(cnd.get_center_uv()),
#             [PointTuple2D(pt) for pt in cnd.get_insertion_points_uv()],
#             [PointTuple2D(pt) for pt in cnd.get_contact_points_uv()],
#             cnd.total_score,
#             cnd.is_valid
#         )
#         for cnd in valid_candidates
#     ],
#         # bbox=bbox_handler.msg,
#         center=PointTuple2D(original_center),
#         target_index=target_index
#     )


class GraspDetectionServer:
    def __init__(self, name: str, finger_num: int, unit_angle: int, hand_radius_mm: int, finger_radius_mm: int,
                 hand_mount_rotation: int, info_topic: str, debug: bool):

        rospy.init_node(name, log_level=rospy.INFO)

        self.finger_num = finger_num
        self.base_angle = 360 // finger_num
        self.hand_radius_mm = hand_radius_mm  # length between center and edge
        self.finger_radius_mm = finger_radius_mm
        self.hand_mount_rotation = hand_mount_rotation

        cam_info: CameraInfo = rospy.wait_for_message(info_topic, CameraInfo, timeout=None)
        frame_size = (cam_info.height, cam_info.width)

        # convert unit
        # ref: https://qiita.com/srs/items/e3412e5b25477b46f3bd
        flatten_corrected_params = cam_info.P
        fp_x, fp_y = flatten_corrected_params[0], flatten_corrected_params[5]
        fp = (fp_x + fp_y) / 2

        # Publishers
        self.dbg_info_publisher = rospy.Publisher("/grasp_detection_server/result/debug", GraspDetectionDebugInfo, queue_size=10) if debug else None
        # Action Clients
        self.is_client = InstanceSegmentationClient()
        # self.cdt_client = ComputeDepthThresholdClient() if enable_depth_filter else None
        self.tf_client = TFClient("base_link") # base_linkの座標系に変換するtf変換クライアント
        self.el_client = ExclusionListClient()
        # Others
        self.bridge = CvBridge()
        self.projector = PointProjector(cam_info)
        self.pose_estimator = PoseEstimator()
        self.grasp_detector = GraspDetector(finger_num=finger_num, hand_radius_mm=hand_radius_mm,
                                            finger_radius_mm=finger_radius_mm,
                                            unit_angle=unit_angle, frame_size=frame_size, fp=fp)

        self.insertion_calculator = InsertionCalculator(finger_num=finger_num, hand_radius_mm=hand_radius_mm,
                                            finger_radius_mm=finger_radius_mm,
                                            unit_angle=unit_angle, frame_size=frame_size, fp=fp)


        self.pool = Pool(100)

        self.server = SimpleActionServer(name, GraspDetectionAction, self.callback, False)
        self.server2 = SimpleActionServer('calcurate_insertion_server', CalcurateInsertionAction, self.callback2, False)
        self.server.start()
        self.server2.start()

        self.count = 0
        self.now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        self.result_publisher = ImageMatPublisher("/grasp_detection_server_result", queue_size=10)

        self.approach_center_pose_stamped_msg = None


        rospy.logwarn("finished detection server constructor")

    # def depth_filtering(self, img_msg, depth_msg, img, depth, masks, thresh=0.5, n=5):
    #     vis_base_img_msg = img_msg
    #     flont_indexes = []

    #     if self.cdt_client:
    #         merged_mask = np.where(np.sum(masks, axis=0) > 0, 255, 0).astype("uint8")
    #         # depthの欠損対策だがすでに不要
    #         # merged_mask = np.where(merged_mask * depth > 0, 255, 0).astype("uint8")
    #         whole_mask_msg = self.bridge.cv2_to_imgmsg(merged_mask)
    #         opt_depth_th = self.cdt_client.compute(depth_msg, whole_mask_msg, n=n)
    #         raw_flont_mask = extract_flont_mask_with_thresh(depth, opt_depth_th, n=n)
    #         flont_indexes = extract_flont_instance_indexes(raw_flont_mask, masks, thresh)
    #         flont_mask = merge_mask(np.array(masks)[flont_indexes])
    #         gray_3c = convert_rgb_to_3dgray(img)
    #         reversed_flont_mask = cv2.bitwise_not(flont_mask)
    #         base_img = cv2.bitwise_and(img, img, mask=flont_mask) + \
    #             cv2.bitwise_and(gray_3c, gray_3c, mask=reversed_flont_mask)
    #         vis_base_img_msg = self.bridge.cv2_to_imgmsg(base_img)
    #         rospy.loginfo(opt_depth_th)

    #     return vis_base_img_msg, flont_indexes

    # def compute_object_center_point_stampd(self, c_3d_c_on_surface, header):
    #     c_3d_c = Point(c_3d_c_on_surface.x, c_3d_c_on_surface.y, c_3d_c_on_surface.z)
    #     c_3d_w = self.tf_client.transform_point(header, c_3d_c)
    #     return c_3d_w
        
    def compute_object_center_pose_stampd(self, c_3d_c_on_surface, header):
        c_3d_c = Point(c_3d_c_on_surface.x, c_3d_c_on_surface.y, c_3d_c_on_surface.z)
        c_3d_w = self.tf_client.transform_point(header, c_3d_c)
        # TMP とりあえず使ってないので無視
        # c_orientation = self.pose_estimator.get_orientation(depth, mask)

        return PoseStamped(
            Header(frame_id="base_link"),
            Pose(
                position=c_3d_w.point,
                # orientation=c_orientation
            )
        )

    # def compute_approach_distance(self, c_3d_c_on_surface, insertion_points_c):
    #     # bottom_z = min([pt.z for pt in insertion_points_c])
    #     bottom_z = max([pt.z for pt in insertion_points_c])
    #     top_z = c_3d_c_on_surface.z
    #     # length_to_center = bottom_z - top_z
    #     length_to_center = (bottom_z - top_z) * self.approach_coef  # インスタンス頂点からのアプローチ距離
    #     return length_to_center

    def compute_object_3d_radiuses(self, depth, bbox_handler):
        bbox_short_side_3d, bbox_long_side_3d = bbox_handler.get_sides_3d(self.projector, depth)
        short_raidus = bbox_short_side_3d / 2
        long_radius = bbox_long_side_3d / 2
        return (short_raidus, long_radius)

    def augment_angles(self, angle):
        angles = []
        for i in range(1, self.finger_num + 1):
            raw_rotated_angle = angle - (self.base_angle * i)
            rotated_angle = raw_rotated_angle + 360 if raw_rotated_angle < -360 else raw_rotated_angle
            reversed_rotated_angle = rotated_angle + 360
            angles.extend([rotated_angle, reversed_rotated_angle])
        angles.sort(key=abs)
        return angles
    
    def instances2centers_contours_masks(self, depth, instances):
        centers = [np.array(instance_msg.center) for instance_msg in instances]
        contours = [multiarray2numpy(int, np.int32, instance_msg.contour) for instance_msg in instances]
        masks = [self.bridge.imgmsg_to_cv2(instance_msg.mask) for instance_msg in instances]
    
        return (centers, contours, masks)


    def distance_point_between_line(self, px, py, x1, y1, x2, y2):
        a = y2 - y1
        b = x1 - x2
        c = -x1 * y2 + x2 * y1
        return np.abs(a * px + b * py + c) / np.sqrt(a * a + b * b)
    
    def get_corner_coordinate(self):
        header = Header()
        header.frame_id = "container_br"
        br_point = self.tf_client.transform_point(header, Point()).point      
        header.frame_id = "container_tr"
        tr_point = self.tf_client.transform_point(header, Point()).point
        header.frame_id = "container_tl"
        tl_point = self.tf_client.transform_point(header, Point()).point
        header.frame_id = "container_bl"
        bl_point = self.tf_client.transform_point(header, Point()).point
        
        return br_point.x, br_point.y, tr_point.x, tr_point.y, tl_point.x, tl_point.y, bl_point.x, bl_point.y
    
    def compute_contours_list(self, img_shape, center, contour):
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

    # def check_wall_contact(self, pose_stamped_msg):
    def check_wall_contact(self, contours_3d_list):
       
        contact_dis = 0.05 # 15cm以内だったらコンテナと接触している
        r, t, l, b = 1, 2, 4, 8
        res = 0

        wall_distance_r = 100000000
        wall_distance_t = 100000000
        wall_distance_l = 100000000
        wall_distance_b = 100000000

        for pose_stamped_msg in contours_3d_list:
            px = pose_stamped_msg.pose.position.x
            py = pose_stamped_msg.pose.position.y
            # px = point_stamped_msg.point.x
            # py = point_stamped_msg.point.y


            br_x, br_y, tr_x, tr_y, tl_x, tl_y, bl_x, bl_y = self.get_corner_coordinate()


            if self.distance_point_between_line(px, py, br_x, br_y, tr_x, tr_y) < contact_dis:
                res |= r
            if self.distance_point_between_line(px, py, tr_x, tr_y, tl_x, tl_y) < contact_dis:
                res |= t
            if self.distance_point_between_line(px, py, tl_x, tl_y, bl_x, bl_y) < contact_dis:
                res |= l
            if self.distance_point_between_line(px, py, bl_x, bl_y, br_x, br_y) < contact_dis:
                res |= b

            wall_distance_r = min(wall_distance_r, self.distance_point_between_line(px, py, br_x, br_y, tr_x, tr_y))
            wall_distance_t = min(wall_distance_t, self.distance_point_between_line(px, py, tr_x, tr_y, tl_x, tl_y))
            wall_distance_l = min(wall_distance_l, self.distance_point_between_line(px, py, tl_x, tl_y, bl_x, bl_y))
            wall_distance_b = min(wall_distance_b, self.distance_point_between_line(px, py, bl_x, bl_y, br_x, br_y))

        printb("wall_distance")
        printb("right : {}, top : {}, left : {}, bottom : {}".format(wall_distance_r, wall_distance_t, wall_distance_l, wall_distance_b))

        return res

    def callback(self, goal: GraspDetectionGoal):
        printb("grasp detect callback called")
        # 0: left, 1: right
        arm_index = goal.arm_index
        img_msg = goal.image
        depth_msg = goal.depth
        points_msg = goal.points
        frame_id = img_msg.header.frame_id
        stamp = img_msg.header.stamp
        header = Header(frame_id=frame_id, stamp=stamp)
        try:
            start_time = time()
            img = self.bridge.imgmsg_to_cv2(img_msg)
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
            instances = self.is_client.predict(img_msg) # List[Instance]

            vis_base_img_msg = img_msg

            (centers, contours, masks) = self.instances2centers_contours_masks(depth, instances)

            printr("center length : {}".format(len(centers)))

            cor_coos = self.get_corner_coordinate()
            exclusion_list = self.el_client.ref(arm_index)
            printc("exclusion_list : {}".format(exclusion_list))
            target_index, result_img, score, centers, contours = self.grasp_detector.detect(img, depth, centers, contours, masks, cor_coos, exclusion_list) # 一番スコアの良いキャベツのインデックス

            printr("center length later: {}".format(len(centers)))
            printr("contours length later: {}".format(len(contours)))

            self.result_publisher.publish(result_img, frame_id, stamp)
       

            c_3d_c_on_surface = self.projector.screen_to_camera_2(points_msg, centers[target_index])
            # insertion_points_c = [self.projector.screen_to_camera(uv, d_mm) for uv, d_mm in best_cand.get_insertion_points_uvd()]
            # c_3d_c_on_surface = self.projector.screen_to_camera(*best_cand.get_center_uvd())
            # compute approach distance
            # length_to_center = self.compute_approach_distance(c_3d_c_on_surface, insertion_points_c)
            length_to_center = 0.2
            # compute center pose stamped (world coords)
            # insertion_points_msg = [pt.point for pt in self.tf_client.transform_points(header, insertion_points_c)]

            
            center_pose_stamped_msg = self.compute_object_center_pose_stampd(c_3d_c_on_surface, header)
            # center_pose_stamped_msg = self.compute_object_center_pose_stampd(depth, masks[target_index], c_3d_c_on_surface, header)

            printb("####")
            contours_list = self.compute_contours_list(img.shape[:2], centers[target_index], contours[target_index])
            printb("$$$$")
            contours_3d_list = [self.compute_object_center_pose_stampd(
                self.projector.screen_to_camera_2(points_msg, contours_list_i), header
            ) for contours_list_i in contours_list]
            contact = self.check_wall_contact(contours_3d_list)


            u, v = centers[target_index]
            point_tuple_2d = PointTuple2D(uv=[u ,v])
            # 絶対値が最も小さい角度
            # nearest_angle = self.augment_angles(angle)[0]
            nearest_angle = 0
            print(score)
            object = DetectedObject(
                center=point_tuple_2d,
                # points=insertion_points_msg,
                center_pose=center_pose_stamped_msg,
                # angle=nearest_angle,
                # short_radius=short_radius_3d,
                # long_radius=long_radius_3d,
                length_to_center=length_to_center,
                # score=best_cand.total_score,
                score=score,
                contact=contact
            )

            self.approach_center_pose_stamped_msg = center_pose_stamped_msg

            # if self.dbg_info_publisher:
            #     self.dbg_info_publisher.publish(GraspDetectionDebugInfo(header, candidates_list))
            spent = time() - start_time
            print(f"stamp: {stamp.to_time()}, spent: {spent:.3f}")
            self.server.set_succeeded(GraspDetectionResult(header, object))

        except Exception as err:
            rospy.logerr(err)


    def convert_mm_to_angle(self, r_target):
        # 引数 r が 60 ~ 100の間でないと精度が低い
        r0 = 95 # 付け根部分の関係
        l = 180 # 指の長さ

        # 解きたい関数をリストで戻す
        def func(x, r):
            t = x[0]
            equations = [
                (r - r0) * t + l * np.sin(np.radians(70)) - l*np.sin(t + np.radians(70))
            ]
            return equations

        # 制約を設定
        def constraint_func(x):
            return x[0] - 0.6  # t >= 0.6

        # 初期値
        cons = (
            {'type': 'ineq', 'fun': constraint_func}
        )

        # 最適化を実行
        initial_guess = [0.0]
        result = optimize.minimize(lambda x: np.sum(np.array(func(x, r_target))**2), initial_guess, constraints=cons, method="SLSQP")
        return result.x[0]

    def convert_angle_to_pressure(self, angle):
        # degree
        a = 53.77
        b = 21.79
        # c = 10.84
        # c = 30
        c = 40
        print(angle)
        if b * b - 4 * a  *(c - angle) < 0:
            return 0
        else:
            return (- b + np.sqrt(b * b - 4 * a  *(c - angle)) ) / (2 * a)

    def convert_mm_to_pascal(self, r):
        angle = np.rad2deg(self.convert_mm_to_angle(r))
        pressure = self.convert_angle_to_pressure(angle)

        # print("\033[92m{}\033[0m".format("radius"))
        # print("\033[92m{}\033[0m".format(r))
        # print("\033[92m{}\033[0m".format("angle"))
        # print("\033[92m{}\033[0m".format(angle))
        # print("\033[92m{}\033[0m".format("pressure"))
        # print("\033[92m{}\033[0m".format(pressure))
        return pressure
    
    def distance_btw_pose_stamped(self, a_msg, b_msg):
        x = a_msg.pose.position.x - b_msg.pose.position.x
        y = a_msg.pose.position.y - b_msg.pose.position.y
        z = a_msg.pose.position.z - b_msg.pose.position.z

        return np.sqrt(x**2 + y**2 + z**2)


    def callback2(self, goal: CalcurateInsertionGoal):
        printb("calculate insertion callback called")
        img_msg = goal.image
        depth_msg = goal.depth
        points_msg = goal.points
        frame_id = img_msg.header.frame_id
        stamp = img_msg.header.stamp
        header = Header(frame_id=frame_id, stamp=stamp)
        try:
            start_time = time()
            img = self.bridge.imgmsg_to_cv2(img_msg)
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
            instances = self.is_client.predict(img_msg) # List[Instance]
            # TODO: depthしきい値を求めるためにmerged_maskが必要だが非効率なので要改善
            # print(instances[0].contour)
            # print(type(instances[0].contour))

            # contours = np.array([multiarray2numpy(int, np.int32, instance_msg.contour) for instance_msg in instances], dtype=object)
            contours = np.array([multiarray2numpy(int, np.int32, instance_msg.contour) for instance_msg in instances])
            centers = np.array([np.array(instance_msg.center) for instance_msg in instances])

            centers_3d_list = [self.compute_object_center_pose_stampd(
                self.projector.screen_to_camera_2(points_msg, centers_i), header
            ) for centers_i in centers]

            distance_list = [self.distance_btw_pose_stamped(self.approach_center_pose_stamped_msg, centers_3d_list_i) for centers_3d_list_i in centers_3d_list]

            target_index = np.argmin(distance_list)
            success = True

            printc("distance! : {} m".format(distance_list[target_index]))
            if distance_list[target_index] > 0.1: # 10cm以上異なったらそれは別のキャベツなのでNG
                success = False

            contour = contours[target_index]
            center = centers[target_index]

            ic_result = self.insertion_calculator.calculate(depth, contour, center)

            x, y, t, r, d = ic_result # r はmm単位

            # 把持のとき、ハンドとキャベツをどのくらい離すか？
            access_distance = self.insertion_calculator.get_access_distance(contour)

            print("DDDDD:", d)
            if d < 0:
                success = False

            if success:
                img_result = self.insertion_calculator.drawResult(img.copy(), contour, x, y, t, r, d)

                frame_id = img_msg.header.frame_id
                stamp = img_msg.header.stamp

                self.result_publisher.publish(img_result, frame_id, stamp)
                # depth_tmp = depth.copy()
                # depth_tmp[depth_tmp > 1000] = 1000
                # depth_img = (depth_tmp / depth_tmp.max() * 255).astype(np.uint8)
                # depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2RGB)
                # img_result2 = self.insertion_calculator.drawResult(depth_tmp, contours, x, y, t, r, d)

                # self.result_publisher.publish(img_result2, frame_id, stamp)

                
                # OUTPUT_DIR = f"{OUTPUTS_PATH}/tmp/{self.now}"
                # print(OUTPUT_DIR)
                # os.makedirs(OUTPUT_DIR, exist_ok=True)
                # os.makedirs(f"{OUTPUT_DIR}/color", exist_ok=True)
                # os.makedirs(f"{OUTPUT_DIR}/depth", exist_ok=True)
                # # print(f'{OUTPUT_DIR}/color/{self.count}.jpg')
                # cv2.imwrite(f'{OUTPUT_DIR}/color/{self.count}.jpg', img_result)
                # cv2.imwrite(f'{OUTPUT_DIR}/depth/{self.count}.jpg', img_result2)
                # self.count += 1
    
                c_3d_c_on_surface = self.projector.screen_to_camera_2(points_msg, (x, y))



                point_stamped_msg = self.tf_client.transform_point(header, c_3d_c_on_surface)

                # print("point_msg")
                # print(point_stamped_msg)

                pose_msg = Pose(
                    position=point_stamped_msg.point
                )
    
                # angle = t
                angle = int(np.rad2deg(t)) - self.hand_mount_rotation
                # 絶対値が最も小さい角度
                nearest_angle = self.augment_angles(angle)[0]

                pressure = self.convert_mm_to_pascal(r)

                spent = time() - start_time
                # print(f"stamp: {stamp.to_time()}, spent: {spent:.3f}, objects: {len(objects)} ({len(instances)})")
                printy(f"stamp: {stamp.to_time()}, spent: {spent:.3f}, angle: {nearest_angle}")

            else:
                pose_msg = Pose()
                nearest_angle = 0
                pressure = 0
                printr("callback2 failed...")

            self.server2.set_succeeded(CalcurateInsertionResult(pose_msg, nearest_angle, access_distance, pressure, success))

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    finger_num = rospy.get_param("finger_num")
    unit_angle = rospy.get_param("unit_angle")
    hand_radius_mm = rospy.get_param("hand_radius_mm")
    finger_radius_mm = rospy.get_param("finger_radius_mm")
    hand_mount_rotation = rospy.get_param("hand_mount_rotation")
    info_topic = rospy.get_param("image_info_topic")
    debug = rospy.get_param("debug")

    GraspDetectionServer(
        "detect_server",
        finger_num=finger_num,
        unit_angle=unit_angle,
        hand_radius_mm=hand_radius_mm,
        finger_radius_mm=finger_radius_mm,
        hand_mount_rotation=hand_mount_rotation,
        info_topic=info_topic,
        debug=debug
    )
    rospy.spin()
