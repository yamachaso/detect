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
                                        VisualizeClient)
from modules.ros.msg_handlers import RotatedBoundingBoxHandler
from modules.ros.utils import PointProjector, PoseEstimator, multiarray2numpy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from modules.ros.publishers import ImageMatPublisher

from modules.const import CONFIGS_PATH, DATASETS_PATH, OUTPUTS_PATH
from scipy import optimize


def process_instance_segmentation_result_routine(depth, instance_msg, detect_func):
    instance_center = np.array(instance_msg.center)
    bbox_handler = RotatedBoundingBoxHandler(instance_msg.bbox)
    contour = multiarray2numpy(int, np.int32, instance_msg.contour)
    candidates = detect_func(center=instance_center, depth=depth, contour=contour)

    return (candidates, instance_center, bbox_handler)


def create_candidates_msg(original_center, valid_candidates, target_index):
    return Candidates(candidates=[
        Candidate(
            PointTuple2D(cnd.get_center_uv()),
            [PointTuple2D(pt) for pt in cnd.get_insertion_points_uv()],
            [PointTuple2D(pt) for pt in cnd.get_contact_points_uv()],
            cnd.total_score,
            cnd.is_valid
        )
        for cnd in valid_candidates
    ],
        # bbox=bbox_handler.msg,
        center=PointTuple2D(original_center),
        target_index=target_index
    )


class GraspDetectionServer:
    def __init__(self, name: str, finger_num: int, unit_angle: int, hand_radius_mm: int, finger_radius_mm: int,
                 hand_mount_rotation: int, approach_coef: float,
                 elements_th: float, el_insertion_th: float, el_contact_th: float, el_bw_depth_th: float,
                 info_topic: str, enable_depth_filter: bool, enable_candidate_filter: bool, debug: bool):
        rospy.init_node(name, log_level=rospy.INFO)

        self.finger_num = finger_num
        self.unit_angle = unit_angle
        self.base_angle = 360 // finger_num
        self.hand_radius_mm = hand_radius_mm  # length between center and edge
        self.finger_radius_mm = finger_radius_mm
        self.hand_mount_rotation = hand_mount_rotation
        self.approach_coef = approach_coef
        self.elements_th = elements_th
        self.el_insertion_th = el_insertion_th
        self.el_contact_th = el_contact_th
        self.el_bw_depth_th = el_bw_depth_th
        self.enable_candidate_filter = enable_candidate_filter
        self.debug = debug
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
        self.cdt_client = ComputeDepthThresholdClient() if enable_depth_filter else None
        self.tf_client = TFClient("base_link") # base_linkの座標系に変換するtf変換クライアント
        self.visualize_client = VisualizeClient()
        # Others
        self.bridge = CvBridge()
        self.projector = PointProjector(cam_info)
        self.pose_estimator = PoseEstimator()
        self.grasp_detector = GraspDetector(finger_num=finger_num, hand_radius_mm=hand_radius_mm,
                                            finger_radius_mm=finger_radius_mm,
                                            unit_angle=unit_angle, frame_size=frame_size, fp=fp,
                                            elements_th=elements_th, el_insertion_th=el_insertion_th, 
                                            el_contact_th=el_contact_th, el_bw_depth_th=el_bw_depth_th)

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
        self.result_publisher = ImageMatPublisher("/result", queue_size=10)
        self.result2_publisher = ImageMatPublisher("/result2", queue_size=10)
        rospy.logwarn("finished detection server constructor")

    def depth_filtering(self, img_msg, depth_msg, img, depth, masks, thresh=0.5, n=5):
        vis_base_img_msg = img_msg
        flont_indexes = []

        if self.cdt_client:
            merged_mask = np.where(np.sum(masks, axis=0) > 0, 255, 0).astype("uint8")
            # depthの欠損対策だがすでに不要
            # merged_mask = np.where(merged_mask * depth > 0, 255, 0).astype("uint8")
            whole_mask_msg = self.bridge.cv2_to_imgmsg(merged_mask)
            opt_depth_th = self.cdt_client.compute(depth_msg, whole_mask_msg, n=n)
            raw_flont_mask = extract_flont_mask_with_thresh(depth, opt_depth_th, n=n)
            flont_indexes = extract_flont_instance_indexes(raw_flont_mask, masks, thresh)
            flont_mask = merge_mask(np.array(masks)[flont_indexes])
            gray_3c = convert_rgb_to_3dgray(img)
            reversed_flont_mask = cv2.bitwise_not(flont_mask)
            base_img = cv2.bitwise_and(img, img, mask=flont_mask) + \
                cv2.bitwise_and(gray_3c, gray_3c, mask=reversed_flont_mask)
            vis_base_img_msg = self.bridge.cv2_to_imgmsg(base_img)
            rospy.loginfo(opt_depth_th)

        return vis_base_img_msg, flont_indexes

    def compute_object_center_pose_stampd(self, depth, mask, c_3d_c_on_surface, header):
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

    def compute_approach_distance(self, c_3d_c_on_surface, insertion_points_c):
        # bottom_z = min([pt.z for pt in insertion_points_c])
        bottom_z = max([pt.z for pt in insertion_points_c])
        top_z = c_3d_c_on_surface.z
        # length_to_center = bottom_z - top_z
        length_to_center = (bottom_z - top_z) * self.approach_coef  # インスタンス頂点からのアプローチ距離
        return length_to_center

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

    def callback(self, goal: GraspDetectionGoal):
        print("receive request")
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
            masks = [self.bridge.imgmsg_to_cv2(instance_msg.mask) for instance_msg in instances]
            # TODO: compute n by camera distance
            print("before")
            # vis_base_img_msg, flont_indexes = self.depth_filtering(img_msg, depth_msg, img, depth, masks, thresh=0.8, n=5)
            vis_base_img_msg = img_msg
            print("after")

            print("instance num : ", len(masks))

            if not self.cdt_client:
                flont_indexes = list(range(len(instances)))
            flont_indexes = list(range(len(instances))) # TMP
            flont_indexes_set = set(flont_indexes)

            objects: List[DetectedObject] = []  # 空のものは省く
            candidates_list: List[Candidates] = []  # 空のものも含む

            print("flont_indexes_set", len(flont_indexes_set))

            # 把持候補の生成 (並列処理)
            routine_args = []
            print("instances loop")
            for i in range(len(instances)):
                # ignore other than instances are located on top of stacks
                # TODO: しきい値で切り出したマスク内に含まれないインスタンスはスキップ
                instance_msg = instances[i]
                mask = masks[i]
                if i not in flont_indexes_set:
                    continue
                routine_args.append((depth, instance_msg, self.grasp_detector.detect))
            results = self.pool.starmap(process_instance_segmentation_result_routine, routine_args)

            print("result loop")
            print(len(results))

            # TODO: 座標変換も並列処理化したい
            for obj_index, (candidates, instance_center, bbox_handler) in enumerate(results):
                if len(candidates) == 0:
                    continue
                # select best candidate

                valid_candidates = [cnd for cnd in candidates if cnd.is_valid] if enable_candidate_filter else candidates
                is_valid = len(valid_candidates) > 0
                valid_scores = [cnd.total_score for cnd in valid_candidates]
                target_index = np.argmax(valid_scores) if is_valid else 0

                # candidates_list は可視化用に空のものも含む
                candidates_list.append(create_candidates_msg(instance_center, valid_candidates, target_index))

                # TODO: is_frameinの判定冗長なので要整理
                include_any_frameout = not np.any([cnd.is_framein for cnd in valid_candidates])
                # if include_any_frameout or not is_valid:
                #     continue

                best_cand = valid_candidates[target_index]
                print("c")

                # 3d projection
                # insertion_points_c = [self.projector.screen_to_camera_2(points_msg, uv) for uv in best_cand.get_insertion_points_uv()]
                print("d")
                c_3d_c_on_surface = self.projector.screen_to_camera_2(points_msg, best_cand.get_center_uv())
                print("e")
                # insertion_points_c = [self.projector.screen_to_camera(uv, d_mm) for uv, d_mm in best_cand.get_insertion_points_uvd()]
                # c_3d_c_on_surface = self.projector.screen_to_camera(*best_cand.get_center_uvd())
                # compute approach distance
                # length_to_center = self.compute_approach_distance(c_3d_c_on_surface, insertion_points_c)
                length_to_center = 0.2
                # compute center pose stamped (world coords)
                print("trans before")
                # insertion_points_msg = [pt.point for pt in self.tf_client.transform_points(header, insertion_points_c)]
                print("trans after")
                center_pose_stamped_msg = self.compute_object_center_pose_stampd(depth, mask, c_3d_c_on_surface, header)
                print("center_pose_stamped_msg after")
                # compute 3d radiuses
                # short_radius_3d, long_radius_3d = self.compute_object_3d_radiuses(depth, bbox_handler)
                short_radius_3d, long_radius_3d = 0, 0
                print("compute_object_3d_radiuses after")
                angle = best_cand.angle - self.hand_mount_rotation
                # 絶対値が最も小さい角度
                # nearest_angle = self.augment_angles(angle)[0]
                nearest_angle = 0
                print("depth depth : ", instance_center, depth[instance_center[1]][instance_center[0]])
                objects.append(DetectedObject(
                    # points=insertion_points_msg,
                    center_pose=center_pose_stamped_msg,
                    angle=nearest_angle,
                    short_radius=short_radius_3d,
                    long_radius=long_radius_3d,
                    length_to_center=length_to_center,
                    # score=best_cand.total_score,
                    score=depth[instance_center[1]][instance_center[0]],
                    index=obj_index  # for visualize
                )
                )
                print("end")
            print("visualize before")
            self.visualize_client.visualize_candidates(vis_base_img_msg, candidates_list, depth_msg)
            print("visualize after")
            if self.dbg_info_publisher:
                self.dbg_info_publisher.publish(GraspDetectionDebugInfo(header, candidates_list))
            spent = time() - start_time
            print(f"stamp: {stamp.to_time()}, spent: {spent:.3f}, objects: {len(objects)} ({len(instances)})")
            self.server.set_succeeded(GraspDetectionResult(header, objects))

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

        print("\033[92m{}\033[0m".format("radius"))
        print("\033[92m{}\033[0m".format(r))
        print("\033[92m{}\033[0m".format("angle"))
        print("\033[92m{}\033[0m".format(angle))
        print("\033[92m{}\033[0m".format("pressure"))
        print("\033[92m{}\033[0m".format(pressure))
        return pressure

    def callback2(self, goal: CalcurateInsertionGoal):
        print("callback2 called")
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
            print(type(instances[0].contour))

            contours = np.array([multiarray2numpy(int, np.int32, instance_msg.contour) for instance_msg in instances], dtype=object)
            centers = np.array([np.array(instance_msg.center) for instance_msg in instances])

            ic_result = self.insertion_calculator.calculate(depth, contours, centers)

            x, y, t, r, d = ic_result

            success = True
            if d < 0:
                success = False

            if success:
                img_result = self.insertion_calculator.drawResult(img.copy(), contours, x, y, t, r, d)

                frame_id = img_msg.header.frame_id
                stamp = img_msg.header.stamp

                self.result_publisher.publish(img_result, frame_id, stamp)
                depth_tmp = depth.copy()
                depth_tmp[depth_tmp > 1000] = 1000
                depth_img = (depth_tmp / depth_tmp.max() * 255).astype(np.uint8)
                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2RGB)
                img_result2 = self.insertion_calculator.drawResult(depth_tmp, contours, x, y, t, r, d)

                # self.result2_publisher.publish(img_result2, frame_id, stamp)

                
                OUTPUT_DIR = f"{OUTPUTS_PATH}/tmp/{self.now}"
                print(OUTPUT_DIR)
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                os.makedirs(f"{OUTPUT_DIR}/color", exist_ok=True)
                os.makedirs(f"{OUTPUT_DIR}/depth", exist_ok=True)
                # print(f'{OUTPUT_DIR}/color/{self.count}.jpg')
                cv2.imwrite(f'{OUTPUT_DIR}/color/{self.count}.jpg', img_result)
                cv2.imwrite(f'{OUTPUT_DIR}/depth/{self.count}.jpg', img_result2)
                self.count += 1
    
                c_3d_c_on_surface = self.projector.screen_to_camera_2(points_msg, (x, y))



                point_msg = self.tf_client.transform_point(header, c_3d_c_on_surface)

                print("point_msg")
                print(point_msg)

                pose_msg = Pose(
                    position=point_msg.point
                )
    
                # angle = t
                angle = int(np.rad2deg(t)) - self.hand_mount_rotation
                # 絶対値が最も小さい角度
                nearest_angle = self.augment_angles(angle)[0]

                pressure = self.convert_mm_to_pascal(r)

                spent = time() - start_time
                # print(f"stamp: {stamp.to_time()}, spent: {spent:.3f}, objects: {len(objects)} ({len(instances)})")
                print(f"stamp: {stamp.to_time()}, spent: {spent:.3f}, angle: {nearest_angle}")

            else:
                pose_msg = Pose()
                nearest_angle = 0
                pressure = 0
                print("callback2 failed...")

            self.server2.set_succeeded(CalcurateInsertionResult(pose_msg, nearest_angle, pressure, success))

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    finger_num = rospy.get_param("finger_num")
    unit_angle = rospy.get_param("unit_angle")
    hand_radius_mm = rospy.get_param("hand_radius_mm")
    finger_radius_mm = rospy.get_param("finger_radius_mm")
    hand_mount_rotation = rospy.get_param("hand_mount_rotation")
    approach_coef = rospy.get_param("approach_coef")
    elements_th = rospy.get_param("elements_th")
    el_insertion_th = rospy.get_param("el_insertion_th")
    el_contact_th = rospy.get_param("el_contact_th")
    el_bw_depth_th = rospy.get_param("el_bw_depth_th")
    info_topic = rospy.get_param("image_info_topic")
    enable_depth_filter = rospy.get_param("enable_depth_filter")
    enable_candidate_filter = rospy.get_param("enable_candidate_filter")
    debug = rospy.get_param("debug")

    GraspDetectionServer(
        "detect_server",
        finger_num=finger_num,
        unit_angle=unit_angle,
        hand_radius_mm=hand_radius_mm,
        finger_radius_mm=finger_radius_mm,
        hand_mount_rotation=hand_mount_rotation,
        approach_coef=approach_coef,
        elements_th=elements_th,
        el_insertion_th=el_insertion_th,
        el_contact_th=el_contact_th,
        el_bw_depth_th=el_bw_depth_th,
        info_topic=info_topic,
        enable_depth_filter=enable_depth_filter,
        enable_candidate_filter=enable_candidate_filter,
        debug=debug
    )
    rospy.spin()
