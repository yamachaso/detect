#!/usr/bin/env python3
from typing import List

import cv2
import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from detect.msg import (Candidate, Candidates, VisualizeCandidatesAction,
                        VisualizeCandidatesGoal, VisualizeTargetAction,
                        VisualizeTargetGoal)
# from modules.ros.msg_handlers import RotatedBoundingBoxHandler
from modules.ros.publishers import ImageMatPublisher
from modules.visualize import (convert_rgb_to_3dgray, draw_candidate,
                               get_color_by_score)

import os
from datetime import datetime
import cv2
from modules.const import CONFIGS_PATH, DATASETS_PATH, OUTPUTS_PATH


class VisualizeServer:
    def __init__(self, name: str, pub_topic: str, visualize_only_best_cnd: bool):
        rospy.init_node(name)

        self.bridge = CvBridge()
        self.publisher = ImageMatPublisher(pub_topic, queue_size=10)
        self.visualize_only_best_cnd = visualize_only_best_cnd

        self.last_image = None
        self.last_candidates_list = []
        self.last_frame_id = None
        self.last_stamp = None

        self.servers = []
        self.servers.append(SimpleActionServer(f"{name}_draw_candidates", VisualizeCandidatesAction, self.draw_candidates, False))
        self.servers.append(SimpleActionServer(f"{name}_draw_target", VisualizeTargetAction, self.draw_target, False))

        for server in self.servers:
            server.start()

        ################
        self.count = 0
        self.now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    def draw_candidates(self, goal: VisualizeCandidatesGoal):
        img_msg = goal.base_image
        img = self.bridge.imgmsg_to_cv2(img_msg)
        depth_msg = goal.depth_image
        depth = self.bridge.imgmsg_to_cv2(depth_msg)
        candidates_list: List[Candidates] = goal.candidates_list
        frame_id = img_msg.header.frame_id
        stamp = img_msg.header.stamp

        # res_img = convert_rgb_to_3dgray(img)
        res_img = img
        for cnds_msg in candidates_list:
            candidates: List[Candidate] = cnds_msg.candidates
            obj_center_uv = cnds_msg.center.uv
            cnd_target_index = cnds_msg.target_index
            for cnd_index, cnd_msg in enumerate(candidates):
                cnd_center_uv = cnd_msg.center.uv
                color = get_color_by_score(cnd_msg.score)
                is_target = cnd_index == cnd_target_index
                if self.visualize_only_best_cnd and not is_target:
                    continue
                for pt_msg in cnd_msg.insertion_points:
                    res_img = draw_candidate(res_img, cnd_center_uv, pt_msg.uv, color, is_target=is_target)
                cv2.circle(res_img, cnd_center_uv, 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            cv2.circle(res_img, obj_center_uv, 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            dep = depth[obj_center_uv[1]][obj_center_uv[0]]
            cv2.putText(img,
                text=f'{dep}',
                org=obj_center_uv,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 200, 0),
                thickness=2,
                lineType=cv2.LINE_4)
         

        #################################
        # OUTPUT_DIR = f"{OUTPUTS_PATH}/tmp/{self.now}"
        # os.makedirs(OUTPUT_DIR, exist_ok=True)
        # os.makedirs(f"{OUTPUT_DIR}/cand", exist_ok=True)
        # cv2.imwrite(f'{OUTPUT_DIR}/cand/{self.count}.jpg', res_img)
        # self.count += 1
        #################################

        self.last_image = res_img
        self.last_candidates_list = candidates_list
        self.last_frame_id = frame_id
        self.last_stamp = stamp
        self.publisher.publish(res_img, frame_id, stamp)

    def draw_target(self, goal: VisualizeTargetGoal):
        if self.last_image is None:
            return
        """ 一度candidatesを描画した後に使用すること """
        print(f"candidates_list: {len(self.last_candidates_list)}, target: {goal.target_index}")
        if type(self.last_image) is not None:
            target_obj_index = goal.target_index
            cnds_msg = self.last_candidates_list[target_obj_index]
            target_cnd_index = cnds_msg.target_index
            res_img = self.last_image.copy()
            obj_center_uv = cnds_msg.center.uv
            cnd_msg = cnds_msg.candidates[target_cnd_index]
            cnd_center_uv = cnd_msg.center.uv
            score = cnd_msg.score

            outer_color = (0, 255, 0)
            inner_color = get_color_by_score(score)
            for color, thickness in ((outer_color, 4), (inner_color, 2)):
                for pt_msg in cnd_msg.insertion_points:
                    res_img = draw_candidate(res_img, cnd_center_uv, pt_msg.uv, color, is_target=True, target_thickness=thickness)

            # cv2.circle(res_img, obj_center_uv, 6, (0, 255, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(res_img, f"{score:.2f}", (obj_center_uv[0] + 10, obj_center_uv[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        self.publisher.publish(res_img, self.last_frame_id, self.last_stamp)


if __name__ == "__main__":
    pub_topic = rospy.get_param("candidates_img_topic")
    visualize_only_best_cnd = rospy.get_param("visualize_only_best_cnd")

    VisualizeServer("visualize_server", pub_topic, visualize_only_best_cnd)

    rospy.spin()
