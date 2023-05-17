#!/usr/bin/env python3
# import numpy as np
import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from detect.msg import (ComputeDepthThresholdAction,
                        ComputeDepthThresholdActionGoal,
                        ComputeDepthThresholdResult)
from modules.image import compute_optimal_depth_thresh


class ComputeDepthThresholdServer:
    def __init__(self, name: str):
        rospy.init_node(name, log_level=rospy.INFO)

        self.bridge = CvBridge()
        self.server = SimpleActionServer(name, ComputeDepthThresholdAction, self.callback, False)
        self.server.start()

    def callback(self, goal: ComputeDepthThresholdActionGoal):
        # img_msg = goal.rgb
        depth_msg = goal.depth
        whole_mask_msg = goal.whole_mask

        try:
            # img = self.bridge.imgmsg_to_cv2(img_msg)
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
            whole_mask = self.bridge.imgmsg_to_cv2(whole_mask_msg)
            n = goal.n

            optimal_th = compute_optimal_depth_thresh(depth, whole_mask, n)
            rospy.loginfo(f"depth_theshold: {optimal_th}")
            # mask_1c = np.where(depth < optimal_th, 1, 0)[:, :, np.newaxis]

            # filtered_img = (img * mask_1c).astype("uint8")
            # filtered_img_msg = self.bridge.cv2_to_imgmsg(filtered_img, "rgb8")
            # filtered_img_msg.header = img_msg.header

            # self.server.set_succeeded(ComputeDepthThresholdResult(filtered_img_msg))
            self.server.set_succeeded(ComputeDepthThresholdResult(optimal_th))

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    ComputeDepthThresholdServer("compute_depth_threshold_server")

    rospy.spin()
