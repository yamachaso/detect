#!/usr/bin/env python3

# $ roslaunch myrobot_moveit multiple_rs_camera.launch camera_serial_no_1:="044322071294"
# $ roslaunch myrobot_moveit multiple_rs_camera.launch camera_serial_no_1:="915112070340"

# $ rosbag play -r 1.0 20230117_134148.bag

import message_filters as mf
import rospy
from cv_bridge import CvBridge
from modules.ros.action_clients import GraspDetectionClient
from sensor_msgs.msg import Image
from modules.type import Mm, Px
from sensor_msgs.msg import CameraInfo
import numpy as np
from modules.ros.publishers import ImageMatPublisher
import cv2
from detectron2.config import CfgNode, get_cfg
from modules.const import CONFIGS_PATH, DATASETS_PATH, OUTPUTS_PATH
from entities.predictor import Predictor
import matplotlib.pyplot as plt


class CalculateInsertionTestClient:
    def __init__(self, name: str, fps: float, image_topic: str, depth_topic: str):
        rospy.init_node(name, log_level=rospy.INFO)


        self.seg_publisher = ImageMatPublisher("/seg_result", queue_size=10)

        config_path = f"{CONFIGS_PATH}/config.yaml"
        weight_path = f"{OUTPUTS_PATH}/mask_rcnn/model_final.pth"
        device = "cuda:0"
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.DEVICE = device
        self.predictor = Predictor(cfg)


        # WARN: rossag playだとfps > 1に対応できない
        delay = 1 / fps

        # Subscribers
        subscribers = [mf.Subscriber(topic, Image) for topic in (image_topic, depth_topic)]
        # Action Clients
        # Others
        self.bridge = CvBridge()

        self.ts = mf.ApproximateTimeSynchronizer(subscribers, 10, delay)
        self.ts.registerCallback(self.callback)

        cam_info: CameraInfo = rospy.wait_for_message("/myrobot/left_camera/color/camera_info", CameraInfo, timeout=None)
        # frame_size = (cam_info.height, cam_info.width)

        # convert unit
        # ref: https://qiita.com/srs/items/e3412e5b25477b46f3bd
        flatten_corrected_params = cam_info.P
        fp_x, fp_y = flatten_corrected_params[0], flatten_corrected_params[5]
        self.fp = (fp_x + fp_y) / 2

        self.hand_radius_mm = 152.5




        rospy.logerr("finished constructor")



    def callback(self, img_msg: Image, depth_msg: Image):
        print("wwwwwww")
        # img_time = img_msg.header.stamp.to_time()
        # depth_time = depth_msg.header.stamp.to_time()
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg)
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
            res = self.predictor.predict(img)

            frame_id = img_msg.header.frame_id
            stamp = img_msg.header.stamp

            seg = res.draw_instances(img[:, :, ::-1])

            depth_list = [depth[center[1]][center[0]] for center in res.centers]
            max_depth = max(depth_list)
            min_depth = min(depth_list)
            print(max_depth, min_depth)
            depth_score = 1 - (depth_list - min_depth) / (max_depth - min_depth)
            print(depth_score)

            ellipse_list = [cv2.fitEllipse(contour) for contour in res.contours]


            ellipse_masks = [cv2.ellipse(np.zeros_like(depth),ellipse, 1, -1).astype(np.bool) for ellipse in ellipse_list]
            masks = [mask.astype(np.bool) for mask in res.masks]

            ellipse_iou = []
            for mask, emask in zip(masks, ellipse_masks):
                ellipse_iou.append(np.sum(mask * emask) / np.sum(mask | emask)) # IoU

            max_iou = max(ellipse_iou)
            min_iou = min(ellipse_iou)
            ellipse_score = (ellipse_iou - min_iou) / (max_iou - min_iou)
                
            # print(ellipse_score)

            final_score = depth_score * 0.5 + ellipse_score * 0.5

            seg2 = cv2.resize(seg, (640, 480)) # seg は (320, 240)
            for i, score in  enumerate(final_score):
                print(score)
                cv2.putText(seg2, f"{score:.2f}", (res.centers[i][0] + 5, res.centers[i][1] + 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
            index = np.argmax(final_score)
            cv2.ellipse(seg2, ellipse_list[index], (0, 255, 255), 3)

            self.seg_publisher.publish(seg2, frame_id, stamp)


        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    # fps = rospy.get_param("fps")
    fps = 1
    # image_topic = rospy.get_param("image_topic")
    image_topic = "/myrobot/left_camera/color/image_raw"
    # depth_topic = rospy.get_param("depth_topic")
    depth_topic = "/myrobot/left_camera/aligned_depth_to_color/image_raw"

    CalculateInsertionTestClient(
        "calculate_insertion_test_client",
        fps=fps,
        image_topic=image_topic,
        depth_topic=depth_topic,
    )

    rospy.spin()
