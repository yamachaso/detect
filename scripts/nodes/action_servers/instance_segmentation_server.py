#!/usr/bin/env python3
import rospy
from actionlib import SimpleActionServer
from cv_bridge import CvBridge
from std_msgs.msg import Header
from detect.msg import (InstanceSegmentationAction, InstanceSegmentationGoal,
                        InstanceSegmentationResult)
from detectron2.config import CfgNode, get_cfg
from entities.predictor import Predictor
from modules.const import CONFIGS_PATH, OUTPUTS_PATH
from modules.ros.msg_handlers import InstanceHandler
from modules.ros.publishers import ImageMatPublisher
import os
from datetime import datetime
import cv2
from modules.const import CONFIGS_PATH, DATASETS_PATH, OUTPUTS_PATH


class InstanceSegmentationServer:
    def __init__(self, name: str, cfg: CfgNode, seg_topic: str):
        rospy.init_node(name, log_level=rospy.INFO)

        self.bridge = CvBridge()
        self.predictor = Predictor(cfg)
        self.seg_publisher = ImageMatPublisher(seg_topic, queue_size=10)

        self.server = SimpleActionServer(name, InstanceSegmentationAction, self.callback, False)
        self.server.start()

        self.count = 0
        self.now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    def callback(self, goal: InstanceSegmentationGoal):
        try:
            img_msg = goal.image
            img = self.bridge.imgmsg_to_cv2(img_msg)
            res = self.predictor.predict(img)
            frame_id = img_msg.header.frame_id
            stamp = img_msg.header.stamp

            seg = res.draw_instances(img[:, :, ::-1])
            self.seg_publisher.publish(seg, frame_id, stamp)

            # OUTPUT_DIR = f"{OUTPUTS_PATH}/tmp/{self.now}"
            # os.makedirs(OUTPUT_DIR, exist_ok=True)
            # os.makedirs(f"{OUTPUT_DIR}/seg", exist_ok=True)
            # cv2.imwrite(f'{OUTPUT_DIR}/seg/{self.count}.jpg', seg)
            # self.count += 1
            

            instances = [InstanceHandler.from_predict_result(res, i) for i in range(res.num_instances)]
            result = InstanceSegmentationResult(Header(frame_id=frame_id, stamp=stamp), instances)
            self.server.set_succeeded(result)

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    seg_topic = rospy.get_param("seg_topic")

    config_path = rospy.get_param("config", f"{CONFIGS_PATH}/config.yaml")
    weight_path = rospy.get_param("weight", f"{OUTPUTS_PATH}/2023_12_15_06_01/model_0004999.pth")
    device = rospy.get_param("device", "cuda:0")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.DEVICE = device

    InstanceSegmentationServer("instance_segmentation_server", cfg, seg_topic)

    rospy.spin()
