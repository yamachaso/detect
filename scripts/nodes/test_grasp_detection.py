#!/usr/bin/env python3
import message_filters as mf
import rospy
from cv_bridge import CvBridge
from modules.ros.action_clients import GraspDetectionClient
from sensor_msgs.msg import Image


class GraspDetectionTestClient:
    def __init__(self, name: str, fps: float, image_topic: str, depth_topic: str):
        rospy.init_node(name, log_level=rospy.INFO)

        # WARN: rossag playだとfps > 1に対応できない
        delay = 1 / fps * 0.5

        # Subscribers
        subscribers = [mf.Subscriber(topic, Image) for topic in (image_topic, depth_topic)]
        # Action Clients
        self.gd_client = GraspDetectionClient()
        # Others
        self.bridge = CvBridge()

        self.ts = mf.ApproximateTimeSynchronizer(subscribers, 10, delay)
        self.ts.registerCallback(self.callback)

    def callback(self, img_msg: Image, depth_msg: Image):
        # img_time = img_msg.header.stamp.to_time()
        # depth_time = depth_msg.header.stamp.to_time()
        try:
            _ = self.gd_client.detect(img_msg, depth_msg)
        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    fps = rospy.get_param("fps")
    image_topic = rospy.get_param("image_topic")
    depth_topic = rospy.get_param("depth_topic")

    GraspDetectionTestClient(
        "grasp_detection_test_client",
        fps=fps,
        image_topic=image_topic,
        depth_topic=depth_topic,
    )

    rospy.spin()
