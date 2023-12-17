#!/usr/bin/env python3
import message_filters as mf
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from actionlib import SimpleActionClient
from detect.msg import GraspDetectionAction, GraspDetectionGoal, CalcurateInsertionAction, CalcurateInsertionGoal
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.ros.publishers import ImageMatPublisher2
from modules.const import HAND_MASK_PATH


# ref: https://qiita.com/ynott/items/8acd03434569e23612f1
# class GraspDetectionClient(SimpleActionClient, object):
class DepthViwer:
    def __init__(self, fps, image_topic, depth_topic, points_topic, wait=True):
        # super(GraspDetectionClient, self).__init__(ns, ActionSpec)
        delay = 1 / fps * 0.5
        # Topics
        self.points_topic = points_topic
        # Subscribers
        subscribers = [mf.Subscriber(topic, Image) for topic in (image_topic, depth_topic)]
        subscribers.append(mf.Subscriber(points_topic, PointCloud2))
        # Others
        self.bridge = CvBridge()
        self.request = None

        self.ts = mf.ApproximateTimeSynchronizer(subscribers, 10, delay)
        # self.ts = mf.ApproximateTimeSynchronizer(subscribers, 1, delay)
        self.ts.registerCallback(self.callback)

        self.publisher = ImageMatPublisher2("/depth_hist_viewer", queue_size=10)

        self.depth_img = None
        self.depth_mask = cv2.imread(f"{HAND_MASK_PATH}/right_image_clear.png", cv2.IMREAD_GRAYSCALE)
        self.depth_mask = self.depth_mask.astype(bool)

    # def callback(self, img_msg, depth_msg):
    def callback(self, img_msg, depth_msg, points_msg):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(depth_msg)
            self.color_img = self.bridge.imgmsg_to_cv2(img_msg)
        except Exception as err:
            rospy.logerr(err)


    def depth_hist_view(self):
        if self.depth_img is None:
            return
        
        # bw_img = np.zeros_like(self.depth_img)
        bw_img = np.full_like(self.depth_img, 255)
        bw_img[self.depth_img > 2000] = 0
        bw_img[self.depth_img < 100] = 0


        valid_depth = self.depth_img[self.depth_mask]

        plt.imshow(bw_img, cmap="gray")
        # plt.hist(valid_depth)

        buf = io.BytesIO() # bufferを用意
        plt.savefig(buf, format='png') # bufferに保持
        enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # bufferからの読み出し
        dst = cv2.imdecode(enc, 1) # デコード
        dst = dst[:,:,::-1] # BGR->RGB
        self.publisher.publish(dst)
        plt.clf()

        good_count = len(valid_depth)
        good_count -= np.count_nonzero(valid_depth < 100)
        good_count -= np.count_nonzero(valid_depth > 1800)

        print("invalid percentage : ", good_count / len(valid_depth))


if __name__ == "__main__":
    rospy.init_node("grasp_detection_client", log_level=rospy.INFO)

    fps = rospy.get_param("fps")
    left_image_topic = rospy.get_param("left_image_topic")
    left_depth_topic = rospy.get_param("left_depth_topic")
    left_points_topic = rospy.get_param("left_points_topic")
    right_image_topic = rospy.get_param("right_image_topic")
    right_depth_topic = rospy.get_param("right_depth_topic")
    right_points_topic = rospy.get_param("right_points_topic")

    cli = DepthViwer(
        fps=fps,
        image_topic=right_image_topic,
        depth_topic=right_depth_topic,
        points_topic=right_points_topic,
    )

    rate = rospy.Rate(1)

    while not rospy.is_shutdown(): 
        cli.depth_hist_view()
        rate.sleep()