#!/usr/bin/env python3
import message_filters as mf
import rospy
from cv_bridge import CvBridge
from modules.ros.action_clients import GraspDetectionClient
from sensor_msgs.msg import Image
from modules.ros.action_clients import TFClient
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header



if __name__ == "__main__":
    rospy.init_node("container_detection", log_level=rospy.INFO)

    tf_clinet = TFClient("base_link")


    header = Header()
    header.frame_id = "container_tr"
    res = tf_clinet.transform_point(header, Point())

    print(res)

    rospy.spin()
