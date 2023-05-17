#! /usr/bin/env python3
# coding: UTF-8
import actionlib
import rospy
from detect.msg import TransformPointAction, TransformPointResult
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener


class TFTransformServer:
    def __init__(self, name: str):
        rospy.init_node(name)

        self.buffer = Buffer()
        self.lisner = TransformListener(self.buffer)
        self.server = actionlib.SimpleActionServer(name, TransformPointAction, self.callback, False)
        self.server.start()

    def callback(self, goal):
        frame_id = goal.source.header.frame_id
        stamp = goal.source.header.stamp
        point = goal.source.point
        trans = self.get_trans(goal.target_frame, frame_id, stamp)
        result = TransformPointResult(self.transform_point(frame_id, stamp, point, trans))
        self.server.set_succeeded(result)

    def get_trans(self, target_frame: str, frame_id: str, stamp: rospy.Time):
        trans = self.buffer.lookup_transform(target_frame, frame_id, stamp)

        return trans

    def transform_point(self, frame_id: str, stamp: rospy.Time, point: Point, trans) -> PointStamped:
        point_stamped = PointStamped(
            header=Header(frame_id=frame_id, stamp=stamp),
            point=point
        )
        tf_point = do_transform_point(point_stamped, trans)
        return tf_point


if __name__ == "__main__":
    TFTransformServer("tf_transform_server")

    rospy.spin()
