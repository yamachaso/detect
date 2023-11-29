#!/usr/bin/env python3
import rospy
from actionlib import SimpleActionServer
from detect.msg import (ContainerPositionAction, ContainerPositionGoal,
                        ContainerPositionResult)
import tf
from geometry_msgs.msg import Point

class ContainerPositionServer:
    def __init__(self, name: str):
        rospy.init_node(name, log_level=rospy.INFO)    
    
        self.server = SimpleActionServer(name, ContainerPositionAction, self.callback)

        self.trans = None

        rate = rospy.Rate(10.0)
        listener = tf.TransformListener()
        while not rospy.is_shutdown():
            try:
                (t, r) = listener.lookupTransform('/base_link', '/container_base', rospy.Time())
                self.trans = t
    
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            
            rate.sleep()

    
    def callback(self, goal: ContainerPositionGoal):
        try:
            if self.trans is None:
                raise ValueError("error!")
            
            point = Point(x=self.trans[0], y=self.trans[1], z=self.trans[2])
            result = ContainerPositionResult(point)
            self.server.set_succeeded(result)

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    ContainerPositionServer("container_position_server") 
    rospy.spin()
