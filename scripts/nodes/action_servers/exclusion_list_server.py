#!/usr/bin/env python3

import rospy
import numpy as np
from actionlib import SimpleActionServer
from std_msgs.msg import Header, String, Int32MultiArray
from detect.msg import (ExclusionListAction, 
                        ExclusionListGoal, 
                        ExclusionListResult)

from modules.ros.utils import multiarray2numpy, numpy2multiarray

class ExclusionListServer:
    def __init__(self, name: str):
        rospy.init_node(name, log_level=rospy.INFO)
        self.exclusion_list = [[], []]

        self.server = SimpleActionServer(name, ExclusionListAction, self.callback, False)
        self.server.start()

    def callback(self, goal: ExclusionListGoal):
        try:
            # 0: left, 1: right
            arm_index = goal.arm_index
            u = goal.u
            v = goal.v
            ref = goal.ref
            clear = goal.clear

            if clear:
                self.exclusion_list[arm_index] = []
            elif ref:
                pass
            else:
                self.exclusion_list[arm_index].append([u, v])
                # 5個より多くなったら古いものから削除
                if len(self.exclusion_list[arm_index]) > 3:
                    self.exclusion_list[arm_index].pop(0)

            result = ExclusionListResult(
                numpy2multiarray(Int32MultiArray, np.array(self.exclusion_list[arm_index]))
            )
        
            self.server.set_succeeded(result)

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    ExclusionListServer("exclusion_list_server")

    rospy.spin()