#!/usr/bin/env python3

import rospy
from actionlib import SimpleActionServer
from std_msgs.msg import Header, String
from ExclusionList.msg import (ExclusionListAction, 
                                ExclusionListGoal, 
                                ExclusionListResult)

class ExclusionListServer:
    def __init__(self, name: str):
        rospy.init_node(name, log_level=rospy.INFO)
        self.exclusion_list = []

        self.server = SimpleActionServer(name, ExclusionListAction, self.callback, False)
        self.server.start()

    def callback(self, goal: ExclusionListGoal):
        try:
            new_point = goal.new_point
            ref = goal.ref
            clear = goal.clear

            if clear is True:
                self.exclusion_list = []
            elif ref is True:
                pass
            else:
                self.exclusion_list.append(new_point)

            result = ExclusionListResult(self.exclusion_list)
            self.server.set_succeeded(result)

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    ExclusionListServer("exclusion_list_server")

    rospy.spin()