#!/usr/bin/env python3


import rospy
import math
import tf
import geometry_msgs


if __name__ == "__main__":
    rospy.init_node("container_tf_adjuster")

    listener = tf.TransformListener()

    rate = rospy.Rate(10.0)

    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform('/base_link', '/ar_marker_0', rospy.Time(0))

            br = tf.TransformBroadcaster()
            br.sendTransform((trans[0], trans[1], trans[2]),
                            # tf.transformations.quaternion_from_euler(math.pi / 2, 0, -math.pi / 2), # z ,y, xの順で回転。引数は 3 ,2 ,1の順番
                            tf.transformations.quaternion_from_euler(0, 0, 0), # z ,y, xの順で回転。引数は 3 ,2 ,1の順番
                            rospy.Time.now(),
                            '/container_origin',
                            "/base_link")


        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        # ref: https://x.gd/1d4F7
        rate.sleep()