#!/usr/bin/env python3
import struct
from typing import Tuple
import rospy
from sensor_msgs.msg import PointCloud2

def get_xyz_from_pc2(msg: PointCloud2, uv: Tuple[int, int]) -> Tuple[float, float, float]:
    fields = msg.fields
    point_step = msg.point_step
    row_step = msg.row_step
    data = msg.data

    offset = (uv[1] * row_step) + (uv[0] * point_step)
    target_data = data[offset:offset+point_step]
    xyz = tuple([struct.unpack("<f", target_data[fields[i].offset:fields[i].offset+4])[0] for i in range(3)])

    return xyz

if __name__ == "__main__":
    rospy.init_node("unpack_pc2_node")

    topic = "/myrobot/left_camera/depth/color/points"
    pc2_msg: PointCloud2 = rospy.wait_for_message(topic, PointCloud2)

    width = pc2_msg.width
    height = pc2_msg.height
    fields = pc2_msg.fields
    point_step = pc2_msg.point_step
    row_step = pc2_msg.row_step
    data = pc2_msg.data

    # print(width, height, width * height)
    # print(fields)
    # print(point_step)
    # print(row_step)
    # print(len(data), len(data) / point_step, len(data) / row_step)
    # print(data[:31])

    target_uv = (100, 200)
    xyz = get_xyz_from_pc2(pc2_msg, target_uv)
    print(xyz)
