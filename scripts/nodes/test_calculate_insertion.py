#!/usr/bin/env python3

# $ roslaunch myrobot_moveit multiple_rs_camera.launch camera_serial_no_1:="044322071294"
# $ rosbag play -r 1.0 20230117_134148.bag

import message_filters as mf
import rospy
from cv_bridge import CvBridge
from modules.ros.action_clients import GraspDetectionClient
from sensor_msgs.msg import Image
from modules.type import Mm, Px
from sensor_msgs.msg import CameraInfo
import numpy as np
from modules.ros.publishers import ImageMatPublisher
import cv2


class CalculateInsertionTestClient:
    def __init__(self, name: str, fps: float, image_topic: str, depth_topic: str):
        rospy.init_node(name, log_level=rospy.INFO)

        self.count = 0

        # WARN: rossag playだとfps > 1に対応できない
        delay = 1 / fps

        # Subscribers
        subscribers = [mf.Subscriber(topic, Image) for topic in (image_topic, depth_topic)]
        # Action Clients
        # Others
        self.bridge = CvBridge()

        self.ts = mf.ApproximateTimeSynchronizer(subscribers, 10, delay)
        self.ts.registerCallback(self.callback)

        cam_info: CameraInfo = rospy.wait_for_message("/myrobot/left_camera/color/camera_info", CameraInfo, timeout=None)
        # frame_size = (cam_info.height, cam_info.width)

        # convert unit
        # ref: https://qiita.com/srs/items/e3412e5b25477b46f3bd
        flatten_corrected_params = cam_info.P
        fp_x, fp_y = flatten_corrected_params[0], flatten_corrected_params[5]
        self.fp = (fp_x + fp_y) / 2

        self.hand_radius_mm = 152.5

        self.result_publisher = ImageMatPublisher("/result", queue_size=10)

        

        rospy.logerr("finished constructor")

    def _convert_mm_to_px(self, v_mm: Mm, d: Mm) -> Px:

        print("self.fp", self.fp)
        v_px = (v_mm / d) * self.fp  # (v_mm / 1000) * self.fp / (d / 1000)
        return v_px
    
    def _convert_px_to_mm(self, v_px: Px, d: Mm) -> Mm:
        v_mm = (v_px / self.fp) * d
        return v_mm

    def is_in_image(self, h, w, i, j) -> bool:
        if i < 0 or i >= h or j < 0 or j >= w:
            return False
        return True

    def drawResult(self, img, depth, x, y, t, r):
        height, width = depth.shape[0], depth.shape[1]
        center_h, center_w = height // 2, width // 2
        center_d = depth[center_h, center_w]

        print("r mm" , r)
        r = self._convert_mm_to_px(r, center_d)

        print("r px", r)

        point1x, point1y = int(x - r * np.sin(np.deg2rad(t))), int(y + r * np.cos(np.deg2rad(t)))
        point2x, point2y = int(x - r * np.sin(np.deg2rad(t+120))), int(y + r * np.cos(np.deg2rad(t+120)))
        point3x, point3y = int(x - r * np.sin(np.deg2rad(t+240))), int(y + r * np.cos(np.deg2rad(t+240)))

        print("####", point1x, point1y, point2x, point2y, point3x, point3y)


        # cv2.circle(img, (point1x, point1y), 15, (255, 0, 255), thickness=-1)
        # cv2.circle(img, (point2x, point2y), 15, (255, 0, 255), thickness=-1)
        # cv2.circle(img, (point3x, point3y), 15, (255, 0, 255), thickness=-1)

        cv2.circle(img, (point1y, point1x), 15, (255, 0, 255), thickness=-1)
        cv2.circle(img, (point2y, point2x), 15, (255, 0, 255), thickness=-1)
        cv2.circle(img, (point3y, point3x), 15, (255, 0, 255), thickness=-1)

        hand_radius_px = self._convert_mm_to_px(self.hand_radius_mm, center_d)

        print("efiefjiefji", width, height)

        for xi in range(-200, 200, 20):
            for yi in range(-200, 200, 20):
                # for ri in range(0, 50, 5):
                #     rr = hand_radius_px - ri
                cv2.circle(img, (center_w + yi, center_h + xi), 5, (0, 0, 255), thickness=-1)

        for ri in range(0, 200, 20):
            rr = int(hand_radius_px - ri)
            cv2.circle(img, (y, x), rr, (0, 255, 0), thickness=2)

        return img




    def calcurate_insertion(self, depth):
        # 単位変換
        height, width = depth.shape[0], depth.shape[1]

        center_h, center_w = height // 2, width // 2
        center_d = depth[center_h, center_w]

        # center_dが欠損すると0 divisionになるので注意
        hand_radius_px = self._convert_mm_to_px(self.hand_radius_mm, center_d)
        # finger_radius_px = self._convert_mm_to_px(self.finger_radius_mm, center_d)
        # ベクトルははじめの角度求めるとかで関数内部で計算してもいいかも

        max_score = 0
        best_x = -1
        best_y = -1
        best_t = -1
        best_r = -1

        best_xi = -1
        best_yi = -1
        best_ti = -1
        best_ri = -1

        for ti in range(0, 120, 5):
            for xi in range(-200, 200, 20):
                for yi in range(-200, 200, 20):
                    for ri in range(0, 200, 20):
                        rr = hand_radius_px - ri
                        point1x, point1y = center_h + xi - rr * np.sin(np.deg2rad(ti)), center_w + yi + rr * np.cos(np.deg2rad(ti))
                        point2x, point2y = center_h + xi - rr * np.sin(np.deg2rad(ti+120)), center_w + yi + rr * np.cos(np.deg2rad(ti+120))
                        point3x, point3y = center_h + xi - rr * np.sin(np.deg2rad(ti+240)), center_w + yi + rr * np.cos(np.deg2rad(ti+240))

                        if not self.is_in_image(height, width, point1x, point1y):
                            continue
                        if not self.is_in_image(height, width, point2x, point2y):
                            continue
                        if not self.is_in_image(height, width, point3x, point3y):
                            continue

                        # tmp_score = depth[int(point1x)][int(point1y)] + depth[int(point2x)][int(point2y)] + depth[int(point3x)][int(point3y)]
                        tmp_score = min(min(depth[int(point1x)][int(point1y)], depth[int(point2x)][int(point2y)]), depth[int(point3x)][int(point3y)])
                        if tmp_score > max_score:
                            best_x = xi + center_h
                            best_y = yi + center_w
                            best_t = ti
                            best_r = rr

                            best_xi = xi
                            best_yi = yi
                            best_ti = ti
                            best_ri = ri

        best_r = self._convert_px_to_mm(best_r, center_d) # mm単位に

        print("注目", best_xi, best_yi, best_ti, best_ri)
        print(best_x, best_y, best_t, best_r)

        return best_x, best_y, best_t, best_r

    def callback(self, img_msg: Image, depth_msg: Image):
        # img_time = img_msg.header.stamp.to_time()
        # depth_time = depth_msg.header.stamp.to_time()
        self.count += 1
        if self.count % 10 != 0:
            return
        
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg)
            depth = self.bridge.imgmsg_to_cv2(depth_msg)

            
            print(img.shape)
            print(depth.shape)
            print(depth[0][0])
            x, y, t, r = self.calcurate_insertion(depth)

            result = self.drawResult(img, depth, x, y, t, r)

            # print(depth.max())

            # depth = (depth / depth.max() * 255).astype(np.uint8)
            # depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
            

            frame_id = img_msg.header.frame_id
            stamp = img_msg.header.stamp

            self.result_publisher.publish(result, frame_id, stamp)

        except Exception as err:
            rospy.logerr(err)


if __name__ == "__main__":
    # fps = rospy.get_param("fps")
    fps = 1
    # image_topic = rospy.get_param("image_topic")
    image_topic = "/myrobot/left_camera/color/image_raw"
    # depth_topic = rospy.get_param("depth_topic")
    depth_topic = "/myrobot/left_camera/aligned_depth_to_color/image_raw"

    CalculateInsertionTestClient(
        "calculate_insertion_test_client",
        fps=fps,
        image_topic=image_topic,
        depth_topic=depth_topic,
    )

    rospy.spin()
