#!/usr/bin/env python3
import rospy
from detect.srv import (ConvertMmToPixel, ConvertMmToPixelRequest,
                        ConvertMmToPixelResponse, ConvertPixelToMm,
                        ConvertPixelToMmRequest, ConvertPixelToMmResponse)
from sensor_msgs.msg import CameraInfo

SERVICE_NAMES_AND_TYPES = {"convert_mm_to_pixel": ConvertMmToPixel, "convert_pixel_to_mm": ConvertPixelToMm}


class UnitConvertServer:
    def __init__(self, name: str, info_topic: str):
        rospy.init_node(name)

        self.cam_info: CameraInfo = rospy.wait_for_message(info_topic, CameraInfo, timeout=None)
        self.px_to_mm_x, self.px_to_mm_y, self.mm_to_px_x, self.mm_to_px_y = self._get_convert_ratio()

        self.servers = [rospy.Service(name, type, getattr(self, name)) for name, type in SERVICE_NAMES_AND_TYPES.items()]

    def _get_convert_ratio(self):
        # 内部パラメータと外部パラメータの焦点距離からmmとピクセルの変換比を算出
        # K,Dはflattenされたカメラパラメータ
        flatten_inner_params = self.cam_info.K
        flatten_outer_params = self.cam_info.P
        fx_px, fy_px = flatten_inner_params[0], flatten_inner_params[4]
        fx_mm, fy_mm = flatten_outer_params[0], flatten_outer_params[5]

        px_to_mm_x = fx_mm / fx_px
        px_to_mm_y = fy_mm / fy_px
        mm_to_px_x = 1 / px_to_mm_x
        mm_to_px_y = 1 / px_to_mm_y

        return (px_to_mm_x, px_to_mm_y, mm_to_px_x, mm_to_px_y)

    def convert_mm_to_pixel(self, req: ConvertMmToPixelRequest):
        x_mm, y_mm = req.input
        return ConvertMmToPixelResponse((x_mm * self.mm_to_px_x, y_mm + self.mm_to_px_y))

    def convert_pixel_to_mm(self, req: ConvertPixelToMmRequest):
        x_px, y_px = req.input
        return ConvertPixelToMmResponse((x_px * self.px_to_mm_x, y_px * self.px_to_mm_y))


if __name__ == "__main__":
    info_topic = rospy.get_param("image_info_topic")

    UnitConvertServer("unit_convert_server", info_topic)

    rospy.spin()
