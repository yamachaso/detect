import pickle

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs


def load_py2_pickle(path):
    with open(path, mode='rb') as f:
        # python2でつくったpickleはpython3ではエンコーディングエラーがでる
        # ref: https://qiita.com/f0o0o/items/4cdad7f3748741a3cf74
        # 自作msgが入ってくるとエラー出る
        data = pickle.load(f, encoding='latin1')

    return data


def imshow(img, show_axis=False, cmap=None):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    if show_axis is False:
        plt.axis("off")
        # Not work: 一部余白が残る
        # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)


class RealsenseBagHandler:
    def __init__(self, path: str, w: int, h: int, fps: int, align_to: rs.stream = rs.stream.color):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        rs.config.enable_device_from_file(self.config, path)

        # RGBの最大解像度はもっと高いが指定できない、録画時の設定の問題？
        self.config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)

        self.align = rs.align(align_to)
        # self.colorizer = rs.colorizer()

        pipeline_profile = self.pipeline.start(self.config)
        stream_profile = pipeline_profile.get_stream(align_to)
        video_stream_profile = stream_profile.as_video_stream_profile()
        intrinsics = video_stream_profile.get_intrinsics()
        self.fp = (intrinsics.fx + intrinsics.fy) / 2

        # decimate = rs.decimation_filter(magnitude=2)
        spatial = rs.spatial_filter(smooth_alpha=0.5, smooth_delta=20, magnitude=2, hole_fill=0)
        temporal = rs.temporal_filter(smooth_alpha=0.4, smooth_delta=20, persistence_control=3)
        hole_filling = rs.hole_filling_filter(mode=2)
        self.filters = [
            # decimate,
            spatial,
            temporal,
            hole_filling
        ]

    def get_images(self, use_filter=True):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        # Get depth frame
        rgb_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if use_filter:
            for f in self.filters:
                depth_frame = f.process(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        rgb = np.asanyarray(rgb_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        return rgb, depth
