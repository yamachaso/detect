# %% 深度画像の欠損値の補完デモ
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from modules.const import SAMPLES_PATH


# %%
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


# %%
path = glob(f"{SAMPLES_PATH}/realsense_viewer_bags/*")[0]
handler = RealsenseBagHandler(path, 640, 480, 30)

img, depth = handler.get_images(use_filter=False)
missing_depth = np.where(depth == depth.min(), 255, 0)
print(img.dtype, depth.dtype)
fig, axes = plt.subplots(1, 3)
axes[0].imshow(img)
axes[1].imshow(depth, cmap="binary")
axes[2].imshow(missing_depth)

# %%
_, filtered_depth = handler.get_images(use_filter=True)
missing_filtered_depth = np.where(filtered_depth == filtered_depth.min(), 255, 0)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(filtered_depth, cmap="binary")
axes[1].imshow(missing_filtered_depth)

# %%
