# %%
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from modules.const import SAMPLES_PATH

# %%
path = glob(f"{SAMPLES_PATH}/realsense_viewer_bags/*")[0]
print(path)
# %%
pipeline = rs.pipeline()
config = rs.config()

rs.config.enable_device_from_file(config, path)
# RGBの最大解像度はもっと高いが指定できない、録画時の設定の問題？
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

align_to = rs.stream.color
align = rs.align(align_to)

colorizer = rs.colorizer()

# pipeline.stop()  # 一回stopしないとstartできない?
pipeline_profile = pipeline.start(config)

print(pipeline_profile.get_device())
# おそらくstreamにはmotionとvideoが存在する
rgb_stream_profile = pipeline_profile.get_stream(rs.stream.color)
rgb_video_stream_profile = rgb_stream_profile.as_video_stream_profile()
print(rgb_stream_profile)
print(rgb_video_stream_profile)
intrinsics = rgb_video_stream_profile.get_intrinsics()
print(intrinsics)
fx, fy = intrinsics.fx, intrinsics.fy
print(fx, fy)
fp = (fx + fy) / 2
# %%

# Get frameset of depth
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)

# Get depth frame
rgb_frame = aligned_frames.get_color_frame()
depth_frame = aligned_frames.get_depth_frame()

# Colorize depth frame to jet colormap
depth_color_frame = colorizer.colorize(depth_frame)

# Convert depth_frame to numpy array to render image in opencv
rgb_image = np.asanyarray(rgb_frame.get_data())
depth_color_image = np.asanyarray(depth_color_frame.get_data())

# Render image in opencv window
print(rgb_image.shape, depth_color_image.shape)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(rgb_image)
axes[1].imshow(depth_color_image)
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

    def get_images(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        # Get depth frame
        rgb_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # Convert depth_frame to numpy array to render image in opencv
        rgb = np.asanyarray(rgb_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        return rgb, depth


# %%
handler = RealsenseBagHandler(path, 640, 480, 30, rs.stream.color)
img, depth = handler.get_images()
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[1].imshow(depth)
# %%
