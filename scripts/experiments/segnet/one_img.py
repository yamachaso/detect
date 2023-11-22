# %%
import numpy as np

import torch

from modules.segnet.segnet import SegNetBasicVer2
from modules.const import WIGHTS_DIR, SEGNET_DATASETS_PATH
# %%
print(WIGHTS_DIR)
print(SEGNET_DATASETS_PATH)
# %%%
from modules.segnet.utils.dataloader import make_datapath_list_angle, one_img_getitem
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

rootpath = f"{SEGNET_DATASETS_PATH}/train"
test_img_list, test_color_anno_list, test_mask_anno_list= make_datapath_list_angle(rootpath=rootpath)

img_input = cv2.imread(f"{SEGNET_DATASETS_PATH}/train/original/color0.jpg")

color_mean = (0.50587555, 0.55623757, 0.4438057)
color_std = (0.19117119, 0.17481992, 0.18417963)
input_size = (160, 120)

img = one_img_getitem(img_input, input_size, color_mean, color_std)
print(img.shape)

# %%
state_dict = torch.load(f"{WIGHTS_DIR}/segnetbasic_10000.pth")
net = SegNetBasicVer2()
net = net.to(device)
net.load_state_dict(state_dict)
net.eval()

img = img.to(device)
outputs, outputs_color = net(img)
y = outputs  # AuxLoss側は無視
y_c = outputs_color

# print(y)
y = y[0].cpu().detach().numpy().transpose((1, 2, 0))
y_c = y_c[0].cpu().detach().numpy().transpose((1, 2, 0))



y = np.clip(y, 0, 1)
y_c = np.clip(y_c, 0, 1)

print(y.shape)
print(y_c.shape)
y = np.squeeze(y, -1)
plt.imshow(y)
plt.show()

mask = cv2.resize(cv2.imread(f"{SEGNET_DATASETS_PATH}/train/anno/mask/anno0.jpg"), (160, 120))[:, :, 0]
mask = mask.astype(np.float32) / 255
plt.imshow(mask)
plt.show()

kakeru = y * mask
print(kakeru.shape)
plt.imshow(kakeru)
plt.show()

sum_value = np.array([0.0, 0.0, 0.0])
cnt = np.array([0.0, 0.0, 0.0])
all_value = 0
for i in range(120):
    for j in range(160):
        sum_value += y_c[i][j] * kakeru[i][j]
        cnt += kakeru[i][j]

sum_value /= cnt
print(sum_value)

import colorsys

hi, si, vi = colorsys.rgb_to_hsv(sum_value[0], sum_value[1], sum_value[2])
#     print(hi, si, vi)
anglei = hi * 90
print(f'推論 : {anglei}')