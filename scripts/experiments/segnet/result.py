# %%
import random
import math
import time
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from modules.segnet.segnet import SegNetBasicVer2
from modules.const import OUTPUTS_PATH, SEGNET_DATASETS_PATH
# %%
WIGHTS_DIR = f"{OUTPUTS_PATH}/segnet_weights"
print(WIGHTS_DIR)
print(SEGNET_DATASETS_PATH)

# %%
# 初期設定
# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
# %%
import cv2
import matplotlib.pyplot as plt

def binarize(src_img):
    bin_img = cv2.threshold(src_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#     bin_img = cv2.threshold(src_img,220,255,cv2.THRESH_BINARY)[1]
    return bin_img

def max_area(stats):
    maxv = -1
    maxi = -1
    for i in range(1, len(stats)):
        if maxv < stats[i][4]:
            maxv = stats[i][4]
            maxi = i
    return maxv, maxi
     
# 各ラベルの座標と面積を描画する
def draw_result(src_img, stats, centroids):
    maxv, maxi = max_area(stats)
    result_img = src_img.copy()
    coordinate = stats[maxi]
    left_top = (coordinate[0], coordinate[1])
    right_bottom = (coordinate[0] + coordinate[2], coordinate[1] + coordinate[3])
    result_img = cv2.rectangle(result_img, left_top, right_bottom, (0, 0, 255), 1)
    result_img = cv2.putText(result_img, str(coordinate[4]), left_top, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    
    coordinate = centroids[maxi]
    center = (int(coordinate[0]), int(coordinate[1]))
    print(f'重心 : {center}')
    result_img = cv2.circle(result_img, center, 1, (255, 0, 0))
    
    return result_img

def draw_bin_img_result(src_img, stats, labels):
    maxv, maxi = max_area(stats)
    result_img = src_img.copy()
    for i in range(120):
        for j in range(160):
            if labels[i][j] != maxi:
                result_img[i][j] = 0.0
    
    return result_img
# %%
from modules.segnet.utils.dataloader import make_datapath_list_angle, DataTransform_angle, MyDataset_angle
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

rootpath = f"{SEGNET_DATASETS_PATH}/train"
test_img_list, test_color_anno_list, test_mask_anno_list= make_datapath_list_angle(rootpath=rootpath)

color_mean = (0.50587555, 0.55623757, 0.4438057)
color_std = (0.19117119, 0.17481992, 0.18417963)

val_dataset = MyDataset_angle(test_img_list, test_color_anno_list, test_mask_anno_list, phase="val", transform=DataTransform_angle(
    input_size=(160, 120), color_mean=color_mean, color_std=color_std))


# img_index = 200

def ok_ng(img_index = 200):
    # 3. PSPNetで推論する
    state_dict = torch.load(f"{WIGHTS_DIR}/segnetbasic_10000.pth")
    net = SegNetBasicVer2()
    net = net.to(device)
    net.load_state_dict(state_dict)
    net.eval()
    img, anno_color_class_img, anno_mask_class_img = val_dataset.__getitem__(img_index)
    x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 475, 475])

    x = x.to(device)
    outputs, outputs_color = net(x)
    y = outputs  # AuxLoss側は無視
    y_c = outputs_color

    # print(y)
    y = y[0].cpu().detach().numpy().transpose((1, 2, 0))
    y_c = y_c[0].cpu().detach().numpy().transpose((1, 2, 0))



    y = np.clip(y, 0, 1)
    y_c = np.clip(y_c, 0, 1)


#     plt.imshow(y, cmap = "gray")
#     plt.show()
    bin_img = binarize((y*255).astype(np.uint8))[:, :, np.newaxis]
#     plt.imshow(bin_img, cmap = "gray")
#     plt.show()

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
    src_img = cv2.resize(cv2.imread(test_img_list[img_index]), dsize=(160, 120))
    result_img = draw_result(src_img, stats, centroids)

#     plt.imshow(result_img)
#     plt.show()

    bin_img_result = draw_bin_img_result(bin_img, stats, labels)
#     plt.imshow(bin_img_result, cmap = "gray")
#     plt.show()

    sum_value = np.array([0.0, 0.0, 0.0])
    cnt = np.array([0.0, 0.0, 0.0])
    all_value = 0
    for i in range(120):
        for j in range(160):
            sum_value += y_c[i][j] * bin_img_result[i][j]
            cnt += bin_img_result[i][j]

    sum_value /= cnt
    print(sum_value)

    import colorsys
    anno_color = anno_color_class_img.cpu().detach().numpy().transpose((1, 2, 0))
    r, g, b = anno_color[0][0]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
#     print(h, s, v)
    angle = h * 90
    print(f'教師 : {angle}')
    


    hi, si, vi = colorsys.rgb_to_hsv(sum_value[0], sum_value[1], sum_value[2])
#     print(hi, si, vi)
    anglei = hi * 90
    print(f'推論 : {anglei}')
    
    if min(abs(angle - anglei), 90 - abs(angle - anglei)) <= 25:
        return True
    else:
        return False
    

number = len(test_img_list)
cnt = 0
for i in range(number):
    print(i)
    if ok_ng(i):
        cnt += 1
cnt