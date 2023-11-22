# %%
# パッケージのimport
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
from modules.segnet.utils.dataloader import make_datapath_list_angle, DataTransform_angle, MyDataset_angle

# ファイルパスリスト作成
rootpath = f"{SEGNET_DATASETS_PATH}/train"
train_img_list, train_anno_color_list, train_anno_mask_list = make_datapath_list_angle(rootpath=rootpath)

color_mean = (0.50587555, 0.55623757, 0.4438057)
color_std = (0.19117119, 0.17481992, 0.18417963)

train_dataset = MyDataset_angle(train_img_list, train_anno_color_list, train_anno_mask_list, phase="train", transform=DataTransform_angle(
    input_size=(160, 120), color_mean=color_mean, color_std=color_std))

batch_size = 32
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
# %%
import matplotlib.pyplot as plt
import cv2

# 実行するたびに変わります

# 画像データの読み込み
index = 2
imges, anno_color_class_imges, anno_mask_class_imges = train_dataset.__getitem__(index)

print(imges.shape)
print(anno_color_class_imges.shape)
print(anno_mask_class_imges.shape)

# 画像の表示
img_val = imges
img_val = img_val.numpy().transpose((1, 2, 0))
plt.imshow(img_val)
plt.show()

# アノテーション画像の表示
anno_color_img_val = anno_color_class_imges
anno_color_img_val = anno_color_img_val.numpy().transpose((1, 2, 0))
plt.imshow(anno_color_img_val)
plt.show()

anno_mask_img_val = anno_mask_class_imges
anno_mask_img_val = anno_mask_img_val.numpy().transpose((1, 2, 0))
plt.imshow(np.squeeze(anno_mask_img_val, -1))
plt.show()
# %%
# xavier : sigmoid
def xavier_init(m):
    if isinstance(m, nn.Conv2d):
        print(m)
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:  # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)

# He ReLU
def kaiming_init(m):
    if isinstance(m, nn.Conv2d):
        print(m)
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:  # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)

net = SegNetBasicVer2()
net.apply(kaiming_init)
net.conv11d.apply(xavier_init)
net.conv11dc.apply(xavier_init)
print('set initials')
# %%
# 損失関数の設定
class SegLoss(nn.Module):
    """PSPNetの損失関数のクラスです。"""

    def __init__(self):
        super(SegLoss, self).__init__()
        
    def f(self, x, y):
        z = torch.zeros_like(x)
        return torch.sum(torch.max(-(x - y), z) ** 2 / 16 + torch.max((x - y), z) ** 2)

    def forward(self, outputs, outputs_color, targets_color, targets_mask):
#         print(outputs.shape)
#         print(targets_color.shape)
#         print(targets_mask.shape)

        loss = self.f(targets_mask, outputs)
        
        outputs_color = outputs_color * targets_mask
        targets_color = targets_color * targets_mask

        loss_color = torch.sum((outputs_color - targets_color) ** 2)

        return loss / 5 + loss_color

criterion = SegLoss()
# %%
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(net.parameters(), lr=1e-8, momentum=0.9) # lr=1e-7 だと発散した

def lambda_epoch(epoch):
    max_epoch = 10000
    return math.pow((1-epoch/max_epoch), 0.9)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)
# %%
# モデルを学習させる関数を作成


def train_model(net, dataloader, criterion, scheduler, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # ネットワークをGPUへ
    net.to(device)
    
#     state_dict = torch.load("./weights/16/segnetbasic_1000.pth")
#     net.load_state_dict(state_dict)   

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    
    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    # multiple minibatch
    batch_multiplier = 3

    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0  # epochの損失和

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        net.train()  # モデルを訓練モードに
        scheduler.step()  # 最適化schedulerの更新
        optimizer.zero_grad()
        print('（train）')

        # データローダーからminibatchずつ取り出すループ
        count = 0  # multiple minibatch
        for imges, anno_color_class_imges, anno_mask_class_imges in dataloader:
            # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
            # issue #186より不要なのでコメントアウト
            # if imges.size()[0] == 1:
            #     continue

            # GPUが使えるならGPUにデータを送る
            imges = imges.to(device)
            anno_color_class_imges = anno_color_class_imges.to(device)
            anno_mask_class_imges = anno_mask_class_imges.to(device)


            # multiple minibatchでのパラメータの更新
#             if (count == 0):
            optimizer.step()
            optimizer.zero_grad()
            count = batch_multiplier

            # 順伝搬（forward）計算
            with torch.set_grad_enabled(True):
#                 print('aaa', imges[0, :, 220: 240, 240])
                outputs, outputs_color = net(imges)
#                 print('bbb', outputs[0, :, 220: 240, 240])
                
                loss = criterion(outputs, outputs_color, anno_color_class_imges, anno_mask_class_imges)
                loss.backward()  # 勾配の計算
                count -= 1  # multiple minibatch

                if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                    t_iter_finish = time.time()
                    duration = t_iter_finish - t_iter_start
                    print('イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                        iteration, loss.item()/batch_size*batch_multiplier, duration))
                    t_iter_start = time.time()

                epoch_train_loss += loss.item() * batch_multiplier
                iteration += 1

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss/num_train_imgs))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # ログを保存
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss / num_train_imgs}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        # 最後のネットワークを保存する
        if (epoch+1) % 100 == 0:
            torch.save(net.state_dict(), f'{WIGHTS_DIR}/segnetbasic_' +
                       str(epoch+1) + '.pth')
# %%
# 学習・検証を実行する
num_epochs = 10000
train_model(net, train_dataloader, criterion, scheduler, optimizer, num_epochs=num_epochs)
# %%
from modules.segnet.utils.dataloader import make_datapath_list_angle, DataTransform_angle, MyDataset_angle
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

rootpath = f"{SEGNET_DATASETS_PATH}/test"
test_img_list, test_color_anno_list, test_mask_anno_list= make_datapath_list_angle(rootpath=rootpath, start_index=447, skip=4)

color_mean = (0.50587555, 0.55623757, 0.4438057)
color_std = (0.19117119, 0.17481992, 0.18417963)

val_dataset = MyDataset_angle(test_img_list, test_color_anno_list, test_mask_anno_list, phase="val", transform=DataTransform_angle(
    input_size=(160, 120), color_mean=color_mean, color_std=color_std))


img_index = 1

print(test_img_list)
print(len(test_img_list))
print(len(test_color_anno_list))
print(len(test_mask_anno_list))


# 1. 元画像の表示
image_file_path = test_img_list[img_index]
img_original = Image.open(image_file_path)   # [高さ][幅][色RGB]
img_width, img_height = img_original.size
plt.imshow(img_original)
plt.show()

# 2. 正解アノテーション画像の表示
anno_color_file_path = test_color_anno_list[img_index]
anno_color_class_img = Image.open(anno_color_file_path)   # [高さ][幅][色RGB]
img_width, img_height = anno_color_class_img.size
plt.imshow(anno_color_class_img)
plt.show()
anno_mask_file_path = test_mask_anno_list[img_index]
anno_mask_class_img = Image.open(anno_mask_file_path)   # [高さ][幅][色RGB]
img_width, img_height = anno_mask_class_img.size
plt.imshow(anno_mask_class_img, cmap = "gray")
plt.show()


print(WIGHTS_DIR)

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

mask = anno_mask_class_img[0].cpu().detach().numpy()

y = np.clip(y, 0, 1)
y_c = np.clip(y_c, 0, 1)

# squeezeで次元削減
plt.imshow(np.squeeze(y, -1), cmap = "gray")
plt.show()
plt.imshow(y_c)
plt.show()

# import cv2
# y_c = cv2.bitwise_and(y_c,y_c, mask= y)
# plt.imshow(y_c)
# plt.show()

sum_value = np.array([0.0, 0.0, 0.0])
cnt = np.array([0.0, 0.0, 0.0])
all_value = 0
for i in range(120):
    for j in range(160):
        sum_value += y_c[i][j] * y[i][j]
        cnt += y[i][j]

sum_value /= cnt
print(sum_value)

import colorsys
h, s, v = colorsys.rgb_to_hsv(sum_value[0], sum_value[1], sum_value[2])
print(h, s, v)
print(f'h : {h * 90}')
# %%
import cv2
import matplotlib.pyplot as plt

def binarize(src_img):
#     bin_img = cv2.threshold(src_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    bin_img = cv2.threshold(src_img,220,255,cv2.THRESH_BINARY)[1]
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

y = np.clip(y, 0, 1)
plt.imshow(np.squeeze(y, -1), cmap = "gray")
plt.show()
bin_img = binarize((y*255).astype(np.uint8))[:, :, np.newaxis]
plt.imshow(np.squeeze(bin_img, -1), cmap = "gray")
plt.show()

retval, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
src_img = cv2.resize(cv2.imread(test_img_list[img_index]), dsize=(160, 120))
result_img = draw_result(src_img, stats, centroids)

plt.imshow(result_img)
plt.show()

bin_img_result = draw_bin_img_result(bin_img, stats, labels)
plt.imshow(np.squeeze(bin_img_result, -1), cmap = "gray")
plt.show()
# %%
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
h, s, v = colorsys.rgb_to_hsv(sum_value[0], sum_value[1], sum_value[2])
print(h, s, v)
print(f'h : {h * 90}')
# %%
radius = 50
line_color = (120, 20, 30)
line_width = 4
circle_color = (230, 10, 240)
circle_size = 3
circle_width = 3

def draw_batten(src_img, stats, centroids, angle):
    maxv, maxi = max_area(stats)
    result_img = src_img.copy()    
    coordinate = centroids[maxi]
    x, y = (int(coordinate[0]), int(coordinate[1]))

    angle_r = np.deg2rad(angle)
    x1, y1 = x+int(radius*np.cos(angle_r)), y-int(radius*np.sin(angle_r))
    x2, y2 = x-int(radius*np.cos(angle_r)), y+int(radius*np.sin(angle_r))
    cv2.line(result_img, (x1, y1), (x2, y2), line_color, line_width)
    cv2.circle(result_img, (x1, y1), circle_width, circle_color, circle_width)
    cv2.circle(result_img, (x2, y2), circle_width, circle_color, circle_width)

    angle_r = np.deg2rad(angle+90)
    x1, y1 = x+int(radius*np.cos(angle_r)), y-int(radius*np.sin(angle_r))
    x2, y2 = x-int(radius*np.cos(angle_r)), y+int(radius*np.sin(angle_r))
    cv2.line(result_img, (x1, y1), (x2, y2), line_color, line_width)
    cv2.circle(result_img, (x1, y1), circle_width, circle_color, circle_width)
    cv2.circle(result_img, (x2, y2), circle_width, circle_color, circle_width)
    
    
    return result_img

src_img = cv2.resize(cv2.imread(test_img_list[img_index]), dsize=(160, 120))
img = draw_batten(src_img, stats, centroids, h * 90)
plt.imshow(img)
plt.show()