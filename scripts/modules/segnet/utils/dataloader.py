# パッケージのimport
#import os.path as osp
import os
from PIL import Image
import torch.utils.data as data

from modules.segnet.utils.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor


def make_datapath_list(rootpath, start_index = 0):
    imgpath_template = os.path.join(rootpath, 'color', 'color%s.jpg')
    annopath_template = os.path.join(rootpath, 'anno', 'anno%s.jpg')
    print(imgpath_template)
    print(annopath_template)
    
    file_num = sum(os.path.isfile(os.path.join(rootpath, 'color', name)) \
                   for name in os.listdir(os.path.join(rootpath, 'color')))

    train_img_list = list()
    train_anno_list = list()
    print(f'ファイル数：{file_num}')
    for i in range(start_index, start_index + file_num):
#         print(i)
        img_path = (imgpath_template % f'{i:03}')  # 画像のパス
        anno_path = (annopath_template % f'{i:03}')  # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    return train_img_list, train_anno_list

class DataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
#                 Scale(scale=[0.5, 1.5]),  # 画像の拡大
#                 RandomRotation(angle=[-10, 10]),  # 回転
#                 RandomMirror(),  # ランダムミラー
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'val': Compose([
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)


class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]

        # 2. アノテーション画像読み込み
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)   # [高さ][幅]

        # 3. 前処理を実施
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img

    
from modules.segnet.utils.data_augumentation import Compose_angle, Resize_angle, Normalize_Tensor_angle

    
def make_datapath_list_angle(rootpath, start_index = 0, skip = 1):
    imgpath_template = os.path.join(rootpath, 'original', 'color%s.jpg')
    anno_color_path_template = os.path.join(rootpath, 'anno/color', 'anno%s.jpg')
    anno_mask_path_template = os.path.join(rootpath, 'anno/mask', 'anno%s.jpg')    
    file_num = sum(os.path.isfile(os.path.join(rootpath, 'original', name)) \
                   for name in os.listdir(os.path.join(rootpath, 'original')))

    train_img_list = list()
    train_anno_color_list = list()
    train_anno_mask_list = list()
    
    print(f'ファイル数：{file_num}')
    for i in range(file_num):
        img_path = (imgpath_template % f'{i*skip + start_index}')  # 画像のパス
        anno_color_path = (anno_color_path_template % f'{i*skip + start_index}')  # アノテーションのパス
        anno_mask_path = (anno_mask_path_template % f'{i*skip + start_index}')  # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_color_list.append(anno_color_path)
        train_anno_mask_list.append(anno_mask_path)

    return train_img_list, train_anno_color_list, train_anno_mask_list


class DataTransform_angle():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose_angle([
                Resize_angle(input_size),  # リサイズ(input_size)
                Normalize_Tensor_angle(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'val': Compose_angle([
                Resize_angle(input_size),  # リサイズ(input_size)
                Normalize_Tensor_angle(color_mean, color_std)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img, anno_color_class_img, anno_mask_class_img):
        return self.data_transform[phase](img, anno_color_class_img, anno_mask_class_img)

class MyDataset_angle(data.Dataset):
    def __init__(self, img_list, anno_color_list, anno_mask_list, phase, transform):
        self.img_list = img_list
        self.anno_color_list = anno_color_list
        self.anno_mask_list = anno_mask_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, anno_color_class_img, anno_mask_class_img = self.pull_item(index)
        return img, anno_color_class_img, anno_mask_class_img

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]

        # 2. アノテーション画像読み込み
        anno_color_file_path = self.anno_color_list[index]
        anno_color_class_img = Image.open(anno_color_file_path)   # [高さ][幅]
        anno_mask_file_path = self.anno_mask_list[index]
        anno_mask_class_img = Image.open(anno_mask_file_path)   # [高さ][幅]

        # 3. 前処理を実施
        img, anno_color_class_img, anno_mask_class_img = self.transform(self.phase, img, anno_color_class_img, anno_mask_class_img)

        return img, anno_color_class_img, anno_mask_class_img
