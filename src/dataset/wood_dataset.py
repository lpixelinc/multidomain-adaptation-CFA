import os
import csv
import pandas as pd
import numpy as np 
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class WoodDataset(Dataset):
    def __init__(self, csv_path, is_train=True, resize=256, cropsize=224, crop_flag=False):
        self.csv_path = csv_path
        self.is_train = is_train
        self.resize = resize
        self.crop_flag = crop_flag
        self.cropsize = cropsize
        
        # csvからデータを読み込む
        (self.image_path_list,
         self.label_list,
         self.mask_path_list
        ) = self.load_dataset_folder()
  
        # 画像の前処理・マスクの前処理を定義
        if self.crop_flag:
            self.transform_image =   T.Compose([T.RandomCrop(cropsize),
                                                T.ToTensor(),
                                                T.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

            self.transform_mask  =    T.Compose([T.RandomCrop(cropsize),
                                                 T.ToTensor()])
                                                
        else:
            self.transform_image =   T.Compose([T.ToTensor(),
                                                T.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

            self.transform_mask  =    T.Compose([T.ToTensor()])
        
    def __getitem__(self, idx):
        # image_path・label・mask_pathを取得
        image_path = self.image_path_list[idx]
        label = self.label_list[idx]
        mask_path = self.mask_path_list[idx]
        # 画像を読み込み、RGBに変換する
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 画像をpaddingする
        image = self.padding(image)
        # 画像をresizeする
        image = cv2.resize(image, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_CUBIC)
        # 標準化、tensor化などの前処理を行う
        image = self.transform_image(image)

        if label == 0:
            if self.crop_flag:
                mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
            else:
                mask = torch.zeros([1, self.resize[0], self.resize[1]])
        else:
            # maskを読み込み、Grayに変換する
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # maskをpaddingする
            mask = self.padding(mask.reshape(mask.shape[0], mask.shape[1], 1))
            # 画像をresizeする
            mask = cv2.resize(mask, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_CUBIC)
            # tensor化などの前処理を行う
            mask = self.transform_mask(mask)

        return image, label, mask.to(torch.int8)

    def __len__(self):
        return len(self.image_path_list)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'

        df = pd.read_csv(self.csv_path)
        image_path_list = df['image_path'].tolist()
        label_list = df['label'].tolist()
        mask_path_list = df['mask_path'].tolist()
        
        assert len(image_path_list) == len(label_list), 'number of image_path and label  should be same'

        return list(image_path_list), list(label_list), list(mask_path_list)
    
    def padding(self, image):
        # imageの縦横サイズを取得
        height, width, color = image.shape
        # 縦長画像→幅を拡張する
        if height > width:
            diffsize = height - width
            # 元画像を中央ぞろえにしたいので、左右に均等に余白を入れる
            padding_half = int(diffsize / 2)
            if color == 3:
                padding_image = cv2.copyMakeBorder(image, 0, 0, padding_half, padding_half, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            else:
                padding_image = cv2.copyMakeBorder(image, 0, 0, padding_half, padding_half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # 横長画像→高さを拡張する
        elif width > height:
            diffsize = width - height
            padding_half = int(diffsize / 2)
            if color == 3:
                padding_image = cv2.copyMakeBorder(image, padding_half, padding_half, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            else:
                padding_image = cv2.copyMakeBorder(image, padding_half, padding_half, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
        # 例外処理について
        elif width == height:
            padding_image = image
            
        return padding_image


if __name__=='__main__':
    wood_dataset = WoodDataset(csv_path='../wood_dataset/train/train_labels.csv')
    breakpoint()