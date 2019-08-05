import os.path
import random
import os
from os.path import join
from functools import reduce, partial
from collections import deque
import csv
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as torch_transforms

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data import transforms
from albumentations import (
    HorizontalFlip, IAAAdditiveGaussianNoise, ShiftScaleRotate, CLAHE, ToGray,
    GaussNoise, ShiftScaleRotate, Blur, RGBShift, JpegCompression, OpticalDistortion, GridDistortion, IAAPiecewiseAffine, HueSaturationValue,
    Resize, IAASharpen, IAAEmboss, RandomGamma, RandomContrast, RandomBrightness, Flip, OneOf, Compose, Normalize
)


class OnlineDataset(BaseDataset):
    def initialize(self, config, filename):
        self.config = config
        self.train = 'train' in filename

        self.A_paths = self.get_files(filename)
#         self.boxes = self.get_boxes()

        self.A_size = self.A_paths.shape[0]
        self.batch_size = config['batch_size']
        self.size = config['image_size']
        self.transform = Compose([
#             Resize(self.size[0], self.size[1]),
            ShiftScaleRotate(shift_limit=0.0, scale_limit=(0.0, 0.1), rotate_limit=10, p=.3),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.4),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
                RandomGamma()
            ], p=0.6),
            OneOf([
                RGBShift(),
                HueSaturationValue(),
            ], p=0.6),
            JpegCompression(quality_lower=40, quality_upper=100, p=0.3),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.used_classes = deque()

    def get_boxes(self):
        with open('/home/okupyn/bounding_boxes_train.csv', 'r') as f:
            reader = csv.reader(f)
            anno = list(reader)
        bb_dict = {}
        for row in anno:
            file_name = row[0]
            top_left = row[1].replace('(', '').replace(')', '').split(',')
            bottom_right = row[2].replace('(', '').replace(')', '').split(',')
            top_left = (int(float(top_left[0])), int(float(top_left[1])))
            bottom_right = (int(float(bottom_right[0])), int(float(bottom_right[1])))
            bb_dict[file_name] = (top_left, bottom_right)
        return bb_dict

    def get_files(self, filename):
        return pd.read_pickle(filename)

    def _add_to_used(self, index, class_id):
        if index == 0 and len(self.used_classes) == self.batch_size:
            self.used_classes.clear()
        if len(self.used_classes) >= self.batch_size:
            self.used_classes.popleft()
        self.used_classes.append(class_id)

    def get_tensor(self, path, grayscale=False):
        try:
            img = cv2.imread(path)
            _, name = os.path.split(path)
#             top_left, bottom_right = self.boxes[name]
#             top_left = (max(0, top_left[0] - 30), max(0, top_left[1] - 30))
#             bottom_right = (min(img.shape[0], bottom_right[0] + 30), min(img.shape[1], bottom_right[1] + 30))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            img = self.transform(image=img)['image']
            if grayscale:
                img = ToGray(p=1.)(image=img)['image']
            img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
        except:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(image=img)['image']
            if grayscale:
                img = ToGray(p=1.)(image=img)['image']
            img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
        return img_tensor

    def __getitem__(self, index):
        if index % 4 == 0:
            class_id = 'new_whale'
            id = 0
        else:
            class_id = random.choice([i for i in range(0, self.A_size - 1) if (i is not 'new_whale' and i not in self.used_classes)])
            self._add_to_used(index, class_id)
            id = index

        img_list = self.A_paths[class_id]

        grayscale = False
        if len(img_list) == 1:
            same_class = [img_list[0], img_list[0], img_list[0]]
            grayscale = True
        elif len(img_list) == 2:
            same_class = [img_list[0], img_list[1], img_list[1]]
        else:
            same_class = random.sample(img_list, 3)

        img1 = self.get_tensor(join(self.config['dataroot'], same_class[0]))
        img2 = self.get_tensor(join(self.config['dataroot'], same_class[1]), grayscale)
        img3 = self.get_tensor(join(self.config['dataroot'], same_class[2]), grayscale)
        return {'img1': img1, 'img2': img2, 'img3': img3, 'class': id}

    def __len__(self):
        return 500 * self.batch_size

    def name(self):
        return 'OnlineDataset'
