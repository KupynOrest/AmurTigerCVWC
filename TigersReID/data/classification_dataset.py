import os
import random
from os.path import join
from functools import reduce, partial
import random
from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as torch_transforms
import glob
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data import transforms
import csv
import json
from albumentations import (
    HorizontalFlip, IAAAdditiveGaussianNoise, ShiftScaleRotate, CLAHE, ToGray,
    GaussNoise, ShiftScaleRotate, Blur, MotionBlur, MedianBlur, OpticalDistortion, GridDistortion, IAAPiecewiseAffine, HueSaturationValue,
    Resize, IAASharpen, IAAEmboss, RandomGamma, RandomContrast, RandomBrightness, Flip, OneOf, Compose, Normalize, RGBShift, JpegCompression
)

class ClassificationDataset(BaseDataset):
    def initialize(self, config, filename):
        self.config = config
        self.train = 'train' in filename
        self.size = config['image_size']

        if self.train:
            self.transform = Compose([
                Resize(self.size[0], self.size[1]),
                ShiftScaleRotate(shift_limit=0.3, scale_limit=(0.05, 0.1), rotate_limit=10, p=.4),
                OneOf([
                    IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.4),
                OneOf([
                    MotionBlur(p=.2),
                    MedianBlur(blur_limit=5, p=.5),
                    Blur(blur_limit=3, p=.5),
                ], p=0.4),
                OpticalDistortion(p=0.4),
                OneOf([
                    CLAHE(clip_limit=3),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomContrast(),
                    RandomBrightness(),
                    RandomGamma()
                ], p=0.6),
                OneOf([
                    RGBShift(),
                    HueSaturationValue(),
                ], p=0.2),
                JpegCompression(quality_lower=30, quality_upper=100, p=0.4),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = Compose([
                Resize(self.size[0], self.size[1]),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            self.anno = list(reader)
            
        with open(self.config['datasets']['bboxs'], 'r') as bboxs_file:
            bboxs_json = json.load(bboxs_file)
            self.bboxs_dict = {}
            for item in bboxs_json:
                if item['image_id'] in self.bboxs_dict:
                    fixed_bbox_new, area_new = self.fix_bbox(item['bbox'])
                    _, area_old = self.fix_bbox(self.bboxs_dict[item['image_id']])
                    if area_new > area_old:
                        self.bboxs_dict[item['image_id']] = fixed_bbox_new
                else:
                    self.bboxs_dict[item['image_id']] = self.fix_bbox(item['bbox'])[0]

        random.shuffle(self.anno)

        self.anno_size = len(self.anno)
    
    def get_bbox(self, img_path):
        img_id = int(img_path.split('.')[0])
        if img_id in self.bboxs_dict:
            bbox = self.bboxs_dict[img_id]
            width = (bbox[2] - bbox[0])
            height = (bbox[3] - bbox[1])
            return bbox if (width > 50 and height > 50) else None
        else:
            return None
    
    def fix_bbox(self, bbox):
        fixed_bbox = np.asarray(bbox).astype(int)
        fixed_bbox[fixed_bbox<0] = 0
        area = (fixed_bbox[2] - fixed_bbox[0]) * (fixed_bbox[3] - fixed_bbox[1])
        return fixed_bbox, area

    def __getitem__(self, index):
        img_path, whale_id = self.anno[index][2], self.anno[index][0]
        whale_id = int(whale_id)
        img = cv2.imread(join(self.config['dataroot'], img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_bbox = self.get_bbox(img_path)
        if img_bbox is not None:
            img = img[img_bbox[1]:img_bbox[3], img_bbox[0]:img_bbox[2]]
    
        img = self.transform(image=img)['image']
        A = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))

        return {'A': A, 'id': whale_id, 'A_paths': join(self.config['dataroot'], img_path)}

    def __len__(self):
        return self.anno_size

    def name(self):
        return 'ClassificationDataset'
