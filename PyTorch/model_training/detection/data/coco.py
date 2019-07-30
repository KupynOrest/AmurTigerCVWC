import os.path as osp

import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
from pycocotools.coco import COCO

from .transforms import get_transform


class COCODetection(data.Dataset):
    """MS Coco Detection Dataset.
    http://cocodataset.org/#format-data
    """

    def __init__(self, ann_path, imgs_path, transform=None):
        """
        Args:
            ann_path (str): path to json file with annotations
            imgs_path (str): path to directory with images
            transform: image and bbox transformation
        """
        self.imgs_path = imgs_path
        self.coco = COCO(ann_path)
        self.transform = transform

        self.ids = list(self.coco.imgToAnns.keys())

    def __getitem__(self, index):
        """Returns image and its annotation
        Args:
            index (int): item index
        Returns:
            tuple: Tuple (image, target).
                   image is torch.tensor of shape (C, W ,H),
                   target is numpy array of (x, y, w, h, c), where c is class id
        """
        im, gt = self.pull_item(index)
        return im, gt

    def __len__(self):
        """Returns number of items in dataset
        Returns:
            int: dataset length
        """
        return len(self.ids)

    def pull_item(self, index):
        """Returns image and its annotation
        Args:
            index (int): index of item
        Returns:
            tuple: Tuple (image, target).
                   image is torch.tensor of shape (C, W ,H),
                   target is numpy.ndarray of (x, y, w, h, c), where c is class id
        """
        try:
            img = self.pull_image(index)
            bboxes, labels = self.pull_annotation(index)

            # random_idx = random.randrange(self.__len__())
            # augm_img = self.pull_image(random_idx)
            # augm_bboxes, augm_labels = self.pull_annotation(random_idx)

            # count = random.randrange(len(augm_labels))
            # indexes = np.arange(len(augm_labels))
            # np.random.shuffle(indexes)
            # indexes = indexes[:count]
            # augm_bboxes, augm_labels = np.array(augm_bboxes)[indexes], np.array(augm_labels)[indexes]

            # for a_box, a_label in zip(augm_bboxes, augm_labels):
            #     labels.append(a_label)
            #
            #     x, y, w, h = a_box
            #     x0 = random.randrange(img.shape[1] - w - 1)
            #     y0 = random.randrange(img.shape[0] - h - 1)
            #     img[y0:y0+h, x0:x0+w] = augm_img[y:y+h, x:x+w]
            #     bboxes.append([x0, y0, w, h])

            if self.transform is not None:
                transformed = self.transform(image=img, bboxes=bboxes, category_id=labels)
                img = transformed["image"]
                bboxes = transformed["bboxes"]
                labels = transformed["category_id"]

            # cv2.imwrite(f'test_{random.randrange(20000)}.jpg', img)
            img = self._preprocess(img)
            _, height, width = img.shape
            bboxes = np.array(bboxes)
            bboxes[:, 2] += bboxes[:, 0]
            bboxes[:, 3] += bboxes[:, 1]
            bboxes = bboxes / np.array([width, height, width, height], dtype=np.float)  # normalize
            target = np.hstack([bboxes, np.expand_dims(labels, axis=1)])
        except Exception as e:
            if len(bboxes) > 0:
                raise e
            print(osp.join(self.imgs_path, self.coco.loadImgs(self.ids[index])[0]["file_name"]))
            return self.pull_item(random.randrange(len(self)))

        return torch.from_numpy(img), target

    def pull_image(self, index):
        """Returns the original image object at index
        Argument:
            index (int): index of img to show
        Return:
            numpy.ndarray: RGB image of shape (W, H, C)
        """
        img_id = self.ids[index]
        img = cv2.imread(osp.join(self.imgs_path, self.coco.loadImgs(img_id)[0]["file_name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def pull_annotation(self, index):
        """Returns the original annotation of image at index
        Argument:
            index (int): index of img to get annotation of
        Return:
            bboxes (list): list of (x, y, w, h)
            categories (list): list of category ids
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes, categories = [], []
        for ann in anns:
            bboxes.append(ann['bbox'])
            categories.append(ann["category_id"] - 1)
        return bboxes, categories

    @staticmethod
    def _preprocess(img):
        """Changes order of image channels to (C, W ,H)
        Args:
            img (numpy.ndarray): RGB image of shape (C, W, H)
        Returns:
            numpy.ndarray: image of shape (C, W, H)
        """
        img = np.transpose(img, (2, 0, 1))
        return img


def get_dataset(config):
    """Creates COCO dataset from given config
    Args:
        config (dict): dictionary with COCO dataset configuration:
            img_size (int): image size
            transform (str): transformation type ('weak', 'strong')
            ann_path (str): path to json file with annotations
            img_path (str): path to directory with images
    Returns:
        COCODetection: dataset for given configuration
    """
    transform = get_transform(config['img_size'], transform_type=config['transform'])
    return COCODetection(config["ann_path"], config["img_path"], transform=transform)
