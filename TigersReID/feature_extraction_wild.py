from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
import numpy as np
from models.net_sphere import SphereNet
from models.networks import  EmbeddingNet
from PIL import Image
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import cv2
from glob import glob
from albumentations import (Resize, Compose, Normalize)
import csv

def get_args():
    parser = argparse.ArgumentParser('Generate feature embeddings')
    parser.add_argument('--dataframe_path', required=True, help='path for CSV files')
    parser.add_argument('--weights_path', required=True, help='pytorch weights path')
    parser.add_argument('--model', type=str, default='triplet_net', help='model name')
    parser.add_argument('--embedding_mapping_file', type=str, default='output_mappings', help='embeddings mapping file name')
    parser.add_argument('--embedding_file', type=str, default='img_embeddings.npm', help='embeddings file name')
    parser.add_argument('--embedding_size', type=int, default=512, help='embeddings size')
    parser.add_argument('--loss_type', type=str, default='cos_loss', help='sphere loss type')
    parser.add_argument('--backbone', type=str, default='se_resnext_50', help='embeddings size')

    return parser.parse_args()


def get_dataset(df_path):
#     df = pd.read_csv(df_path)
#     with open(df_path, 'r') as f:
#         reader = csv.reader(f)
#         img_path_list = list(reader)

#     return img_path_list
    imgs_list = glob('dataset/wild/test_cut/*.jpg')
    return imgs_list


def get_model(args):
    if args.model == 'sphere_net':
        model = SphereNet(EmbeddingNet(backbone=args.backbone), loss_type=args.loss_type, classnum=107)
    elif args.model == 'triplet_net':
        model = EmbeddingNet(backbone=args.backbone)
    else:
        raise ValueError("Model [%s] not recognized." % model_name)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.weights_path)['model'])
    model.cuda().eval()
    return model


def get_boxes():
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


def compute_embeddings(img_dataset, model, args):
    n_images = len(img_dataset)
    print("Number of images in the dataset = {}".format(n_images))
    output_embeddings_file = np.memmap(
        args.embedding_file,
        dtype='float32',
        mode='w+',
        shape=(n_images, args.embedding_size)
    )

    transform = Compose([
        Resize(256, 512),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
#     boxes = get_boxes()

    output_mappings = [None] * n_images
    print(img_dataset[0:10])

    def get_embedding(i, data):
        img_path = data
        with torch.no_grad():
            try:
                img = cv2.imread(img_path)
#                 top_left, bottom_right = boxes[data['Image']]
#                 top_left = (max(0, top_left[0] - 30), max(0, top_left[1] - 30))
#                 bottom_right = (min(img.shape[0], bottom_right[0] + 30), min(img.shape[1], bottom_right[1] + 30))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
                img = transform(image=img)['image']
                image_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
            except:
                print(img_path)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transform(image=img)['image']
                image_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
            image_tensor = image_tensor.unsqueeze(0).cuda()
            embedding = model.module.get_embedding(image_tensor)

        output_mappings[i] = str(i) + ',' + data.split('/')[-1]
        output_embeddings_file[i, :] = embedding.cpu().numpy()
    Parallel(n_jobs=32, require='sharedmem')(delayed(get_embedding)(i, data) for (i, data) in enumerate(tqdm(img_dataset)))
    print(output_embeddings_file)
    print(output_embeddings_file.shape)
    np.save(args.embedding_mapping_file, np.array(output_mappings))


if __name__ == '__main__':
    args = get_args()
    model = get_model(args)
    dataset = get_dataset(args.dataframe_path)
    compute_embeddings(dataset, model, args)
