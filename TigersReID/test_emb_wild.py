from __future__ import print_function
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine
import cv2
import json
from sklearn import preprocessing
import os
import torchvision.transforms as torch_transforms
from models.net_sphere import SphereNet
from models.networks import  EmbeddingNet
from torchvision import models, transforms
from torch.autograd import Variable
import shutil
import glob
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cosine
import csv
from albumentations import (Resize, Compose, Normalize)

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


def get_pred(image_path, model, boxes, t):
    with torch.no_grad():
        try:
            img = cv2.imread(image_path)
            _, name = os.path.split(image_path)
#             top_left, bottom_right = boxes[name]
#             top_left = (max(0, top_left[0] - 30), max(0, top_left[1] - 30))
#             bottom_right = (min(img.shape[0], bottom_right[0] + 30), min(img.shape[1], bottom_right[1] + 30))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            img = t(image=img)['image']
            image_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
        except:
            print(image_path)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = t(image=img)['image']
            image_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
        image_tensor = image_tensor.unsqueeze(0).cuda()
        embedding = model.module.get_embedding(image_tensor)

    return embedding.cpu().numpy().flatten()


def get_closest(embedding, emb_file, emb_mappings):
    def get_distances(x):
        return abs(0 - cosine(embedding, x))

    distances = np.apply_along_axis(get_distances, 1, emb_file)
    result = list(zip(distances, emb_mappings))
    result = sorted(result, key=lambda tup: tup[0])
    res = []
    for score, path in result:
#         class_id = path.split(',')[0]
        class_id = (path.split(',')[1]).split('.')[0]
        if class_id not in res:
            res.append(class_id)
#             if len(res) == 11:
#                 break
    return res


def get_boxes():
    with open('/home/okupyn/bounding_boxes_test.csv', 'r') as f:
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


def test(model):
    output_embeddings_file = np.memmap('embeddings.npm', dtype='float32',
                                       shape=(3843, 512),
                                       mode='r', )
    output_mappings = np.load('mappings.npy', mmap_mode='r')
    print(output_embeddings_file.shape)
    print(output_embeddings_file[:10])
    print(output_mappings.shape)
    print(output_mappings[:10])
#     test_file_list = glob.glob('/home/vbudzan/gan-lab/stacked_hourglass/data_aligned/data/test/*.jpg')
#     boxes = {}#get_boxes()
#     transform = Compose([
#         Resize(256, 256),
#         Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225],
#         ),
#     ])
    sub = {}
    for file in tqdm(output_mappings):
        _id, _name = file.split(',')
        sub[_name.split('.')[0]] = get_closest(output_embeddings_file[int(_id)], output_embeddings_file, output_mappings)

#     print(sub)
    with open('submission_wild.txt', 'w') as outfile:
        json.dump(sub, outfile)

if __name__ == '__main__':
    args = get_args()
#     model = get_model(args)
    test(None)
