from __future__ import print_function

import argparse
import math
import os
import time

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class ToyUniformDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        self.x_data = x
        self.y_data = [0 for _ in range(x.size(0))]

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=60000, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--seeds', type=int, default=5,
                    help='random seed (default: 5)')
parser.add_argument('--checkpoint', type=str, default='model/',
                    help='Checkpoint name')
parser.add_argument('--data', type=str, default='',
                    help='Data')
parser.add_argument('--dist-mat', type=str, default='mnist_dist_matrix.pt',
                    help='Data')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU no. (default: 0)')
parser.add_argument('--method', type=str, default='dist', choices=['dist', 'knn', 'ent'],
                    help='')
parser.add_argument('--degree', type=int, default=10,
                    help='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
for_sKD_diff_test = True

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:' + str(args.gpu))

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

# tr = datasets.MNIST('./data_mnist', train=True, download=True,
#                     transform=transforms.Compose([
#                         transforms.ToTensor(),
#                         transforms.Normalize((0.1307,), (0.3081,))
#                     ]))
# train_loader = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, **kwargs)
if args.data:
    test_dataset = torch.load(args.data)
else:
    test_dataset = datasets.MNIST('./data_mnist', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=10000000, shuffle=False, **kwargs)

num_classes = 10



dist_mat_file = args.dist_mat


def get_dist_mat():
    mat = [[0 for __ in range(len(test_dataset))] for _ in range(len(test_dataset))]
    with torch.no_grad():
        for data, target in test_loader:
            # if args.cuda:
            #     data, target = data.to(device), target.to(device)
            for i in tqdm(range(data.size(0))):
                for j in range(data.size(0)):
                    if i < j:
                        mat[i][j] = torch.pow(data[i] - data[j], 2).sum().item()
        # for i, x in tqdm(enumerate(test_dataset)):
        #     for j, y in enumerate(test_dataset):
        #         if i < j:
        #             mat[i][j] = torch.pow(x[0] - y[0], 2).sum()
        for i in range(len(test_dataset)):
            for j in range(len(test_dataset)):
                if j < i:
                    mat[i][j] = mat[j][i]
    torch.save(mat, dist_mat_file)
    return mat


try:
    dist_mat = torch.load(dist_mat_file)
except FileNotFoundError:
    dist_mat = get_dist_mat()
