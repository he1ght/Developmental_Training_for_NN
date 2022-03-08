from __future__ import print_function

import argparse
import os
import time

import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F

start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--save', type=str, default='data',
                    help='')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--amount', type=int, default=10000,
                    help='')
parser.add_argument('--restrict', action='store_true',
                    help='')

args = parser.parse_args()

torch.manual_seed(args.seed)

num_classes = 10

tr = datasets.MNIST('../data_mnist', train=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.1307,), (0.3081,))
                    ]))
train_loader = torch.utils.data.DataLoader(tr, batch_size=16, shuffle=True)


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


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, input):
        x = input / 255.0
        # print(x)
        x = x - self.mean
        x = x / self.std
        return x


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


# [batch x 1 x 28 x 28]
def _generate_uniform_toy(size: tuple, amount: int, _min: int, _max: int):
    dim = (amount, *[s for s in size])
    dummy = torch.randint(_min, _max, size=dim)
    dummy = dummy.to(torch.uint8)
    norm = Normalize(0.1307, 0.3081)
    dummy = norm(dummy)
    dummy = ToyUniformDataset(dummy)
    torch.save(dummy, args.save + '.pt')

def _generate_restricted_uniform_toy(size: tuple, amount: int, _min_mat, _max_mat):
    dim = (amount, *[s for s in size])
    _tot_len = 1
    dummy_list = []
    for i in range(len(_min_mat)):
        for j in range(len(_min_mat[i])):
            dummy = torch.randint(int(_min_mat[i][j]), int(_max_mat[i][j]) + 1, size=(amount,))
            dummy_list.append(dummy)
    dummy_list = torch.stack(dummy_list)
    dummy_list = dummy_list.transpose(0, 1)
    dummy_list = dummy_list.resize(*[s for s in dim])
    dummy = dummy_list.to(torch.uint8)
    norm = Normalize(0.1307, 0.3081)
    dummy = norm(dummy)
    dummy = ToyUniformDataset(dummy)
    torch.save(dummy, args.save + '.pt')


# dist_mat = get_dist_mat()




if args.restrict:
    with torch.no_grad():
        _min_mat = torch.min(tr.data, dim=0, keepdim=True)[0].squeeze()
        _max_mat = torch.max(tr.data, dim=0, keepdim=True)[0].squeeze()
        _generate_restricted_uniform_toy((28, 28), args.amount, _min_mat, _max_mat)
else:
    _generate_uniform_toy((28, 28), args.amount, 0, 256)