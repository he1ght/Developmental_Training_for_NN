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

class ffn_two_layers(nn.Module):
    def __init__(self, hidden=500, dropout=0.1, batch_norm=False):
        super(ffn_two_layers, self).__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(self.hidden)
            self.bn2 = nn.BatchNorm1d(self.hidden)

        self.fc1 = nn.Linear(28 * 28, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        return x


# model_list = glob.glob(args.checkpoint + '*.pt')
#
# name_list = copy.deepcopy(model_list)
# name_list = [_name[:-4] for _name in name_list]
# name_list = list(set(name_list))



dist_mat_file = args.dist_mat
if args.data:
    save_path = '/'.join(args.checkpoint.split('/')[:-1]) + '/' + args.data[:-3]
else:
    save_path = '/'.join(args.checkpoint.split('/')[:-1]) + '/' + 'MNIST_eval'
try:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
except FileExistsError:
    pass
save_path += '/{}_{}'.format(args.method, args.degree)
try:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
except FileExistsError:
    pass
save_name = args.checkpoint.split('/')[-1][:-6]
try:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
except FileExistsError:
    pass

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

def get_entropy():
    # Sum{-p(x)log_2(p(x))}
    torch.distributions.Distribution.entropy()
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            for i in tqdm(range(data.size(0))):
                preds[i] = pred[i].item()
                gt[i] = target[i].item()

def get_pred_vec(model):
    preds = [0 for _ in range(len(test_dataset))]
    # gt = [0 for _ in range(len(test_dataset))]
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            for i in range(data.size(0)):
                preds[i] = pred[i].item()
                # gt[i] = target[i].item()
    return preds, None # gt

def _cal_entropy(prob):
    _ent = []
    for _p in prob:
        if not math.isnan(_p):
            if _p == 0. or _p == 1.:
                _ent.append(0.)
            else:
                # print(_p)
                _ent.append(- (1. - _p) * np.log2(1. - _p) - _p * np.log2(_p))
        else:
            _ent.append(float('nan'))
    return _ent

def detect_inner_points(dist_mat, pred_vec, dist=1, n_class=10):
    _inner_o, _inner_x = [0 for _ in range(n_class)], [0 for _ in range(n_class)]

    for i in range(len(dist_mat)):
        for j in range(len(dist_mat)):
            if i != j and dist_mat[i][j] <= dist:
                if pred_vec[i] == pred_vec[j]:
                    _inner_o[pred_vec[i]] += 1
                else:
                    _inner_x[pred_vec[i]] += 1

    _prob_x = []
    for c in range(n_class):
        try:
            _prob_x.append(float(_inner_x[c] / (_inner_o[c] + _inner_x[c])))
        except ZeroDivisionError:
            _prob_x.append(float('nan'))
    return _prob_x, _cal_entropy(_prob_x)

def detect_neigh_points(dist_mat, pred_vec, count=1, n_class=10):
    _inner_o, _inner_x = [0 for _ in range(n_class)], [0 for _ in range(n_class)]
    for i in range(len(dist_mat)):
        _count = [0 for _ in range(n_class)]
        for j in sorted(range(len(dist_mat[i])), key=dist_mat[i].__getitem__):
            if i != j and _count[pred_vec[i]] < count:
                if pred_vec[i] == pred_vec[j]:
                    _inner_o[pred_vec[i]] += 1
                else:
                    _inner_x[pred_vec[i]] += 1
                _count[pred_vec[i]] += 1

    _prob_x = []
    for c in range(n_class):
        try:
            _prob_x.append(_inner_x[c] / (_inner_o[c] + _inner_x[c]))
        except ZeroDivisionError:
            _prob_x.append(0.)
    return _prob_x, _cal_entropy(_prob_x)

# dist_mat = get_dist_mat()

if args.method == 'ent':
    pass
else:
    try:
        dist_mat = torch.load(dist_mat_file)
    except FileNotFoundError:
        # dist_mat = get_dist_mat()
        dist_mat = None

    if args.method == 'dist':
        detect_func = detect_inner_points
    else:
        detect_func = detect_neigh_points
# print(dist_mat)
# f = open('mnist_dist_matrix.csv', 'w', newline='')
# wr = csv.writer(f)
# wr.writerows(dist_mat)
# f.close()
# import matplotlib.pyplot as plt
# yy = []
# for i in range(len(test_dataset)):
#     for j in range(len(test_dataset)):
#         if j < i:
#             # yy.append(dist_mat[i][j])
#             if dist_mat[i][j] < 30:
#                 print(dist_mat[i][j], end=' ')
# plt.hist(yy,
#          bins=20, ## 몇 개의 바구니로 구분할 것인가.
#          density=True, ## ytick을 퍼센트비율로 표현해줌
#          cumulative=False, ## 누적으로 표현하고 싶을 때는 True
#          histtype='bar',  ## 타입. or step으로 하면 모양이 바뀜.
#          orientation='vertical', ## or horizontal
#          rwidth=0.8, ## 1.0일 경우, 꽉 채움 작아질수록 간격이 생김
#          )
# plt.savefig('test.png')
# plt.close()

if dist_mat is not None:
    prob_result, ent_result = [], []
    __seed = 1
    for _seed in tqdm(range(1, args.seeds + 1)):
        try:
            checkpoint_seed_name = args.checkpoint + str(_seed) + '.pt'
            checkpoint = torch.load(checkpoint_seed_name, map_location=lambda storage, loc: storage.cuda(args.gpu))
            model_args = checkpoint['args']
            model = ffn_two_layers(model_args['hidden'], model_args['dropout'], batch_norm=model_args['batch_norm'])

            model.load_state_dict(checkpoint['model'])
            if args.cuda:
                model = model.to(device)

            pred_list, ground_truth = get_pred_vec(model)
            prob_list, entropy_list = detect_func(dist_mat, pred_list, args.degree, num_classes)
            prob_result.append(prob_list)
            ent_result.append(entropy_list)
            __seed = _seed
        except FileNotFoundError:
            __seed = _seed - 1
            break
    if __seed > 0:
        prob_result = np.array(prob_result)
        ent_result = np.array(ent_result)
        with open(save_path + '/' + save_name + '.dat', 'w') as f:
            for p, e in zip(prob_result.mean(axis=0), ent_result.mean(axis=0)):
                f.write('{} {}\n'.format(p, e))
            # f.writelines(result.mean(axis=0))

# print("--- %s seconds ---" % (time.time() - start_time))
