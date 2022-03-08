from __future__ import print_function

import argparse
import math
import time

import numpy as np
import torch

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
parser.add_argument('--dir', type=str, default='test_model/',
                    help='Checkpoint dir name')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU no. (default: 0)')
parser.add_argument('--method', type=str, default='dist', choices=['dist', 'knn', 'ent'],
                    help='')
parser.add_argument('--degree', type=int, default=10,
                    help='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:' + str(args.gpu))

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

# print(dist_mat)
# f = open('mnist_dist_matrix.csv', 'w', newline='')
# wr = csv.writer(f)
# wr.writerows(dist_mat)
# f.close()
import matplotlib.pyplot as plt
from glob import glob


dir_list = glob(args.dir + '*/')
for dat_dir in dir_list:
    dat_list = glob(dat_dir + '*.dat')

    naive_x = [_.split('/')[-1][:-4] for _ in dat_list]
    name_key_map = \
        {'10E_FF_500h_0.1do_1e-5L1_1e-6L2_PT_0.1a': 'Tf_self',
         'BAN_500h_0.1do_1e-5L1_1e-6L2_k0': 'Base',
         'FF_500h_0.99do_skd_0.1do_1e-5L1_1e-6L2_bn_0.005a': 'mDT-KD',
         'FF_500h_5e-4L2_bn': 'L2',
         'FF_500h_5e-5L1_bn': 'L1',
         'FF_500h_0.01LS_bn': 'LS',
         'Tf-reg_500h_0.1do_1e-5L1_1e-6L2': 'Tf_reg',
         'FF_500h_0.1do_1e-5L1_1e-6L2_0.1LS_bn': 'LS*',
         'FF_500h_0.4do_bn': 'Dropout',
         'FF_500h_0.99do_skd_0.1do_1e-5L1_1e-6L2_bn_0.2-a-ada': 'mDT-KD_ada',
         'BAN_500h_0.1do_1e-5L1_1e-6L2_k1': 'BAN',
         'RI_FF_500h_0.1do_1e-5L1_1e-6L2_skd_0.1a': 'RI-KD',
         'SimpleKD_500h_0.1do_1e-5L1_1e-6L2': 'Self-KD',
         'GT': 'GT'}
    x = [name_key_map[_x] for _x in naive_x]
    y_p, y_e = [], []
    for dat_name in dat_list:
        p, e = [], []
        with open(dat_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                _p, _e = line.split()
                _p, _e = float(_p), float(_e)
                # if not math.isnan(_p):
                #     p.append(_p)
                # if not math.isnan(_e):
                #     e.append(_e)
                p.append(_p)
                e.append(_e)
            y_p.append(np.mean(p))
            y_e.append(np.mean(e))
    fig, ax1 = plt.subplots()
    ax1.plot(x, y_p, color='green')

    ax2 = ax1.twinx()
    ax2.plot(x, y_e, color='deeppink')

    # plt.bar(x,y,align='center') # A bar chart
    ax1.set_xlabel('Model')
    ax1.set_xticklabels(x, rotation=45, ha='right')
    ax1.set_ylabel('P(!c)')
    ax2.set_ylabel('Entropy')
    plt.title("/".join(dat_dir.split('/')[1:]))
    # ax1.plot(x, y_p)
    # ax2.plot(x, y_e)
    # plt.xticks(rotation=45, ha='right')
    # plt.ylim(bottom=0.7)
    plt.autoscale(enable=True, axis='y')
    plt.tight_layout()
    # for i in range(len(y)):
    #     plt.hlines(y[i],0,x[i]) # Here you are drawing the horizontal lines
    # plt.show()

    # plt.hist(y_p,
    #          # bins=20, ## 몇 개의 바구니로 구분할 것인가.
    #          # density=True, ## ytick을 퍼센트비율로 표현해줌
    #          # cumulative=False, ## 누적으로 표현하고 싶을 때는 True
    #          # histtype='bar',  ## 타입. or step으로 하면 모양이 바뀜.
    #          # orientation='vertical', ## or horizontal
    #          # rwidth=0.8, ## 1.0일 경우, 꽉 채움 작아질수록 간격이 생김
    #          )
    plt.savefig(dat_dir + 'sensitivity.png')
    # plt.close()
    plt.clf()

# dist_mat = torch.load('toy_data/dist-mat_restricted_uniform_10T.pt')
# print(dist_mat)