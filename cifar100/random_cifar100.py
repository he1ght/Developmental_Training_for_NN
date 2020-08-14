from __future__ import print_function

import argparse
import time
import torch
import numpy

from torchvision import datasets, transforms

# from cifar10_model import *

start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Example')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--limit_size', type=int, default=30000,
                    help='limit CIFAR10 data size')
parser.add_argument('--one_shot', action='store_true',
                    help='Make only one data per class')
parser.add_argument('--save', type=str, default='cifar100_data',
                    help='Save data name')
args = parser.parse_args()

torch.manual_seed(args.seed)
numpy.random.seed(args.seed)

tr_split_len = args.limit_size
tr = datasets.CIFAR100('./data_cifar100', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]))
part_tr = torch.utils.data.random_split(tr, [tr_split_len, len(tr) - tr_split_len])[0]

torch.save(part_tr, args.save + '.pt')

print("--- %s seconds ---" % (time.time() - start_time))
