from __future__ import print_function

import argparse
import time
import torch
import numpy

from torchvision import datasets, transforms

from mnist_model import *

start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--limit_size', type=int, default=30000,
                    help='limit MNIST data size')
parser.add_argument('--one_shot', action='store_true',
                    help='Make only one data per class')
parser.add_argument('--save', type=str, default='mnist_data',
                    help='Save data name')
args = parser.parse_args()

torch.manual_seed(args.seed)
numpy.random.seed(args.seed)

if args.one_shot:
    tr = datasets.MNIST('./data_mnist', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))

    classes = []
    for idx in range(10):
        classes.append([])
        for i, c in enumerate(tr.targets):
            if idx == int(c):
                classes[idx].append(i)
    selected_index = [int(numpy.random.choice(classes[_], 1)) for _ in range(10)]

    part_tr = torch.utils.data.Subset(tr, selected_index)
else:
    tr_split_len = args.limit_size
    tr = datasets.MNIST('./data_mnist', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    part_tr = torch.utils.data.random_split(tr, [tr_split_len, len(tr) - tr_split_len])[0]
    print(part_tr)
    print(part_tr.dataset)
torch.save(part_tr, args.save + '.pt')

print("--- %s seconds ---" % (time.time() - start_time))
