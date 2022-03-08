from __future__ import print_function

import argparse
import time
import torch
import numpy

from torchvision import datasets, transforms

# from cifar10_model import *

start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--limit_size', type=int, default=30000,
                    help='limit CIFAR10 data size')
parser.add_argument('--save', type=str, default='cifar100_data',
                    help='Save data name')
args = parser.parse_args()

torch.manual_seed(args.seed)
numpy.random.seed(args.seed)

tr_split_len = args.limit_size
num_class = 10
each_lengths = []
temp = tr_split_len
part_len = tr_split_len // num_class
if num_class > tr_split_len:
    part_len = 1
for length in range(num_class):
    each_lengths.append(part_len if part_len <= temp else temp)
    temp -= each_lengths[-1]
assert temp == 0
numpy.random.shuffle(each_lengths)

if True:
    print("Data Distribution: {}".format(each_lengths))
    tr = datasets.CIFAR10('./data_cifar10', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))

    classes = []
    for idx in range(num_class):
        classes.append([])
        for i, c in enumerate(tr.targets):
            if idx == int(c):
                classes[idx].append(i)

    selected_index = []
    for class_idx in range(num_class):
        for idx in numpy.random.choice(classes[class_idx], each_lengths[class_idx]):
            selected_index.append(idx)

    part_tr = torch.utils.data.Subset(tr, selected_index)
else:
    tr = datasets.CIFAR10('./data_cifar10', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))
    part_tr = torch.utils.data.random_split(tr, [tr_split_len, len(tr) - tr_split_len])[0]

torch.save(part_tr, args.save + '.pt')

print("--- %s seconds ---" % (time.time() - start_time))
