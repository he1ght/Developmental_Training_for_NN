from __future__ import print_function

import argparse
import time
import torch

from torchvision import datasets, transforms

from mnist_model import *

start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--limit_size', type=int, default=30000,
                    help='limit MNIST data size')
args = parser.parse_args()

torch.manual_seed(args.seed)

tr_split_len = args.limit_size
tr = datasets.MNIST('./data_mnist', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
part_tr = torch.utils.data.random_split(tr, [tr_split_len, len(tr) - tr_split_len])[0]
torch.save(part_tr, 'mnist_data.pt')

print("--- %s seconds ---" % (time.time() - start_time))
