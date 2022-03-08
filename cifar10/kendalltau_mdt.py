from __future__ import print_function

import argparse
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from cifar_model import *
from evaluater import * 
import scipy.stats as stats
start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--mdt', type=str, default='teacher_MLP.pt',
                    help='Checkpoint name')
parser.add_argument('--model', type=str, default='teacher_MLP.pt',
                    help='Checkpoint name')
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU no. (default: 0)')
parser.add_argument('--seed', type=int, default=1,
                    help='GPU no. (default: 0)')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda and args.gpu != -1:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:' + str(args.gpu))
else:
    device = torch.device('cpu')

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data_cifar100', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])),
    batch_size=args.batch_size, shuffle=False, **kwargs)

num_classes = 10

# torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2', pretrained=True)

mdt_ckp = torch.load(args.mdt, map_location=lambda storage, loc: storage.cuda(args.gpu))
checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage.cuda(args.gpu))
model_args = checkpoint['args']
mdt_args = mdt_ckp['args']
# model_args['gpu'] = args.gpu
if model_args['model'] == 'wrn': # wide resnet
    depth, width = 28, 10
    mdt = WideResNet(depth, num_classes, widen_factor=width, dropRate=mdt_args['dropout'])
    model = WideResNet(depth, num_classes, widen_factor=width, dropRate=model_args['dropout'])
else:
    model, mdt = None, None

mdt.load_state_dict(mdt_ckp['model'])
model.load_state_dict(checkpoint['model'])
if args.cuda:
    mdt = mdt.to(device)
    model = model.to(device)

progress_x_axis = 0

def test(mdt, model):
    mdt.eval()
    model.eval()
    taus, pv = [], []
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.to(device), target.to(device)
            p1 = mdt(data)
            p2 = model(data)
            # print('P: ', p1, p2)
            r1 = torch.argsort(p1, dim=1)
            r2 = torch.argsort(p2, dim=1)
            # print('r: ', r1, r2)
            for x1, x2 in zip(r1.data.numpy(), r2.data.numpy()):
                tau, p_value = stats.kendalltau(x1, x2)
                taus.append(tau)
                pv.append(p_value)
                # print(x1, x2, tau)
    n = len(pv)
    result = np.mean(taus)
    print(result)

test(mdt, model)
