from __future__ import print_function

import argparse
import copy
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from mnist_model import *
from evaluater import * 

start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--cifar', type=int, choices=[10, 100], default=100,
                    help='')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU no. (default: 0)')
parser.add_argument('--ban-k', type=int, default=3,
                    help='k-ban iteration. (default: 1)')
parser.add_argument('--model', type=str, default='model/BANs',
                    help='k-ban iteration. (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
for_sKD_diff_test = True

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:' + str(args.gpu))
else:
    device = torch.device('cpu')

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data_mnist', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

num_classes = 10

models = []
for k in range(0, args.ban_k + 1):
# if __name__ == '__main__':
    best_acc = 0.0
    if args.cuda:
        checkpoint = torch.load(args.model + '_k' + str(k) + '.pt',
                         map_location=lambda storage, loc: storage.cuda(args.gpu))
    else:
        checkpoint = torch.load(args.model + '_k' + str(k) + '.pt',
                            map_location=lambda storage, loc: storage.cpu())
    model_args = checkpoint['args']
    model = ffn_two_layers(model_args['hidden'], model_args['dropout'])
    model.load_state_dict(checkpoint['model'])

    if args.cuda:
        model = model.to(device)

    def test(k, model):
        global best_acc
        model.eval()
        test_loss = 0
        correct = 0
        test_te1 = 0
        test_te3 = 0
        test_te5 = 0
        test_nll = 0
        with torch.no_grad():
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target)
                te = top_error(output, target, (1, 3, 5))
                nl = nll(output, target)
                test_te1 += te[0] * int(data.size(0))
                test_te3 += te[1] * int(data.size(0))
                test_te5 += te[2] * int(data.size(0))
                test_nll += nl * int(data.size(0))
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            test_loss /= len(test_loader.dataset)
            test_te1 /= len(test_loader.dataset)
            test_te3 /= len(test_loader.dataset)
            test_te5 /= len(test_loader.dataset)
            test_nll /= len(test_loader.dataset)
            print('BANs [K: {}] Test set: T1: {:.4f} T3: {:.4f} T5: {:.4f} Accuracy: {}/{} ({:.2f}%)'.format(
                k, test_te1, test_te3, test_te5, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))


    test(k, model)
    if k>0:
        models.append(copy.deepcopy(model))

    # print("--- Time taken so far : {}".format(time.time() - start_time))

for k in range(2, args.ban_k + 1):

    def ensemble(k):
        test_loss = 0
        correct = 0
        test_te1 = 0
        test_te3 = 0
        test_te5 = 0
        test_nll = 0
        with torch.no_grad():
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.to(device), target.to(device)
                output = None
                for model in models[:k]:
                    if output is None:
                        output = model(data)
                    else:
                        output += model(data)
                test_loss += F.cross_entropy(output, target)
                te = top_error(output, target, (1, 3, 5))
                nl = nll(output, target)
                test_te1 += te[0] * int(data.size(0))
                test_te3 += te[1] * int(data.size(0))
                test_te5 += te[2] * int(data.size(0))
                test_nll += nl * int(data.size(0))
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            test_loss /= len(test_loader.dataset)
            test_te1 /= len(test_loader.dataset)
            test_te3 /= len(test_loader.dataset)
            test_te5 /= len(test_loader.dataset)
            test_nll /= len(test_loader.dataset)
            print('BANE [K: {}] Test set: T1: {:.4f} T3: {:.4f} T5: {:.4f} Accuracy: {}/{} ({:.2f}%)'.format(
                k, test_te1, test_te3, test_te5, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    ensemble(k)

# print("--- %s seconds ---" % (time.time() - start_time))
