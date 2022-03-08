from __future__ import print_function

import argparse
import csv
import glob
import time

import torch

from torchvision import datasets, transforms

from evaluater import *
from mnist_model import *

start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=60000, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                    help='SGD momentum (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden', type=int, default=500, metavar='N',
                    help='hidden unit size')
parser.add_argument('--batch-norm', action='store_true', default=False,
                    help='batch_norm')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout rate')
parser.add_argument('--no-limit', default=False, action='store_true',
                    help='No limit MNIST data size')
parser.add_argument('--T', type=int, default=1, metavar='N',
                    help='Temperature')
parser.add_argument('--alpha', type=float, default=0.7, metavar='N',
                    help='Alpha')
parser.add_argument('--checkpoint', type=str, default='model/',
                    help='Checkpoint name')
parser.add_argument('--data', type=str, default='mnist_data.pt',
                    help='Data name')
parser.add_argument('--save', type=str, default='distill',
                    help='Save name')
parser.add_argument('--model', type=str, choices=['ffnn', 'alexnet', 'resnet50', 'wrn'], default='ffnn',
                    help='')
parser.add_argument('--kd_loss', type=str, choices=['kl', 'ce'], default='ce',
                    help='Knowledge distillation loss type. [KL-divergence, Cross Entropy]')
parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='Tensorboard')
parser.add_argument('--tb_dir', type=str, default='tb_log/',
                    help='Tensorboard log dir')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU no. (default: 0)')
parser.add_argument('--l1', type=float, default=0,
                    help='L1 Penalty lambda')
parser.add_argument('--lrate', type=float, default=5e-4,
                    help='L2 Penalty lambda')
parser.add_argument('--ada-alpha', action='store_true', default=False,
                    help='Adaptive alpha')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
for_sKD_diff_test = True

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:' + str(args.gpu))

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


tr = datasets.MNIST('./data_mnist', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
train_loader = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data_mnist', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

num_classes = 10
model_list = glob.glob(args.checkpoint + '*.pt')
# ft = open('mnist_train_result.csv', 'w', newline='')
# wrt = csv.writer(ft)
# wrt.writerow(['', 'train_nll', 'train_te1', 'train_te3', 'train_te5'])
f = open('mnist_test_result.csv', 'w', newline='')
wr = csv.writer(f)
wr.writerow(['', 'test_nll', 'test_te1', 'test_te3', 'test_te5'])
for checkpoint_name in model_list:
    checkpoint = torch.load(checkpoint_name, map_location=lambda storage, loc: storage.cuda(args.gpu))
    model_args = checkpoint['args']
    model = ffn_two_layers(args.hidden, args.dropout, batch_norm=args.batch_norm)

    model.load_state_dict(checkpoint['model'])
    if args.cuda:
        model = model.to(device)

    progress_x_axis = 0

    def train( model, name=''):
        model.eval()
        test_loss = 0
        correct = 0
        test_te1 = 0
        test_te3 = 0
        test_te5 = 0
        test_nll = 0
        with torch.no_grad():
            for data, target in train_loader:
                if args.cuda:
                    data, target = data.to(device), target.to(device)
                output = model(data)
                te = top_error(output, target, (1, 3, 5))
                nl = nll(output, target)
                test_te1 += te[0] * int(data.size(0))
                test_te3 += te[1] * int(data.size(0))
                test_te5 += te[2] * int(data.size(0))
                test_nll += nl * int(data.size(0))
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_te1 /= len(train_loader.dataset)
            test_te3 /= len(train_loader.dataset)
            test_te5 /= len(train_loader.dataset)
            test_nll /= len(train_loader.dataset)
        print('\n{}: Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
              'NLL: {:.4f}, T1: {:.4f}, T3: {:.4f}, T5: {:.4f}'.format(name,
            test_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset),
            test_nll, test_te1, test_te3, test_te5)
        )
        return test_nll, test_te1, test_te3, test_te5


    def test( model, name=''):
        global wr
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
                te = top_error(output, target, (1, 3, 5))
                nl = nll(output, target)
                test_te1 += te[0] * int(data.size(0))
                test_te3 += te[1] * int(data.size(0))
                test_te5 += te[2] * int(data.size(0))
                test_nll += nl * int(data.size(0))
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_te1 /= len(test_loader.dataset)
            test_te3 /= len(test_loader.dataset)
            test_te5 /= len(test_loader.dataset)
            test_nll /= len(test_loader.dataset)
        print('\n{}: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
              'NLL: {:.4f}, T1: {:.4f}, T3: {:.4f}, T5: {:.4f}'.format(name,
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            test_nll, test_te1, test_te3, test_te5)
        )
        return test_nll, test_te1, test_te3, test_te5

    # test_nll, test_te1, test_te3, test_te5 = train(model, name=checkpoint_name)
    # wrt.writerow([checkpoint_name, test_nll, test_te1, test_te3, test_te5])
    test_nll, test_te1, test_te3, test_te5 = test(model, name=checkpoint_name)
    wr.writerow([checkpoint_name, test_nll, test_te1, test_te3, test_te5])
print("--- %s seconds ---" % (time.time() - start_time))
f.close()
