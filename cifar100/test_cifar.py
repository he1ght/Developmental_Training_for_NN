from __future__ import print_function

import argparse
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from cifar_model import *
from evaluater import * 

start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
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
parser.add_argument('--hidden', type=int, default=1000, metavar='N',
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
parser.add_argument('--checkpoint', type=str, default='teacher_MLP.pt',
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
parser.add_argument('--cifar', type=int, default=10,
                    help='GPU no. (default: 0)')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
for_sKD_diff_test = True

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:' + str(args.gpu))

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
if args.cifar == 100:
    tr = datasets.CIFAR100('./data_cifar100', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))

    train_loader = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data_cifar100', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.cifar == 10:
    tr = datasets.CIFAR10('./data_cifar10', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))

    train_loader = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data_cifar10', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

num_classes = args.cifar

# torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2', pretrained=True)

checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(args.gpu))
# teacher = torch.load(args.teacher, map_location=lambda storage, loc: storage)
model_args = checkpoint['args']
# model_args['gpu'] = args.gpu
if model_args['model'] == 'ffnn':
    model = ffn_two_layers(args.hidden, args.dropout, num_classes=100)
elif model_args['model'] == 'alexnet':
    model = AlexNet(num_classes=num_classes)
elif model_args['model'] == 'resnet50':
    model = ResNet50(num_classes=num_classes)
elif model_args['model'] == 'wrn': # wide resnet
    depth, width = 28, 10
    # teacher_model = Wide_ResNet(depth, width, model_args['dropout'], num_classes=num_classes)
    model = WideResNet(depth, num_classes, widen_factor=width, dropRate=model_args['dropout'])
else:
    model = None

model.load_state_dict(checkpoint['model'])
if args.cuda:
    model = model.to(device)

progress_x_axis = 0
def train( model):
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
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
          'NLL: {:.4f}, T1: {:.4f}, T3: {:.4f}, T5: {:.4f}'.format(
        test_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset),
        test_nll, test_te1, test_te3, test_te5)
    )
def test( model):
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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
          'NLL: {:.4f}, T1: {:.4f}, T3: {:.4f}, T5: {:.4f}'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        test_nll, test_te1, test_te3, test_te5)
    )

# train(model)
test(model)
print("--- %s seconds ---" % (time.time() - start_time))
