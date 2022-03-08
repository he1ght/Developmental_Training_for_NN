from __future__ import print_function

import argparse
import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from cifar_model import *
from evaluater import *

start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--data', type=str, default='cifar100_data.pt',
                    help='Data name')
parser.add_argument('--model', type=str, choices=['ffnn', 'alexnet', 'resnet50', 'wrn'], default='ffnn',
                    help='')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden', type=int, default=1000, metavar='N',
                    help='hidden unit size')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout rate')
parser.add_argument('--label_smoothing', type=float, default=0.,
                    help='Label smoothing epsilon(0: Turn off)')
parser.add_argument('--no-limit', default=False, action='store_true',
                    help='No limit CIFAR100 data size')
parser.add_argument('--save', type=str, default='teacher_model',
                    help='Save name')
parser.add_argument('--l1', type=float, default=0,
                    help='L1 Penalty lambda')
parser.add_argument('--lrate', type=float, default=5e-4,
                    help='L2 Penalty lambda')
parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='Tensorboard')
parser.add_argument('--tb_dir', type=str, default='tb_log/',
                    help='Tensorboard log dir')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU no. (default: 0)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:' + str(args.gpu))
if args.tensorboard:
    writer = SummaryWriter(args.tb_dir + args.save)
else:
    writer = None
draw_graph = False
progress_x_axis = 0

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

if args.no_limit:
    tr = datasets.CIFAR10('./data_cifar10', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))
else:
    tr = torch.load(args.data)
train_loader = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data_cifar10', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

num_classes = 10
if args.model == 'ffnn':
    model = ffn_two_layers(args.hidden, args.dropout, num_classes=num_classes)
elif args.model == 'alexnet':
    model = AlexNet(num_classes=num_classes)
elif args.model == 'resnet50':
    model = ResNet50(num_classes=num_classes)
elif args.model == 'wrn': # wide resnet
    depth, width = 28, 10
    # model = Wide_ResNet(depth, width, args.dropout, num_classes=num_classes)
    model = WideResNet(depth, num_classes, widen_factor=width, dropRate=args.dropout)
else:
    model = None

if args.cuda:
    model = model.to(device)


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.lrate)

if args.label_smoothing > 0:
    crit = LabelSmoothingCrossEntropy(epsilon=args.label_smoothing)
else:
    crit = torch.nn.CrossEntropyLoss()


def train(epoch, model):
    global draw_graph, progress_x_axis
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
        if writer is not None and not draw_graph:
            writer.add_graph(model, data)
            draw_graph = True
        optimizer.zero_grad()
        output = model(data)
        loss = crit(output, target)
        if args.l1 > 0:
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.sum(torch.abs(param))
            loss += l1_loss * args.l1
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if writer is not None:
            progress_x_axis += data.size(0)
            writer.add_scalar('progress/loss', loss.item(), progress_x_axis)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    if writer is not None:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', 100. * correct / len(train_loader.dataset), epoch)


def test(model):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        test_te1 = 0
        test_te3 = 0
        test_te5 = 0
        test_nll = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
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
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        if writer is not None:
            writer.add_scalar('test/loss', test_loss, epoch)
            writer.add_scalar('test/acc', 100. * correct / len(test_loader.dataset), epoch)

            writer.add_scalar('test/Top Error 1', test_te1, epoch)
            writer.add_scalar('test/Top Error 3', test_te3, epoch)
            writer.add_scalar('test/Top Error 5', test_te5, epoch)
            writer.add_scalar('test/NLL loss', test_nll, epoch)


prev_lr = args.lr


def adjust_lr(rate=0.1):
    global prev_lr
    lr = prev_lr * rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    prev_lr = lr


for epoch in range(1, args.epochs + 1):
    if args.model == 'resnet50' and (epoch == 150 or epoch == 225):
        adjust_lr()
    if args.model == 'wrn' and (epoch == 60 or epoch == 120 or epoch == 160):
        adjust_lr(rate=0.2)
    train(epoch, model)
    test(model)
save_dict = {'args': args.__dict__, 'model': model.state_dict()}
torch.save(save_dict, args.save + '.pt')

print("--- %s seconds ---" % (time.time() - start_time))
