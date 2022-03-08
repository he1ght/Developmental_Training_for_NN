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
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
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
parser.add_argument('--save', type=str, default='distill',
                    help='Save name')
parser.add_argument('--data', type=str, default='',
                    help='Save name')
parser.add_argument('--cifar', type=int, choices=[10, 100], default=100,
                    help='')
parser.add_argument('--model', type=str, choices=['ffnn', 'alexnet', 'resnet50', 'wrn'], default='ffnn',
                    help='')
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
parser.add_argument('--start-k', type=int, default=0,
                    help='k-ban iteration. (default: 1)')
parser.add_argument('--ban-k', type=int, default=1,
                    help='k-ban iteration. (default: 1)')
parser.add_argument('--ensemble', action='store_true', default=False,
                    help='BANE')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='rate')



def born_again(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1)) * \
               alpha + F.cross_entropy(y, labels) * (1. - alpha)

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
    if args.data:
        tr = torch.load(args.data)
    train_loader = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data_cifar100', train=False, transform=transforms.Compose([
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
    if args.data:
        tr = torch.load(args.data)
    train_loader = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data_cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

num_classes = args.cifar

for k in range(args.start_k, args.ban_k + 1):
    best_acc = 0.0

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

    if k > 0:
        teacher = torch.load(args.save + '_k' + str(k - 1) + '.pt', map_location=lambda storage, loc: storage.cuda(args.gpu))
        model_args = teacher['args']
        if args.model == 'ffnn':
            teacher_model = ffn_two_layers(args.hidden, args.dropout, num_classes=num_classes)
        elif args.model == 'alexnet':
            teacher_model = AlexNet(num_classes=num_classes)
        elif args.model == 'resnet50':
            teacher_model = ResNet50(num_classes=num_classes)
        elif args.model == 'wrn':  # wide resnet
            depth, width = 28, 10
            teacher_model = WideResNet(depth, num_classes, widen_factor=width, dropRate=args.dropout)
        else:
            teacher_model = None
        teacher_model.load_state_dict(teacher['model'])
        if args.cuda:
            teacher_model = teacher_model.to(device)

    if args.cuda:
        model = model.to(device)

    if args.tensorboard:
        writer = SummaryWriter(args.tb_dir + args.save + '_k' + str(k))
    else:
        writer = None

    draw_graph = False
    prev_lr = args.lr

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.lrate)

    progress_x_axis = 0

    def train(k, epoch, model):
        global progress_x_axis, draw_graph
        model.train()
        if k != 0:
            teacher_model.eval()
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
            if k == 0:
                loss = F.cross_entropy(output, target)
            else:
                teacher_output = teacher_model(data)
                teacher_output = teacher_output.detach()
                loss = born_again(output, target, teacher_output, T=1, alpha=args.alpha)
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
            writer.add_scalar('train/lr', prev_lr, epoch)


    def test(k, epoch, model):
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
        if best_acc < 100. * correct / len(test_loader.dataset):
            best_acc = 100. * correct / len(test_loader.dataset)
            save_dict = {'args': args.__dict__, 'model': model.state_dict()}
            torch.save(save_dict, args.save + '_k' + str(k) + '.pt')


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
        train(k, epoch, model)
        test(k, epoch, model)
        print("--- Time taken so far [E {}]: {}".format(epoch, time.time() - start_time))
    writer.close()





print("--- %s seconds ---" % (time.time() - start_time))
