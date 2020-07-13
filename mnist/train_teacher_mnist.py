from __future__ import print_function

import argparse
import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from mnist_model import *
from torch.utils.tensorboard import SummaryWriter

start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--data', type=str, default='mnist_data.pt',
                    help='Data name')
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
parser.add_argument('--save', type=str, default='teacher_MLP',
                    help='Save name')
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

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

if args.no_limit:
    tr = datasets.MNIST('./data_mnist', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
else:
    tr = torch.load(args.data)
train_loader = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data_mnist', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = ffn_two_layers(args.hidden, args.dropout, batch_norm=args.batch_norm)
if args.cuda:
    model = model.to(device)




optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=5e-4)
crit = torch.nn.CrossEntropyLoss()


def train(epoch, model):
    model.train()
    train_loss = 0
    correct = 0
    global draw_graph
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
        if writer is not None and not draw_graph:
            writer.add_graph(model, data)
            draw_graph = True
        optimizer.zero_grad()
        output = model(data)
        loss = crit(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if writer is not None:
            writer.add_scalar('progress/loss', loss.item(),
                              (batch_idx+1)*data.size(0) + (epoch -1 ) * len(train_loader.dataset))
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


def test(epoch, model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if writer is not None:
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', 100. * correct / len(test_loader.dataset), epoch)

for epoch in range(1, args.epochs + 1):
    train(epoch, model)
    test(epoch, model)

writer.close()
save_dict = {'args': args.__dict__, 'model': model.state_dict()}
torch.save(save_dict, args.save + '.pt')
# the_model = Net()
# the_model.load_state_dict(torch.load('teacher_MLP.pth.tar'))

# test(the_model)
# for data, target in test_loader:
#     data, target = Variable(data, volatile=True), Variable(target)
#     teacher_out = the_model(data)
# print(teacher_out)
print("--- %s seconds ---" % (time.time() - start_time))
