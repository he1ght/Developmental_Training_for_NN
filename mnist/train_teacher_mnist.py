from __future__ import print_function

import argparse
import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from evaluater import top_error
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
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                    help='SGD momentum (default: 0.)')
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
parser.add_argument('--dropout', type=float, default=0.,
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
parser.add_argument('--lrate', type=float, default=5e-4,
                    help='L2 Penalty lambda')
parser.add_argument('--ls', type=float, default=0.,
                    help='LS epsilono')
parser.add_argument('--l1', type=float, default=0.,
                    help='L1 Penalty lambda')

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
best_acc = 0
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

tr = datasets.MNIST('./data_mnist', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
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
                      weight_decay=args.lrate)

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

if args.ls > 0:
    crit = LabelSmoothingCrossEntropy(epsilon=args.ls)
else:
    crit = torch.nn.CrossEntropyLoss()

progress_x_axis = 0

def train(epoch, model):
    model.train()
    train_loss = 0
    correct = 0
    global draw_graph, progress_x_axis
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


def test(epoch, model):
    model.eval()
    global best_acc
    test_loss = 0
    correct = 0
    test_te1 = 0
    test_te3 = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
        te = top_error(output, target, (1, 3, 5))
        test_te1 += te[0] * int(data.size(0))
        test_te3 += te[1] * int(data.size(0))
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_te1 /= len(test_loader.dataset)
    test_te3 /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if writer is not None:
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', 100. * correct / len(test_loader.dataset), epoch)
        writer.add_scalar('test/Top Error 1', test_te1, epoch)
        writer.add_scalar('test/Top Error 3', test_te3, epoch)
    if best_acc < 100. * correct / len(test_loader.dataset):
        best_acc = 100. * correct / len(test_loader.dataset)
        save_dict = {'args': args.__dict__, 'model': model.state_dict()}
        torch.save(save_dict, args.save + '.pt')

for epoch in range(1, args.epochs + 1):
    train(epoch, model)
    test(epoch, model)
if args.epochs == 0:
    save_dict = {'args': args.__dict__, 'model': model.state_dict()}
    torch.save(save_dict, args.save + '.pt')

if writer is not None:
    writer.close()
# save_dict = {'args': args.__dict__, 'model': model.state_dict()}
# torch.save(save_dict, args.save + '.pt')
# the_model = Net()
# the_model.load_state_dict(torch.load('teacher_MLP.pth.tar'))

# test(the_model)
# for data, target in test_loader:
#     data, target = Variable(data, volatile=True), Variable(target)
#     teacher_out = the_model(data)
# print(teacher_out)
print("--- %s seconds ---" % (time.time() - start_time))
