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
parser.add_argument('--no-limit', default=False, action='store_true',
                    help='No limit MNIST data size')
parser.add_argument('--T', type=int, default=1, metavar='N',
                    help='Temperature')
parser.add_argument('--alpha', type=float, default=0.7, metavar='N',
                    help='Alpha')
parser.add_argument('--beta', type=float, default=0.3, metavar='N',
                    help='Beta')
parser.add_argument('--teacher_1', type=str, default='teacher_MLP.pt',
                    help='Checkpoint name')
parser.add_argument('--teacher_2', type=str, default='teacher_MLP.pt',
                    help='Checkpoint name')
parser.add_argument('--teacher_3', type=str, default='teacher_MLP.pt',
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
parser.add_argument('--lrate', type=float, default=5e-4,
                    help='L2 Penalty lambda')
parser.add_argument('--ada-alpha', action='store_true', default=False,
                    help='Adaptive alpha')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
for_sKD_diff_test = True

assert args.alpha + args.beta <= 1

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:' + str(args.gpu))

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
if args.no_limit:
    tr = datasets.CIFAR100('./data_cifar100', train=True, download=True,
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
    datasets.CIFAR100('./data_cifar100', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
num_classes = 100
if args.model == 'ffnn':
    model = ffn_two_layers(args.hidden, args.dropout, num_classes=100)
elif args.model == 'alexnet':
    model = AlexNet(num_classes=100)
elif args.model == 'resnet50':
    model = ResNet50(num_classes=100)
elif args.model == 'wrn':  # wide resnet
    depth, width = 28, 10
    model = Wide_ResNet(depth, width, args.dropout, num_classes=num_classes)
else:
    model = None
teacher_1 = torch.load(args.teacher_1, map_location=lambda storage, loc: storage.cuda(args.gpu))
model_args_1 = teacher_1['args']
# model_args['gpu'] = args.gpu
if model_args_1['model'] == 'ffnn':
    teacher_model_1 = ffn_two_layers(args.hidden, args.dropout, num_classes=100)
elif model_args_1['model'] == 'alexnet':
    teacher_model_1 = AlexNet(num_classes=100)
elif model_args_1['model'] == 'resnet50':
    teacher_model_1 = ResNet50(num_classes=100)
elif model_args_1['model'] == 'wrn':  # wide resnet
    depth, width = 28, 10
    teacher_model_1 = Wide_ResNet(depth, width, model_args_1['dropout'], num_classes=num_classes)
else:
    teacher_model_1 = None
teacher_model_1.load_state_dict(teacher_1['model'])

teacher_2 = torch.load(args.teacher_2, map_location=lambda storage, loc: storage.cuda(args.gpu))
model_args_2 = teacher_2['args']
if model_args_2['model'] == 'ffnn':
    teacher_model_2 = ffn_two_layers(model_args_2.hidden, model_args_2.dropout, num_classes=100)
elif model_args_2['model'] == 'alexnet':
    teacher_model_2 = AlexNet(num_classes=100)
elif model_args_2['model'] == 'resnet50':
    teacher_model_2 = ResNet50(num_classes=100)
elif model_args_2['model'] == 'wrn':  # wide resnet
    depth, width = 28, 10
    teacher_model_2 = Wide_ResNet(depth, width, model_args_2['dropout'], num_classes=num_classes)
else:
    teacher_model_2 = None
teacher_model_2.load_state_dict(teacher_2['model'])

teacher_3 = torch.load(args.teacher_3, map_location=lambda storage, loc: storage.cuda(args.gpu))
model_args_3 = teacher_2['args']
if model_args_3['model'] == 'ffnn':
    teacher_model_3 = ffn_two_layers(model_args_3.hidden, model_args_3.dropout, num_classes=100)
elif model_args_3['model'] == 'alexnet':
    teacher_model_3 = AlexNet(num_classes=100)
elif model_args_3['model'] == 'resnet50':
    teacher_model_3 = ResNet50(num_classes=100)
elif model_args_3['model'] == 'wrn':  # wide resnet
    depth, width = 28, 10
    teacher_model_3 = Wide_ResNet(depth, width, model_args_3['dropout'], num_classes=num_classes)
else:
    teacher_model_3 = None
teacher_model_3.load_state_dict(teacher_3['model'])

if args.cuda:
    model = model.to(device)
    teacher_model_1 = teacher_model_1.to(device)
    teacher_model_2 = teacher_model_2.to(device)
    teacher_model_3 = teacher_model_3.to(device)

if args.tensorboard:
    writer = SummaryWriter(args.tb_dir + args.save)
else:
    writer = None

draw_graph = False
prev_lr = args.lr

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.lrate)


def softmax_cross_entropy_with_softtarget(input, target, reduction='mean', T=1):
    """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
    logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1) / T, dim=1)
    batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')


def distillation(y, labels, teacher_scores, loss_type, T, alpha):
    if loss_type == 'ce':
        return softmax_cross_entropy_with_softtarget(y, F.softmax(teacher_scores / T, dim=1)) * (
                T * T * alpha) + F.cross_entropy(y, labels) * (1. - alpha)
    elif loss_type == 'kl':
        return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1)) * \
               (T * T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)
    else:
        return None


def double_distillation(y, labels, teacher_scores_1, teacher_scores_2, T, alpha, beta):
    return softmax_cross_entropy_with_softtarget(y, F.softmax(teacher_scores_1 / T, dim=1)) * (T * T * alpha) \
           + softmax_cross_entropy_with_softtarget(y, F.softmax(teacher_scores_2 / T, dim=1)) * (T * T * beta) \
           + F.cross_entropy(y, labels) * (1. - (alpha + beta))


def smart_distillation(y, labels, teacher_scores_1, teacher_scores_2, teacher_scores_3, T, alpha, beta):
    return softmax_cross_entropy_with_softtarget(y, F.softmax(teacher_scores_1 / T, dim=1)) * (T * T * alpha) \
           + softmax_cross_entropy_with_softtarget(y, F.softmax(teacher_scores_2 / T, dim=1)) * (T * T * beta) \
           + softmax_cross_entropy_with_softtarget(y, F.softmax(teacher_scores_3 / T, dim=1)) * (T * T * beta) \
           + F.cross_entropy(y, labels) * (1. - (alpha + beta * 2))


progress_x_axis = 0


def train(epoch, model, loss_fn):
    model.train()
    teacher_model_1.eval()
    teacher_model_2.eval()
    train_loss = 0
    correct = 0
    if args.ada_alpha:
        alpha_t = args.alpha * (1 - (epoch - 1) / args.epochs)
        beta_t = args.beta * (1 - (epoch - 1) / args.epochs)
    else:
        alpha_t = args.alpha
        beta_t = args.beta
    global draw_graph, progress_x_axis, prev_lr
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)

        if writer is not None and not draw_graph:
            writer.add_graph(model, data)
            draw_graph = True
        optimizer.zero_grad()
        output = model(data)
        teacher_output_1 = teacher_model_1(data)
        teacher_output_1 = teacher_output_1.detach()
        teacher_output_2 = teacher_model_2(data)
        teacher_output_2 = teacher_output_2.detach()
        teacher_output_3 = teacher_model_3(data)
        teacher_output_3 = teacher_output_3.detach()

        loss = loss_fn(output, target, teacher_output_1, teacher_output_2, teacher_output_3, T=args.T, alpha=alpha_t, beta=beta_t)
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
        writer.add_scalar('train/alpha', alpha_t, epoch)


def test(epoch, model, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    test_te1 = 0
    test_te5 = 0
    test_nll = 0
    with torch.no_grad():
        if args.ada_alpha:
            alpha_t = args.alpha * (1 - (epoch - 1) / args.epochs)
            beta_t = args.beta * (1 - (epoch - 1) / args.epochs)
        else:
            alpha_t = args.alpha
            beta_t = args.beta
        for data, target in test_loader:
            if args.cuda:
                data, target = data.to(device), target.to(device)
            output = model(data)
            teacher_output_1 = teacher_model_1(data)
            teacher_output_1 = teacher_output_1.detach()
            teacher_output_2 = teacher_model_2(data)
            teacher_output_2 = teacher_output_2.detach()
            teacher_output_3 = teacher_model_3(data)
            teacher_output_3 = teacher_output_3.detach()
            test_loss += loss_fn(output, target, teacher_output_1, teacher_output_2, teacher_output_3, T=args.T,
                                 alpha=alpha_t, beta=beta_t).detach()
            te = top_error(output, target, (1, 5))
            nl = nll(output, target)
            test_te1 += te[0] * int(data.size(0))
            test_te5 += te[1] * int(data.size(0))
            test_nll += nl * int(data.size(0))
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        test_te1 /= len(test_loader.dataset)
        test_te5 /= len(test_loader.dataset)
        test_nll /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        if writer is not None:
            writer.add_scalar('test/loss', test_loss, epoch)
            writer.add_scalar('test/acc', 100. * correct / len(test_loader.dataset), epoch)

            writer.add_scalar('test/Top Error 1', test_te1, epoch)
            writer.add_scalar('test/Top Error 5', test_te5, epoch)
            writer.add_scalar('test/NLL loss', test_nll, epoch)


def adjust_lr(rate=0.1):
    global prev_lr
    lr = prev_lr * rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    prev_lr = lr


for epoch in range(1, args.epochs + 1):
    if args.model == 'resnet50' and (epoch == 150 or epoch == 225):
        adjust_lr()
    # if args.model == 'wrn' and (epoch == 100 or epoch == 160 or epoch == 230):
    #     adjust_lr(rate=0.2)
    train(epoch, model, loss_fn=smart_distillation)
    test(epoch, model, loss_fn=smart_distillation)

writer.close()
save_dict = {'args': args.__dict__, 'model': model.state_dict()}
torch.save(save_dict, args.save + '.pt')
# the_model = Net()
# the_model.load_state_dict(torch.load('student.pth.tar'))

# test(the_model)
# for data, target in test_loader:
#     data, target = Variable(data, volatile=True), Variable(target)
#     teacher_out = the_model(data)
# print(teacher_out)
print("--- %s seconds ---" % (time.time() - start_time))
