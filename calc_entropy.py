import argparse
import subprocess
from glob import glob
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from cifar10.cifar_model import WideResNet
from scipy.stats import entropy
from math import log, e
import pandas as pd


def cof_arg():
    parser = argparse.ArgumentParser(description='Python Visualization')
    parser.add_argument('-title', type=str, default='', nargs='+',
                        help='')
    parser.add_argument('-xlabel', type=str, default='x-coordinate',
                        help='')
    parser.add_argument('-ylabel', type=str, default='probability',
                        help='')
    parser.add_argument('-draw_margin', type=float, default=0.01,
                        help='')
    parser.add_argument('-interval', type=float, default=0.0025,
                        help='')
    parser.add_argument('-model', type=str, required=True,
                        help='')
    parser.add_argument('-task', type=str, default='cifar10', choices=['mnist', 'cifar10', 'cifar100'],
                        help='')
    parser.add_argument('-test_batch_size', type=int, default=64,
                        help='')
    parser.add_argument('-gpu', type=int, default=-1,
                        help='')
    parser.add_argument('-seed', type=int, default=1,
                        help='')
    parser.add_argument('-done', action='store_true',
                        help='')

    args = parser.parse_args()
    return args


# def entropy(labels, base=None):
#     """ Computes entropy of label distribution. """
#
#     n_labels = len(labels)
#
#     if n_labels <= 1:
#         return 0
#
#     value, counts = np.unique(labels, return_counts=True)
def entropy(probs, base=None):
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


def test(model, num_classes):
    model.eval()
    correct = 0
    no_case = [0 for _ in range(num_classes)]
    total_prod = torch.Tensor([0.0 for _ in range(num_classes)])
    sm = torch.nn.Softmax(dim=-1)
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            if args.cuda:
                data, target = data.to(device), target.to(device)
            output = model(data)
            prob = sm(output)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # print(prob)
            # total_prod.append(prob)
            for i, p in enumerate(pred):
                total_prod[p] += prob[i][p]
                no_case[p] += 1
        # total_prod = torch.cat(total_prod, dim=0).reshape(-1, num_classes)
    for i in range(num_classes):
        total_prod[i] /= no_case[i]
    # print(total_prod.mean(dim=0))
    print(total_prod)
    print(no_case)
    # fig = plt.figure(figsize=(2 * w, w))
    plt_font_size = 24
    plt.rcParams.update({'font.size': plt_font_size})
    linewidth = 2.0
    plt.title(" ".join(args.title))
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    x = [_ for _ in range(num_classes)]
    plt.bar(x, total_prod)
    plt.show()


if __name__ == '__main__':
    args = cof_arg()
    num_classes = 10 if args.task == 'mnist' or args.task == 'cifar10' else 100
    if not args.done:
        args.cuda = args.gpu is not -1 and torch.cuda.is_available()
        for_sKD_diff_test = True

        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            device = torch.device('cuda:' + str(args.gpu))
        else:
            device = torch.device('cpu')

        kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
        if args.task == 'cifar10' or args.task == 'cifar100':
            checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage.cuda(args.gpu))
            model_args = checkpoint['args']
            depth, width = 28, 10
            model = WideResNet(depth, num_classes, widen_factor=width, dropRate=model_args['dropout'])
        else:
            checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage.cuda(args.gpu))
            model_args = checkpoint['args']
            depth, width = 28, 10
            model = WideResNet(depth, num_classes, widen_factor=width, dropRate=model_args['dropout'])
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        if args.task == 'cifar10':
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data_cifar10', train=False, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)
        elif args.task == 'cifar100':
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('./data_cifar100', train=False, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)
        else:
            pass

        test(model, num_classes)
    else:
        observed_result = {
            'cifar10': {
                'base': {
                    'p': [0.9740, 0.9838, 0.9672, 0.9571, 0.9762, 0.9675, 0.9846, 0.9804, 0.9852, 0.9860],
                    'n': [1022, 1020, 989, 977, 1012, 1027, 997, 977, 984, 995]
                },
                'mDT': {
                    'p': [0.9830, 0.9911, 0.9735, 0.9615, 0.9775, 0.9705, 0.9857, 0.9856, 0.9882, 0.9900],
                    'n': [972, 1012, 1005, 953, 992, 1096, 983, 963, 1050, 974]
                },
                'mDT-KD': {
                    'p': [0.9745, 0.9838, 0.9697, 0.9583, 0.9790, 0.9704, 0.9882, 0.9811, 0.9840, 0.9855],
                    'n': [1007, 995, 977, 1000, 995, 1012, 993, 1006, 999, 1016]
                }
            },
            'cifar100': {
                'base': {
                    'p': [0.9740, 0.9838, 0.9672, 0.9571, 0.9762, 0.9675, 0.9846, 0.9804, 0.9852, 0.9860],
                    'n': [1022, 1020, 989, 977, 1012, 1027, 997, 977, 984, 995]
                },
                'mDT': {
                    'p': [0.9386, 0.8926, 0.8324, 0.7543, 0.8144, 0.8407, 0.8821, 0.8575, 0.9295,
                          0.8914, 0.8000, 0.8289, 0.8797, 0.8708, 0.8297, 0.8618, 0.8575, 0.9267,
                          0.8553, 0.8901, 0.9539, 0.8885, 0.9229, 0.8801, 0.9005, 0.8177, 0.7998,
                          0.7949, 0.9006, 0.8111, 0.8214, 0.8575, 0.8446, 0.8275, 0.8353, 0.7920,
                          0.8754, 0.8908, 0.8166, 0.9173, 0.8596, 0.9453, 0.8260, 0.8789, 0.7493,
                          0.7747, 0.7479, 0.8575, 0.9383, 0.9338, 0.7564, 0.8285, 0.8680, 0.9460,
                          0.9357, 0.7358, 0.9193, 0.9250, 0.9238, 0.8517, 0.9530, 0.9213, 0.8922,
                          0.8488, 0.7728, 0.7776, 0.8574, 0.7923, 0.9483, 0.9019, 0.8665, 0.9129,
                          0.7758, 0.8147, 0.8314, 0.9153, 0.9293, 0.8256, 0.8508, 0.8635, 0.8133,
                          0.8911, 0.9358, 0.8608, 0.8669, 0.8905, 0.8996, 0.8982, 0.8856, 0.9068,
                          0.8584, 0.8849, 0.8545, 0.8119, 0.9470, 0.8918, 0.8521, 0.8501, 0.7268,
                          0.8400],
                    'n': [101, 98, 81, 94, 169, 78, 105, 92, 79, 91, 106, 87, 104, 86, 52, 99, 110, 107, 105, 84, 90,
                          111, 73, 135, 108, 86, 97, 111, 84, 78, 114, 85, 87, 176, 111, 135, 88, 86, 89, 102, 74, 92,
                          97, 114, 45, 101, 52, 110, 106, 110, 96, 97, 135, 93, 106, 108, 100, 96, 77, 84, 127, 99, 138,
                          139, 100, 92, 92, 81, 112, 111, 122, 112, 73, 72, 117, 95, 122, 99, 80, 100, 115, 110, 120,
                          87, 95, 78, 86, 107, 130, 97, 112, 113, 79, 86, 120, 131, 116, 91, 53, 124]

                },
                'mDT-KD': {
                    'p': [0.9745, 0.9838, 0.9697, 0.9583, 0.9790, 0.9704, 0.9882, 0.9811, 0.9840, 0.9855],
                    'n': [1007, 995, 977, 1000, 995, 1012, 993, 1006, 999, 1016]
                }
            }
        }
        # cifar10, 0.001L2
        n = observed_result[args.task]['base']['n']
        N = sum(n)
        p = [_ / N for _ in n]
        ent = entropy(p, base=2)
        print(ent)
        n = observed_result[args.task]['mDT']['n']
        p = [_ / N for _ in n]
        ent = entropy(p, base=2)
        print(ent)
        n = observed_result[args.task]['mDT-KD']['n']
        p = [_ / N for _ in n]
        ent = entropy(p, base=2)
        print(ent)