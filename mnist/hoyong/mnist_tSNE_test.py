import torch
import torch.nn as nn
import torch.nn.functional as F


class ffn_two_layers(nn.Module):
    def __init__(self, hidden=500, dropout=0.1, batch_norm=False):
        super(ffn_two_layers, self).__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(self.hidden)
            self.bn2 = nn.BatchNorm1d(self.hidden)

        self.fc1 = nn.Linear(28 * 28, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        return x

class ToyUniformDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        self.x_data = x
        self.y_data = [0 for _ in range(x.size(0))] # Y는 Dummy label.

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, input):
        x = input / 255.0
        # print(x)
        x = x - self.mean
        x = x / self.std
        return x

file_nickname_map = \
        {'10E_FF_500h_0.1do_1e-5L1_1e-6L2_PT_0.1a': 'Tf_self',
         'BAN_500h_0.1do_1e-5L1_1e-6L2_k0': 'Base',
         'FF_500h_0.99do_skd_0.1do_1e-5L1_1e-6L2_bn_0.005a': 'mDT-KD',
         'FF_500h_5e-4L2_bn': 'L2',
         'FF_500h_5e-5L1_bn': 'L1',
         'FF_500h_0.01LS_bn': 'LS',
         'Tf-reg_500h_0.1do_1e-5L1_1e-6L2': 'Tf_reg',
         'FF_500h_0.1do_1e-5L1_1e-6L2_0.1LS_bn': 'LS*',
         'FF_500h_0.4do_bn': 'Dropout',
         'FF_500h_0.99do_skd_0.1do_1e-5L1_1e-6L2_bn_0.2-a-ada': 'mDT-KD_ada',
         'BAN_500h_0.1do_1e-5L1_1e-6L2_k1': 'BAN',
         'RI_FF_500h_0.1do_1e-5L1_1e-6L2_skd_0.1a': 'RI-KD',
         'SimpleKD_500h_0.1do_1e-5L1_1e-6L2': 'Self-KD',
         'GT': 'GT'}

if __name__ == '__main__':
    # model load
    model = torch.load('demo.pt') # class: ffn_two_layers

    # data load
    if args.data:
        fit_dataset = torch.load(args.data)
    else:
        fit_dataset = datasets.MNIST('./data_mnist', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    data_loader = torch.utils.data.DataLoader(
        fit_dataset,
        batch_size=10000000, shuffle=False, **kwargs)