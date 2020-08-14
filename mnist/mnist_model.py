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
