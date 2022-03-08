import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class AllForgetLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1,
                 dropout=0, bidirectional=False):
        super(AllForgetLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.dropout = dropout
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dh = nn.Dropout(dropout)
        self.dc = nn.Dropout(dropout)

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for layer in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            if bidirectional:
                self.b_cells.append(nn.LSTMCell(input_size=input_size,
                                                hidden_size=hidden_size))
            input_size = hidden_size * self.num_directions

        self.reset_parameters()

    def reset_parameters(self):
        for i in self.parameters():
            if len(i.shape) > 1:
                nn.init.orthogonal_(i)
            else:
                nn.init.zeros_(i)

    def layer_forward(self, x, hx, cell, reverse=False):
        h, c = hx
        init_h, init_c = h, c
        output, seq_len = [], len(x)
        hs, cs = [], []
        steps = reversed(range(seq_len)) if reverse else range(seq_len)
        # if self.training:
        #     hid_mask = SharedDropout.get_mask(h, self.dropout)
        for t in steps:
            # batch_size = batch_sizes[t]
            # if len(h) < batch_size:
            #     h = torch.cat((h, init_h[last_batch_size:batch_size]))
            #     c = torch.cat((c, init_c[last_batch_size:batch_size]))
            # else:
            #     h = h[:batch_size]
            #     c = c[:batch_size]
            # print(t)
            # print(x[t].size())
            h, c = cell(input=x[t], hx=(h, c))
            output.append(h.unsqueeze(0))
            # if self.training:
            #     h = h * hid_mask[:batch_size]
            # last_batch_size = batch_size
        if reverse:
            output.reverse()
        output = torch.cat(output)
        output = self.dropout(output)

        return output, (h, c)

    def forward(self, x, hx=None):
        # x, batch_sizes = x, length
        batch_size = x.size(1)
        hs, cs = [], []
        hs_, cs_ = [], []

        if hx is None:
            # init = x.new_zeros(batch_size, self.hidden_size)
            init = torch.zeros((batch_size, self.hidden_size)).to(x.device)
            hx = (init, init)

        for layer in range(self.num_layers):
            # if self.training:
            #     mask = SharedDropout.get_mask(x[:batch_size], self.dropout)
            #     mask = torch.cat([mask[:batch_size]
            #                       for batch_size in batch_sizes])
            #     x *= mask
            # x = torch.split(x, batch_sizes.tolist())
            f_output, (h_f, c_f) = self.layer_forward(x=x,
                                                      hx=hx,
                                                      cell=self.f_cells[layer],
                                                      # batch_sizes=batch_sizes,
                                                      reverse=False)

            if self.bidirectional:
                b_output, (h_b, c_b) = self.layer_forward(x=x,
                                                          hx=hx,
                                                          cell=self.b_cells[layer],
                                                          # batch_sizes=batch_sizes,
                                                          reverse=True)
            # if self.bidirectional:
                x = torch.cat([f_output, b_output], -1)
                # hs.append(torch.cat([h_f, h_b], -1))
                # cs.append(torch.cat([c_f, c_b], -1))
            else:
                x = f_output
            hs.append(h_f)
            cs.append(c_f)
            if self.bidirectional:
                hs.append(h_b)
                cs.append(c_b)
            x = self.dropout(x)
        # x = PackedSequence(x, batch_sizes)
        hs = torch.stack(hs)
        cs = torch.stack(cs)
        hs, cs = self.dh(hs), self.dc(cs)
        return x, (hs, cs)


class AllForgetInputFeedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(AllForgetInputFeedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.dh = nn.Dropout(dropout)
        self.dc = nn.Dropout(dropout)

        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        h_1, c_1 = self.dh(h_1), self.dc(c_1)

        return input_feed, (h_1, c_1)
