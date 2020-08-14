import configargparse as cfargparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from word2vec.data_reader import DataReader, Word2vecDataset
from word2vec.model import SkipGramModel

def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')


def w2v_trainer_opts(parser):
    group = parser.add_argument_group('Model')
    group.add('--input', '-input', type=str, required=True,
              help="")
    group.add('--output', '-output', type=str, default="output",
              help="")
    group.add('--emb_dim', '-emb_dim', type=int, default=100,
              help="")
    group.add('--batch_size', '-batch_size', type=int, default=32,
              help="")
    group.add('--window_size', '-window_size', type=int, default=5,
              help="")
    group.add('--epochs', '-epochs', type=int, default=3,
              help="")
    group.add('--lr', '-lr', type=float, default=0.001,
              help="")
    group.add('--min_cnt', '-min_cnt', type=int, default=1,
              help="")
    group.add('--device', '-device', type=int, default=0,
              help="")
    group.add('--dict_file', '-dict_file', type=str, default="",
              help="")


def _get_parser():
    parser = cfargparse.ArgParser()

    config_opts(parser)
    w2v_trainer_opts(parser)
    return parser

class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, epochs=3,
                 initial_lr=0.001, min_count=12, vocab=None, device=0):

        self.data = DataReader(input_file, min_count, vocab=vocab)
        dataset = Word2vecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(device) if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):

        for epoch in range(self.epochs):

            print("\n\n\nIteration: " + str(epoch + 1))
            # optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    # if i > 0 and i % 500 == 0:
                    #     print(" [Epoch {}, Iter {}/{}] Loss: ".format(epoch + 1, i + 1, 0) + str(running_loss))
            print(" [Epoch {}] Loss: ".format(epoch + 1) + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name + '_' + str(epoch + 1) + ".vec")


if __name__ == '__main__':
    parser = _get_parser()

    opt = parser.parse_args()

    vocab = None
    if opt.dict_file:
        with open(opt.dict_file, 'r') as f:
            line = f.readline()
            vocab = line.split()
    w2v = Word2VecTrainer(input_file=opt.input, output_file=opt.output,
                          emb_dimension=opt.emb_dim, batch_size=opt.batch_size,
                          window_size=opt.window_size, epochs=opt.epochs,
                          initial_lr=opt.lr, min_count=opt.min_cnt, vocab=vocab, device=opt.device)
    w2v.train()
