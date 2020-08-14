import configargparse as cfargparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from onmt.inputters.inputter import _old_style_vocab
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
    group.add('--dict_file', '-dict_file', type=str, required=True,
              help="")
    group.add('--src', '-src', type=str, required=True,
              help="")
    group.add('--tgt', '-tgt', type=str, required=True,
              help="")
    group.add('--output_src', '-output_src', type=str, default="output_src.txt",
              help="")
    group.add('--output_tgt', '-output_tgt', type=str, default="output_tgt.txt",
              help="")
    group.add('--pad_size', '-pad_size', type=int, default=30,
              help="")


def _get_parser():
    parser = cfargparse.ArgParser()

    config_opts(parser)
    w2v_trainer_opts(parser)
    return parser

ENCODER_OPTS = ['<unk>', '<blank>']
DECODER_OPTS = ['<unk>', '<blank>', '<s>', '</s>']

def get_vocabs(dict_path):
    fields = torch.load(dict_path)

    vocs = []
    for side in ['src', 'tgt']:
        if _old_style_vocab(fields):
            vocab = next((v for n, v in fields if n == side), None)
        else:
            try:
                vocab = fields[side].base_field.vocab
            except AttributeError:
                vocab = fields[side].vocab
        vocs.append(vocab)
    enc_vocab, dec_vocab = vocs
    return enc_vocab, dec_vocab

if __name__ == '__main__':
    parser = _get_parser()

    opt = parser.parse_args()

    enc_vocab, dec_vocab = get_vocabs(opt.dict_file)
    enc_vocab = enc_vocab.stoi.keys()
    dec_vocab = dec_vocab.stoi.keys()

    # Encoder
    i = 0
    with open(opt.src, 'r') as f:
        with open(opt.output_src, 'w') as wf:
            sents = []
            for line in tqdm(f.readlines()):
                words = line.split()
                new_words = []
                # new_words.append('<s>')
                for word in words:
                    if not ( word in enc_vocab ):
                        word = '<unk>'
                    new_words.append(word)
                # new_words.append('</s>')
                if len(words) < opt.pad_size:
                    for _ in range(opt.pad_size - len(words)):
                        new_words.append('<blank>')
                sents.append(" ".join(new_words))
                i += 1
                if i > 1000:
                    sent = "\n".join(sents) + "\n"
                    wf.write(sent)
                    sents = []
                    i = 0
            if i != 0:
                sent = "\n".join(sents)
                wf.write(sent)

    # Decoder
    i = 0
    with open(opt.tgt, 'r') as f:
        with open(opt.output_tgt, 'w') as wf:
            sents = []
            for line in tqdm(f.readlines()):
                words = line.split()
                new_words = []
                new_words.append('<s>')
                for word in words:
                    if not (word in dec_vocab):
                        word = '<unk>'
                    new_words.append(word)
                new_words.append('</s>')
                if len(words) < opt.pad_size:
                    for _ in range(opt.pad_size - len(words)):
                        new_words.append('<blank>')
                sents.append(" ".join(new_words))
                i += 1
                if i > 1000:
                    sent = "\n".join(sents) + "\n"
                    wf.write(sent)
                    sents = []
                    i = 0
            if i != 0:
                sent = "\n".join(sents)
                wf.write(sent)