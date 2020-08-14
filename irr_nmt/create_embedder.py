import numpy as np
import torch

from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _old_style_vocab


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')


def extractor_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add('-dict_file', required=True,
                        help="Dictionary file")
    group.add('-vec_size', type=int, default=512,
                        help=".")
    group.add('-type', default='random',
              choices=['random', 'bert'],
              help=".")
    group.add('-seed', type=int, default=0,
              help=".")
    group.add('-vec_len', type=float, default=0.1,
              help=".")


def _get_parser():
    parser = ArgumentParser(description='create_embedder.py')

    config_opts(parser)
    extractor_opts(parser)
    return parser


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


def random_three_vector(size):
    phi = np.random.uniform(0, np.pi * 2, size=size)
    costheta = np.random.uniform(-1, 1, size=size)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return (x, y, z)


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    enc_vocab, dec_vocab = get_vocabs(opt.dict_file)
    enc_vocab = enc_vocab.stoi.keys()
    dec_vocab = dec_vocab.stoi.keys()

    enc_file_name = 'random_encoder_emb.txt' if opt.type == 'random' else 'bert_encoder_emb.txt'
    dec_file_name = 'random_decoder_emb.txt' if opt.type == 'random' else 'bert_decoder_emb.txt'

    np.random.seed(opt.seed)
    vecs = np.random.uniform(-opt.vec_len, opt.vec_len, size=(len(enc_vocab), opt.vec_size))
    print("Encoder Vectors generated ... ")

    with open(enc_file_name, 'w') as f:
        for word, v in zip(enc_vocab, vecs):
            line = ""
            line += word
            for e in v:
                line += " " + str(e)
            line += "\n"
            f.write(line)

    vecs = np.random.uniform(-opt.vec_len, opt.vec_len, size=(len(dec_vocab), opt.vec_size))
    print("Decoder Vectors generated ... ")
    with open(dec_file_name, 'w') as f:
        for word, v in zip(dec_vocab, vecs):
            line = ""
            line += word
            for e in v:
                line += " " + str(e)
            line += "\n"
            f.write(line)

