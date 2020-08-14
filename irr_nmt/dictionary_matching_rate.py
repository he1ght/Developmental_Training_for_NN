import numpy as np
import torch
from tqdm import tqdm

from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _old_style_vocab


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')


def dict_matching_rate_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add('-dict_file', required=True,
              help="Dictionary file")
    group.add('-src', type=str, required=True,
              help=".")
    group.add('-tgt', type=str, required=True,
              help=".")


def _get_parser():
    parser = ArgumentParser(description='create_embedder.py')

    config_opts(parser)
    dict_matching_rate_opts(parser)
    return parser


def count_sents(directory):
    f = open(directory, "r", encoding="utf8")
    count = 0

    while True:
        line = f.readline()
        if not line:
            break
        count += 1
    f.close()
    return count


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


if __name__ == "__main__":
    parser = _get_parser()
    opt = parser.parse_args()
    enc_vocab, dec_vocab = get_vocabs(opt.dict_file)
    enc_vocab = enc_vocab.stoi.keys()
    dec_vocab = dec_vocab.stoi.keys()
    enc_special_wl = 2
    dec_special_wl = 4

    src_len = count_sents(opt.src)
    tgt_len = count_sents(opt.tgt)

    src_tok_tot, src_tok_cor = 0, 0
    src_word_cor = 0
    src_words = []

    print("== {}: {} sents, {}: {} sents ==".format(opt.src, src_len, opt.tgt, tgt_len))
    with open(opt.src, 'r') as f:
        for _ in enumerate(tqdm(range(src_len))):
            line = f.readline()
            if not line:
                break
            words = line.split()
            src_tok_tot += len(words)
            for w in words:
                if w in enc_vocab:
                    src_tok_cor += 1
                if not (w in src_words):
                    src_words.append(w)
                    if w in enc_vocab:
                        src_word_cor += 1
    src_dict_rate = (src_word_cor + enc_special_wl) / (len(src_words) + enc_special_wl)
    src_tok_rate = src_tok_cor / src_tok_tot
    print(" * src matching rate. Dictionary: {:.2f}% ({}/{}), Token: {:.2f}% ({}/{})".format(src_dict_rate * 100,
                                                                                             src_word_cor +
                                                                                             enc_special_wl,
                                                                                             len(src_words) +
                                                                                             enc_special_wl,
                                                                                             src_tok_rate * 100,
                                                                                             src_tok_cor,
                                                                                             src_tok_tot))

    tgt_tok_tot, tgt_tok_cor = 0, 0
    tgt_word_cor = 0
    tgt_words = []
    with open(opt.tgt, 'r') as f:
        for _ in enumerate(tqdm(range(tgt_len))):
            line = f.readline()
            if not line:
                break
            words = line.split()
            tgt_tok_tot += len(words)
            for w in words:
                if w in dec_vocab:
                    tgt_tok_cor += 1
                if not (w in tgt_words):
                    tgt_words.append(w)
                    if w in dec_vocab:
                        tgt_word_cor += 1
    tgt_dict_rate = (tgt_word_cor + dec_special_wl) / (len(tgt_words) + dec_special_wl)
    tgt_tok_rate = tgt_tok_cor / tgt_tok_tot
    print(" * tgt matching rate. Dictionary: {:.2f}% ({}/{}), Token: {:.2f}% ({}/{})".format(tgt_dict_rate * 100,
                                                                                             tgt_word_cor +
                                                                                             dec_special_wl,
                                                                                             len(tgt_words) +
                                                                                             dec_special_wl,
                                                                                             tgt_tok_rate * 100,
                                                                                             tgt_tok_cor,
                                                                                             tgt_tok_tot))
