import configargparse as cfargparse
import torch

from onmt.inputters.inputter import _old_style_vocab


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
    group.add('--output', '-output', type=str, default="vocab",
              help="")


def _get_parser():
    parser = cfargparse.ArgParser()

    config_opts(parser)
    w2v_trainer_opts(parser)
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



if __name__ == '__main__':
    parser = _get_parser()

    opt = parser.parse_args()

    enc_vocab, dec_vocab = get_vocabs(opt.dict_file)
    enc_vocab = enc_vocab.stoi.keys()
    dec_vocab = dec_vocab.stoi.keys()

    # Encoder
    with open(opt.output + "_enc.txt", 'w') as f:
        sent = " ".join(enc_vocab)
        f.write(sent)

    # Decoder
    with open(opt.output + "_dec.txt", 'w') as f:
        sent = " ".join(dec_vocab)
        f.write(sent)
