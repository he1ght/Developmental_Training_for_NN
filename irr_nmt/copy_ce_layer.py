#!/usr/bin/env python
"""Train models."""
from onmt.utils import Optimizer


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')


def copy_ce_layer_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')
    group.add('--train_from', '-train_from', default='', type=str, required=True,
              help="If training from a checkpoint then this is the "
                   "path to the pretrained model's state_dict.")
    group.add('--ce_model', '-ce_model', default='', type=str, required=True,
              help="If training from a checkpoint then this is the "
                   "path to the pretrained model's state_dict.")
    group.add('--save_model', '-save_model', default='model',
              help="Model filename (the model will be saved as "
                   "<save_model>_N.pt where N is the number "
                   "of steps")
    group.add('--reset_optim', '-reset_optim', default='none',
              choices=['none', 'all', 'states', 'keep_states'],
              help="Optimization resetter when train_from.")
    group.add('--model_dtype', '-model_dtype', default='fp32',
              choices=['fp32', 'fp16'],
              help='Data type of the model.')
    group.add('--keep_checkpoint', '-keep_checkpoint', type=int, default=-1,
              help="Keep X checkpoints (negative: keep all)")


# !/usr/bin/env python
"""Training on a single process."""
import os

import torch

from onmt.inputters.inputter import load_old_vocab, old_style_vocab
from onmt.model_builder import build_model
from onmt.models import build_model_saver
from onmt.utils.parse import ArgumentParser


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def copy_ce_layer(opt):
    checkpoint = torch.load(opt.train_from,
                            map_location=lambda storage, loc: storage)
    model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)

    checkpoint_ce = torch.load(opt.ce_model,
                            map_location=lambda storage, loc: storage)
    model_opt_ce = ArgumentParser.ckpt_model_opts(checkpoint_ce["opt"])
    ArgumentParser.update_model_opts(model_opt_ce)
    ArgumentParser.validate_model_opts(model_opt_ce)

    vocab = checkpoint['vocab']

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, 'text', dynamic_dict=False)
    else:
        fields = vocab

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    model_ce = build_model(model_opt, opt, fields, checkpoint_ce)

    model_opt.concept_equalization = model_opt_ce.concept_equalization
    model.ce_layer = model_ce.ce_layer
    _check_save_model_path(opt)

    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    model_saver.save(step=optim._training_step - 1)


def _get_parser():
    parser = ArgumentParser(description='copy_ce_layer_opts.py')

    config_opts(parser)
    copy_ce_layer_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    copy_ce_layer(opt)


if __name__ == "__main__":
    main()
