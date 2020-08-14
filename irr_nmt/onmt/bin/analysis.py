#!/usr/bin/env python
"""Train models."""
import traceback

import torch

import onmt
import onmt.opts as opts
from onmt.inputters.inputter import build_dataset_iter, \
    build_dataset_iter_multiple
from onmt.model_builder import load_test_model
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import set_random_seed
from onmt.utils.parse import ArgumentParser


def analysis(opt):
    # ArgumentParser.validate_train_opts(opt)
    # ArgumentParser.update_model_opts(opt)
    # ArgumentParser.validate_model_opts(opt)

    opt.gpu_ranks = [opt.gpu] if opt.gpu != -1 else []

    if opt.gpu != -1 and torch.cuda.is_available():  # case 1 GPU only
        run_single(opt, opt.gpu)
    else:  # case only CPU
        run_single(opt, -1)


def _get_parser():
    parser = ArgumentParser(description='analysis.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.analysis_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    analysis(opt)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def run_single(opt, device_id, batch_queue=None, semaphore=None):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)

    # Build model.
    logger.info('Loading model from %s' % opt.model)
    fields, model, model_opt = load_test_model(opt, opt.model)
    print(model)
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    logger.info('Loading vocab from model at %s.' % opt.model)

    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)

    steper = Steper()

    trainer = build_trainer(
        opt, device_id, model, fields, steper)

    if len(opt.data_ids) > 1:
        train_shards = []
        for train_id in opt.data_ids:
            shard_base = "train_" + train_id
            train_shards.append(shard_base)
        train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
    else:
        if opt.data_ids[0] is not None:
            shard_base = "train_" + opt.data_ids[0]
        else:
            shard_base = "train"
        train_iter = build_dataset_iter(shard_base, fields, opt)

    if opt.gpu != -1:
        logger.info('Starting training on GPU: %s' % opt.gpu)
    else:
        logger.info('Starting training on CPU, could be very slow')

    train_steps = 0

    trainer.train(
        train_iter,
        train_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()


def build_trainer(opt, device_id, model, fields, steper, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        steper (:obj:`onmt.utils.steperizer`): steperizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    tgt_field = dict(fields)["tgt"].base_field
    train_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt, do_backward=False)

    if device_id >= 0:
        n_gpu = 1
        gpu_rank = device_id
    else:
        gpu_rank = 0
        n_gpu = 0


    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = Trainer(model, train_loss, steper,
                      n_gpu=n_gpu, gpu_rank=gpu_rank, report_manager=report_manager,
                      with_align=True if opt.lambda_align > 0 else False)
    return trainer


class Steper(object):
    def __init__(self):
        self.analysis_step = 0
        self._fp16 = None

    def step(self, step=1):
        self.analysis_step += step


class Trainer(object):

    def __init__(self, model, train_loss, steper,
                 trunc_size=0, shard_size=32,
                 norm_method="sents",
                 n_gpu=1, gpu_rank=1,
                 report_manager=None, with_align=False, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32'):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.steper = steper
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.with_align = with_align
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype

        self.model.train()

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            yield batches, normalization
            batches = []
            normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1) / (step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps):

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):
            self.steper.step(batches[0].tgt.size(1))
            step = self.steper.analysis_step

            if self.n_gpu > 1:
                normalization = sum(onmt.utils.distributed
                                    .all_gather_list
                                    (normalization))

            self._gradient_accumulation(
                batches, normalization, total_stats,
                report_stats)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps,
                report_stats)

            if 0 < train_steps <= step:
                break

        return total_stats

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):

        for k, batch in enumerate(true_batches):
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt

            bptt = False
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                outputs, attns, new_cost = self.model(src, tgt, src_lengths, bptt=bptt,
                                                      with_align=self.with_align)
                bptt = True

                # 3. Compute loss.
                loss, batch_stats = self.train_loss(
                    batch,
                    outputs,
                    attns,
                    normalization=normalization,
                    shard_size=self.shard_size,
                    trunc_start=j,
                    trunc_size=trunc_size,
                    new_cost=new_cost)

                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, 0, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(0,
                step, train_stats=train_stats,
                valid_stats=valid_stats)


if __name__ == "__main__":
    main()
