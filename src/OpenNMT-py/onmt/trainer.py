"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import numpy as np
import torch
import traceback
import math
import random
import onmt
from onmt.utils.logging import logger


def build_trainer(opt, device_id, model, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    train_loss = onmt.utils.loss.build_loss_compute(model, opt)
    valid_loss = onmt.utils.loss.build_loss_compute(model, opt, train=False)

    trunc_size = opt.truncated_decoder
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt), base_path=opt.save_model) \
        if opt.early_stopping > 0 else None

    source_noise = None
    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = onmt.Trainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           with_align=True if opt.lambda_align > 0 else False,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           dropout_steps=dropout_steps,
                           source_noise=source_noise,
                           batch_size=opt.batch_size,
                           trim_size=opt.trim_size)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1, gpu_verbose_level=0,
                 report_manager=None, with_align=False, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0],
                 source_noise=None, batch_size=16, trim_size=512):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.with_align = with_align
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.source_noise = source_noise
        self.batch_size = batch_size
        self.device = torch.device("cuda:%s" % self.gpu_rank)
        self.trim_size = trim_size

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def build_probs_list(self, temp=4):
        probs_list = {}
        for example in list(self.neighbours_list):
            edists = self.edists_list[example]
            edists_v = torch.Tensor(edists)
            probs = torch.softmax(-edists_v / temp, dim=0).numpy()
            probs_list[example] = probs
        self.probs_list = probs_list

    def totensor_batch(self, batch):
        source = [d[0] for d in batch]
        target = [d[1] for d in batch]
        index_map = []
        inputs = self.model.tokenizer(source, return_tensors="pt", max_length=self.trim_size, truncation=True, padding=True)
        input_ids = (inputs.input_ids)[:, :self.trim_size].to(self.device, non_blocking=True)
        attn_mask = (inputs.attention_mask)[:, :self.trim_size].to(self.device, non_blocking=True)
        eos = '<extra_id_99>'
        # next step use eos = '<eos>'
        eos_token = self.model.tokenizer(eos)['input_ids'][0]
        for i in range(len(input_ids)):
            this_map = []
            for ind in range(len(input_ids[i])):
                if input_ids[i][ind] == eos_token:
                    this_map.append(ind)
            this_map = torch.tensor(this_map)
            index_map.append(this_map)
        index_map = torch.stack(index_map)
        if hasattr(self.model, "tgt_stoi"):
            target = [sent.split() + ['<sep>'] for sent in target]
            max_length = max([len(sent) for sent in target])
            labels = torch.zeros((len(target), max_length))
            for i, sent in enumerate(target):
                idx_sent = torch.tensor([self.model.tgt_stoi[s] for s in sent])
                labels[i][:len(idx_sent)] = idx_sent
            dinput_ids = labels[:, :self.trim_size].to(self.device, non_blocking=True).long()
            dattn_mask = None
        else:
            labels = self.model.tokenizer(target, return_tensors="pt", max_length=self.trim_size, truncation=True, padding=True)
            label = target #multicoice
            dinput_ids = (label)
            #dinput_ids = (labels.input_ids)[:, :self.trim_size].to(self.device, non_blocking=True)
            dattn_mask = (labels.attention_mask)[:, :self.trim_size].to(self.device, non_blocking=True)

        return input_ids, attn_mask, dinput_ids, dattn_mask, index_map

    def process_batch(self, data, train=True, shuffle=True):
        batch_ids = None
        if train:
            last_batch = self.last_train_batch
        else:
            last_batch = self.last_test_batch

        batch_size = self.batch_size
        if (last_batch + batch_size) <= len(data):
            next_batch = last_batch + batch_size
            if train:
                batch_ids = self.sample_ids[last_batch:next_batch]
                batch = [data[d] for d in batch_ids]
            else:
                batch = data[last_batch:next_batch]
        else:
            if train:
                batch_ids = self.sample_ids[last_batch:]
                self.sample_ids = np.random.permutation(len(data))
                batch_ids = np.concatenate((batch_ids, self.sample_ids[:(last_batch + batch_size) - len(data)]))
                batch = [data[d] for d in batch_ids]
                next_batch = (last_batch + batch_size) - len(data)
            else:
                batch = data[last_batch:]
                next_batch = 0

        if train:
            self.last_train_batch = next_batch
        else:
            self.last_test_batch = next_batch

        return self.totensor_batch(batch), batch_ids

    def train(self,
              train_data,
              train_steps=30000,
              save_checkpoint_steps=5000,
              valid_data=None,
              valid_steps=10000):
        if valid_data is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...', valid_steps)

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        best_acc = 0
        self.last_train_batch = 0
        self.sample_ids = np.random.permutation(len(train_data))
    
        for i in range(train_steps):
            step = self.optim.training_step
            self._gradient_accumulation(train_data, self.batch_size*self.accum_count, total_stats, report_stats)

            report_stats = self._maybe_report_training(step, train_steps, self.optim.learning_rate(), report_stats)

            if valid_data is not None and step % valid_steps == 0:
                valid_stats = self.validate(valid_data)
                valid_stats = self._maybe_gather_stats(valid_stats)
                self._report_step(self.optim.learning_rate(), step, valid_stats=valid_stats)
                if valid_stats.accuracy() > best_acc:
                    best_acc = valid_stats.accuracy()
                    print("Best Acc: %s" % best_acc)
                    self.model_saver.save("best")
                else:
                    self.model_saver.save("last")
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    if self.earlystopper.has_stopped():
                        break

            if train_steps > 0 and step >= train_steps:
                self.model_saver.save("last")
                break

        return step 

    def validate(self, valid_data):
        valid_model = self.model
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()
            self.last_test_batch = 0
            iters = math.ceil(len(valid_data) / self.batch_size)

            for i in range(iters):
                (input_ids, attn_mask, dinput_ids, dattn_mask, index_map), _ = self.process_batch(valid_data, train=False, shuffle=False)
                scores = valid_model(input_ids, attn_mask, dinput_ids, dattn_mask)
                _, batch_stats = self.valid_loss(dinput_ids.transpose(0,1).contiguous()[1:], scores, back=False)
                stats.update(batch_stats)

        valid_model.train()
        return stats

    def _gradient_accumulation(self, train_data, normalization, total_stats, report_stats):
        self.optim.zero_grad()

        for k in range(self.accum_count):
            (input_ids, attn_mask, dinput_ids, dattn_mask, index_map), _ = self.process_batch(train_data, train=True)
            report_stats.n_src_words += attn_mask.sum().item()
            scores = self.model(input_ids, attn_mask, dinput_ids, dattn_mask, index_map)
            #dinput_ids = dinput_ids.transpose(0, 1).contiguous()[1]
            target = dinput_ids.transpose(0, 1).contiguous()[1]
            _, batch_stats = self.train_loss(
                target,
                scores,
                normalization=normalization)

            report_stats.update(batch_stats)

        self.optim.step()

    
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

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

