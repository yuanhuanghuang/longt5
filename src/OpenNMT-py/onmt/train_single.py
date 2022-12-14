#!/usr/bin/env python
"""Training on a single process."""
import os
import torch
import random
import pickle

from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser

MODEL_BOS_DICT = {
    "bert_small": ("no task", "<cls>"),
    "bart": ("no task", "<s>"),
    "t5": ("task", "<pad>"),
    "longt5": ("task", "<pad>"),
}

def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def print_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % (enc + dec))


def check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def load_dataset(task, architect, filename, filename2):
    source_bos = task + ": " if MODEL_BOS_DICT[architect][0] == "task" else ""
    target_bos = MODEL_BOS_DICT[architect][1] + " "
    with open(filename) as f:
        data = f.read().splitlines()
    with open(filename2) as f:
        data2 = f.read().splitlines()
    if task == 'mc':
        return [(d[0], target_bos + d[1]) for d in zip(data, data2)]
    else:
        return [(source_bos + d[0], target_bos + d[1]) for d in zip(data, data2)]


def main(opt, device_id, batch_queue=None, semaphore=None):

    # Init seeds and logger
    configure_process(opt, device_id)
    init_logger(opt.log_file)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from, map_location=lambda storage, loc: storage)
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
    else:
        checkpoint = None
        model_opt = opt
    logger.info('args: {}'.format(opt))

    # Build model
    model = build_model(model_opt, opt, checkpoint)
    print_parameters(model)
    check_save_model_path(opt)

    train_data = load_dataset(opt.task_name, opt.architect, opt.train_data, opt.train_datat)
    valid_data = load_dataset(opt.task_name, opt.architect, opt.valid_data, opt.valid_datat)

    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)
    model_saver = build_model_saver(model_opt, opt, model, optim)
    trainer = build_trainer(opt, device_id, model, optim, model_saver=model_saver)

    logger.info('Start normal training.')

    trainer.train(
        train_data,
        train_steps=opt.train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_data=valid_data,
        valid_steps=opt.valid_steps)
        

