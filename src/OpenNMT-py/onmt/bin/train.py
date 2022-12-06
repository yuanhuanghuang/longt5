#!/usr/bin/env python
"""Train models."""
import os
import signal
import torch

import onmt 
from onmt import opts
from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.train_single import main as single_main
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import build_dataset_iter, patch_fields, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple

from itertools import cycle


def train(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)
    nb_gpu = len(opt.gpu_ranks)

    if nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0)
    else:   # case only CPU
        single_main(opt, -1)


def batch_producer(generator_to_serve, queues, semaphore, opt):
    init_logger(opt.log_file)
    set_random_seed(opt.seed, False)
    # generator_to_serve = iter(generator_to_serve)

    def pred(x):
        """
        Filters batches that belong only
        to gpu_ranks of current node
        """
        for rank in opt.gpu_ranks:
            if x[0] % opt.world_size == rank:
                return True

    generator_to_serve = filter(
        pred, enumerate(generator_to_serve))

    def next_batch(device_id):
        new_batch = next(generator_to_serve)
        semaphore.acquire()
        return new_batch[1]

    b = next_batch(0)

    for device_id, q in cycle(enumerate(queues)):
        b.dataset = None
        if isinstance(b.src, tuple):
            b.src = tuple([_.to(torch.device(device_id))
                           for _ in b.src])
        else:
            b.src = b.src.to(torch.device(device_id))
        b.tgt = b.tgt.to(torch.device(device_id))
        b.indices = b.indices.to(torch.device(device_id))
        b.alignment = b.alignment.to(torch.device(device_id)) \
            if hasattr(b, 'alignment') else None
        b.src_map = b.src_map.to(torch.device(device_id)) \
            if hasattr(b, 'src_map') else None
        b.align = b.align.to(torch.device(device_id)) \
            if hasattr(b, 'align') else None
        b.corpus_id = b.corpus_id.to(torch.device(device_id)) \
            if hasattr(b, 'corpus_id') else None

        # hack to dodge unpicklable `dict_keys`
        b.fields = list(b.fields)
        q.put(b)
        b = next_batch(device_id)


def run(opt, device_id, error_queue, batch_queue, semaphore):
    """ run process """
    try:
        gpu_rank = src.onmt.utils.distributed.multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        single_main(opt, device_id, batch_queue, semaphore)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    train(opt)


if __name__ == "__main__":
    main()
