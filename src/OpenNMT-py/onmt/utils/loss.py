"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import onmt
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax

POS_TOKEN = 0


def build_loss_compute(model, opt, train=True):
    device = torch.device("cuda" if src.onmt.utils.misc.use_gpu(opt) else "cpu")
    padding_idx = POS_TOKEN
    criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
    compute = NMTLossCompute(criterion)
    compute.to(device)
    return compute


class LossComputeBase(nn.Module):
    """
    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _compute_loss(self, output, target, **kwargs):
        """
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def __call__(self, tgt, output, normalization=1.0, back=True):
        """
        Args:
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        batch_stats = onmt.utils.Statistics()
        loss, stats = self._compute_loss(output, tgt)
        if back:
            loss.div(float(normalization)).backward()
        else:
            loss = loss.div(float(normalization))
        batch_stats.update(stats)
        return loss, batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)


class NMTLossCompute(LossComputeBase):
    def __init__(self, criterion, normalization="sents"):
        super(NMTLossCompute, self).__init__(criterion)

    def _compute_loss(self, scores, target):
        gtruth = target.view(-1)
        loss = self.criterion(scores, gtruth)
        stats = self._stats(loss.clone(), scores, gtruth)
        return loss, stats

### add new loss function here
