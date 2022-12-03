#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import time
import numpy as np
from itertools import count, zip_longest

import torch
import math

from src import onmt
import onmt.inputters as inputters
import onmt.decoders.ensemble
from onmt.translate.beam_search import BeamSearch
from onmt.translate.greedy_search import GreedySearch
from onmt.utils.misc import tile, set_random_seed, report_matrix
from onmt.utils.alignment import extract_alignment, build_align_pharaoh
from onmt.modules.copy_generator import collapse_copy_scores
from tqdm import tqdm

UNK_TOKEN=1
EOS_TOKEN=2
BOS_TOKEN=1
PAD_TOKEN=0

def build_translator(opt, report_score=True, logger=None, out_file=None):
    model, model_opt = src.onmt.model_builder.load_test_model(opt)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)
    translator = Translator.from_opt(
        model,
        scorer,
        opt,
        model_opt,
        logger=logger
    )
    return translator, model_opt


class Translator(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        random_sampling_temp (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        block_ngram_repeat (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            model,
            global_scorer=None,
            gpu=-1,
            min_length=0,
            max_length=100,
            beam_size=5,
            random_sampling_topk=1,
            random_sampling_temp=1,
            stepwise_penalty=None,
            block_ngram_repeat=0,
            replace_unk=False,
            logger=None,
            trim_size=512,
            seed=-1):
        self.model = model
        self.trim_size = trim_size
        if not hasattr(self.model, "tgt_stoi"):
            self.model.config.tie_word_embeddings = False
        self.global_scorer = global_scorer
        self._gpu = gpu
        self.device = torch.device("cuda", self._gpu)

        self.min_length = min_length
        self.max_length = max_length
        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk

        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.replace_unk = replace_unk
        self.logger = logger
        # self.reserved_dict = {t:self.model.tokenizer.convert_tokens_to_ids(t) for t in reserved_tokens}

        set_random_seed(seed, True)

    @classmethod
    def from_opt(
            cls,
            model,
            scorer,
            opt,
            model_opt,
            logger=None):

        return cls(
            model,
            global_scorer=scorer,
            gpu=opt.gpu,
            min_length=opt.min_length,
            max_length=opt.max_length,
            beam_size=opt.beam_size,
            random_sampling_topk=opt.random_sampling_topk,
            random_sampling_temp=opt.random_sampling_temp,
            stepwise_penalty=opt.stepwise_penalty,
            block_ngram_repeat=opt.block_ngram_repeat,
            replace_unk=opt.replace_unk,
            logger=logger,
            trim_size=model_opt.trim_size,
            seed=opt.seed)

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def process_batch(self, data, train=True, shuffle=False):
        last_batch = self.last_test_batch
        batch_size = self.batch_size

        if (last_batch + batch_size) <= len(data):
            next_batch = last_batch + batch_size
            batch = data[last_batch:next_batch]
        else:
            batch = data[last_batch:]
            if shuffle:
                random.shuffle(data)
                batch += data[:(last_batch + batch_size) - len(data)]
                next_batch = (last_batch + batch_size) - len(data)
            else:
                next_batch = 0

        source = [d[0] for d in batch]
        target = [d[1] for d in batch]
        inputs = self.model.tokenizer(source, return_tensors="pt", max_length=self.trim_size,
                                      truncation=True, padding=True)
        input_ids = inputs.input_ids.to(self.device, non_blocking=True)
        attn_mask = inputs.attention_mask.to(self.device, non_blocking=True)

        if hasattr(self.model, "tgt_stoi"):
            target = [sent.split() + ['<sep>'] for sent in target]
            max_length = max([len(sent) for sent in target])
            labels = torch.zeros((len(target), max_length))
            for i, sent in enumerate(target):
                idx_sent = torch.tensor([self.model.tgt_stoi[s] for s in sent])
                labels[i][:len(idx_sent)] = idx_sent
            dinput_ids = labels.long()
        else:
            labels = self.model.tokenizer(target, return_tensors="pt", max_length=self.trim_size,
                                          truncation=True, padding=True)
            dinput_ids = labels.input_ids

        self.last_test_batch = next_batch
        return input_ids, attn_mask, target, dinput_ids

    def translate(self, data, batch_size=None):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """

        self.batch_size = batch_size
        self.last_test_batch = 0
        
        # Statistics
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0
        pred_acc_total, pred_sents_total = 0, 0

        all_scores = []
        all_predictions = []

        iters = math.ceil(len(data) / self.batch_size)        

        for i in tqdm(range(iters)):
            input_ids, attn_mask, labels, labels_ids = self.process_batch(data, train=False, shuffle=False)
            if hasattr(self.model, "tgt_stoi"):
                greedy_output = self.translate_batch(input_ids, attn_mask)
                preds = []
                for sent in greedy_output["predictions"]:
                    preds.append(' '.join([self.model.tgt_itos[w] for w in sent[0].tolist()]))
                labels = [' '.join(text[1:]) for text in labels]
            else:
                greedy_output = self.model.generate(input_ids,
                                    attention_mask=attn_mask,
                                    num_beams=self.beam_size,
                                    min_length=self.min_length,
                                    max_length=self.max_length,
                                    early_stopping=True,
                                    no_repeat_ngram_size=self.block_ngram_repeat,
                                    top_k=self.sample_from_topk,
                                    temperature=self.random_sampling_temp)
                preds = [self.model.tokenizer.decode(s).replace(self.model.tokenizer.pad_token, "").replace(self.model.tokenizer.eos_token, "").strip() for s in greedy_output]
                labels = [self.model.tokenizer.decode(s).replace(self.model.tokenizer.pad_token, "").replace(self.model.tokenizer.eos_token, "").strip() for s in labels_ids]

            correct = [pred==gold for (pred, gold) in zip(preds, labels)]
            pred_acc_total += sum(correct)
            pred_sents_total += len(preds)   
            all_predictions += preds

        msg = 'PRED ACC : {} ({}/{})'.format(pred_acc_total/pred_sents_total, pred_acc_total, pred_sents_total)
        self._log(msg)

        return all_scores, all_predictions

    def translate_batch(self, input_ids, attn_mask):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if self.beam_size == 1:
                decode_strategy = GreedySearch(
                    pad=PAD_TOKEN, bos=BOS_TOKEN, eos=EOS_TOKEN, batch_size=input_ids.shape[0],
                    min_length=self.min_length, max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens={},
                    return_attention=self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk)
            else:
                decode_strategy = BeamSearch(self.beam_size,
                    pad=PAD_TOKEN, bos=BOS_TOKEN, eos=EOS_TOKEN, batch_size=input_ids.shape[0],
                    min_length=self.min_length, max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens={},
                    return_attention=self.replace_unk,
                    n_best=1, global_scorer=self.global_scorer,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=0)
            return self._translate_batch_with_strategy(input_ids, attn_mask,
                                                       decode_strategy)


    def _run_encoder(self, src, attn_mask):
        lengths = attn_mask.sum(dim=1)
        hidden = self.model.encoder(src, attn_mask)
        enc_states, memory_bank, _ = self.model.decoder.post_encoder(hidden, lengths)
        src = src.unsqueeze(-1).transpose(0, 1).contiguous()
        return src, enc_states, memory_bank, lengths

    def _decode_and_generate(
            self,
            decoder_in,
            memory_bank,
            memory_lengths,
            step=None,
            batch_offset=None):

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder.post_decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
        )

        # Generator forward.
        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None
        log_probs = self.model.generator(dec_out.squeeze(0))
        return log_probs, attn

    def _translate_batch_with_strategy(
            self,
            input_ids, 
            attn_mask,
            decode_strategy):
        """Translate a batch of sentences step by step using cache.

        Args:
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = input_ids.shape[0]

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(input_ids, attn_mask)
        self.model.decoder.post_decoder.init_state(src, memory_bank, enc_states)

        results = {
            "predictions": None,
            "scores": None
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        _, memory_bank, memory_lengths, _ = decode_strategy.initialize(memory_bank, src_lengths, None)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                memory_lengths=memory_lengths,
                step=step,
                batch_offset=decode_strategy.batch_offset)

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                memory_bank = memory_bank.index_select(1, select_indices)
                memory_lengths = memory_lengths.index_select(0, select_indices)

            if parallel_paths > 1 or any_finished:
                self.model.decoder.post_decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        return results


    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            avg_score = score_total / words_total
            ppl = np.exp(-score_total.item() / words_total)
            msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name, avg_score,
                name, ppl))
        return msg
