"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from transformers import T5Model, T5Config, T5Tokenizer, \
    BertTokenizer, BertConfig, BertModel, \
    BartConfig, BartModel, BartTokenizer, \
    LongT5Model, AutoTokenizer, AutoConfig
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt
from onmt.encoders import str2enc
from onmt.decoders import str2dec

from onmt.modules import Embeddings, VecEmbedding, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser

BertLayerNorm = torch.nn.LayerNorm

def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    batch_size = opt.batch_size
    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    model_opt.batch_size = batch_size
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)

    if model_opt.architect == 't5' or model_opt.architect == 'longt5':
        model = build_base_model(model_opt, use_gpu(opt), checkpoint, evaluate=True)
    elif model_opt.architect == 'bart':
        model = build_bart_model(model_opt, use_gpu(opt), checkpoint, evaluate=True)
    else:
        model = build_small_model(model_opt, opt.target_vocab, use_gpu(opt), checkpoint, evaluate=True)

    model.eval()
    model.generator.eval()
    return model, model_opt


def build_transformer(model_opt, vocab_size):
    configuration = BertConfig(**{"hidden_size": 512, "hidden_act": "gelu", "initializer_range": 0.02, "vocab_size": vocab_size, "hidden_dropout_prob": 0.1, "num_attention_heads": 8, "type_vocab_size": 2, "max_position_embeddings": 512, "num_hidden_layers": 4, "intermediate_size": 2048, "attention_probs_dropout_prob": 0.1})
    transformer = BertModel(configuration)
    cp = torch.load(model_opt.bert_path)
    transformer.load_state_dict(cp)
    return transformer


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    enc_type = opt.encoder_type if opt.model_type == "text" \
        or opt.model_type == "vec" else opt.model_type
    return str2enc[enc_type].from_opt(opt, embeddings)


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed \
               else opt.decoder_type
    return str2dec[dec_type].from_opt(opt, embeddings)


def build_small_model(model_opt, vocab_file, gpu, checkpoint, evaluate=False):

    def fix_key(s):
        s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                   r'\1.layer_norm\2.bias', s)
        s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                   r'\1.layer_norm\2.weight', s)
        return s

    # Build embeddings.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transformer = build_transformer(model_opt, len(tokenizer.vocab))

    # Build encoder.
    encoder = build_encoder(model_opt, None)

    # Build decoder.
    tgt_vocab = open(vocab_file).read().splitlines()
    tgt_vocab = ['<pad>', '<cls>', '<sep>'] + tgt_vocab
    tgt_emb = Embeddings(model_opt.tgt_word_vec_size, len(tgt_vocab), 0,
        position_encoding=model_opt.position_encoding)
    decoder = build_decoder(model_opt, tgt_emb)

    # Build Generator.
    if evaluate:
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size, len(tgt_vocab))
        )
    else:
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size, len(tgt_vocab)),
            Cast(torch.float32),
            nn.LogSoftmax(dim=-1)
        )

    # Build NMTModel(= encoder + decoder).
    device = torch.device("cuda")
    model = onmt.models.SmallModel(transformer, encoder, decoder, generator, tokenizer, tgt_vocab)


    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        model.load_state_dict(checkpoint['model'], strict=False)
        model.generator.load_state_dict(checkpoint['generator'], strict=False)
    else:
        for p in model.encoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for p in model.decoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for p in generator.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        if model_opt.partial_load_path is not None:
            checkpoint = torch.load(model_opt.partial_load_path, map_location=lambda storage, loc: storage)
            checkpoint['model'] = {fix_key(k): v
                for k, v in checkpoint['model'].items() if k.startswith('encoder')}
            model.load_state_dict(checkpoint['model'], strict=False)

    model.to(device)
    return model

def build_bart_model(model_opt, gpu, checkpoint, evaluate=False):
    """
    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.

    Returns:
        the T5Model.
    """
    if model_opt.architect == "t5":
        from onmt.models import NMT2Model as T5GenerationModel
        from onmt.models import NMTModel as T5Model
        model_name = 't5-base'
    elif model_opt.architect == "longt5":
        from onmt.models import LT2Model as T5GenerationModel
        from onmt.models import LTModel as T5Model
        model_name = 'google/long-t5-tglobal-base'
    else:
        raise NotImplementedError()

    # This preserves backward-compat for models using customed layernorm
    def fix_key(s):
        s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                   r'\1.layer_norm\2.bias', s)
        s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                   r'\1.layer_norm\2.weight', s)
        return s

    # Build T5Model (= encoder + decoder).
    device = torch.device("cuda")
    config = BartConfig.from_pretrained('facebook/bart-large-cnn')

    if evaluate:
        model = onmt.models.BT2Model(config=config)
    elif model_opt.load_t5:
        model = onmt.models.BTModel.from_pretrained('facebook/bart-large-cnn')
        model.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    else:
        model = onmt.models.BTModel(config=config)
    model.config.tie_word_embeddings = True

    # Build Generator.
    if evaluate:
        generator = nn.Sequential(
            nn.Linear(model.config.d_model, model.config.d_model),
            nn.ReLU(),
            BertLayerNorm(model.config.d_model, eps=1e-12),
            nn.Linear(model.config.d_model, model.config.d_model),
            nn.ReLU(),
            BertLayerNorm(model.config.d_model, eps=1e-12),
            nn.Linear(model.config.d_model, model.config.vocab_size, bias=False),
        )
    elif model_opt.load_lm or model_opt.reset_decoder:
        generator = nn.Sequential(
            nn.Linear(model.config.d_model, model.config.d_model),
            nn.ReLU(),
            BertLayerNorm(model.config.d_model, eps=1e-12),
            nn.Linear(model.config.d_model, model.config.d_model),
            nn.ReLU(),
            BertLayerNorm(model.config.d_model, eps=1e-12),
            nn.Linear(model.config.d_model, model.config.vocab_size, bias=False),
            Cast(torch.float32),
            nn.LogSoftmax(dim=-1)
        )
        model2 = onmt.models.BT2Model.from_pretrained('facebook/bart-large-cnn')
        lm_head = {'0.weight': model2.state_dict()['lm_head.weight']}
        generator.load_state_dict(lm_head)
    else:
        generator = nn.Sequential(
            nn.Linear(model.config.d_model, model.config.d_model),
            nn.ReLU(),
            BertLayerNorm(model.config.d_model, eps=1e-12),
            nn.Linear(model.config.d_model, model.config.d_model),
            nn.ReLU(),
            BertLayerNorm(model.config.d_model, eps=1e-12),
            nn.Linear(model.config.d_model, model.config.vocab_size, bias=False),
            Cast(torch.float32),
            nn.LogSoftmax(dim=-1)
        )
    model.generator = generator

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'], strict=False)
            if evaluate:
                model.lm_head = generator
    elif model_opt.partial_load_path is not None:
        checkpoint = torch.load(model_opt.partial_load_path,
                                map_location=lambda storage, loc: storage)
        if model_opt.reset_decoder:
            model_state_dict = {k: v for k, v in checkpoint['model'].items()
                                if 'decoder' not in k}
            model.load_state_dict(model_state_dict, strict=False)      
        else:
            model.load_state_dict(checkpoint['model'], strict=False)
            if 'generator' in checkpoint and not model_opt.load_lm:
                generator.load_state_dict(checkpoint['generator'], strict=False)
    # must load t5

    if model_opt.fix_enc:
        for p in model.encoder.parameters():
            p.requires_grad = False
    model.to(device)
    return model



def build_base_model(model_opt, gpu, checkpoint, evaluate=False):
    """
    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        gpu (bool): whether to use gpu.
        checkpoint: the model generated by train phase, or a resumed snapshot
                    model from a stopped training.

    Returns:
        the T5Model.
    """
    if model_opt.architect == "t5":
        from onmt.models import NMT2Model as T5GenerationModel
        from onmt.models import NMTModel as T5Model
        model_name = 't5-base'
    elif model_opt.architect == "longt5":
        from onmt.models import LT2Model as T5GenerationModel
        from onmt.models import LTModel as T5Model
        if model_opt.mode == 'mc':
            #from onmt.models import LTEncModel as T5Model
            from onmt.models import LTModel as T5Model
        model_name = 'google/long-t5-tglobal-base'
    else:
        raise NotImplementedError()

    # This preserves backward-compat for models using customed layernorm
    def fix_key(s):
        s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                   r'\1.layer_norm\2.bias', s)
        s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                   r'\1.layer_norm\2.weight', s)
        return s

    # Build T5Model (= encoder + decoder).
    device = torch.device("cuda")
    config = AutoConfig.from_pretrained(model_name)

    if evaluate:
        model = T5GenerationModel(config=config)
    elif model_opt.load_t5:
        model = T5Model.from_pretrained(model_name)
        model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = T5Model(config=config)
    model.config.tie_word_embeddings = False

    # Build Generator.
    if evaluate:
        generator = nn.Sequential(
            nn.Linear(model.config.d_model, model.config.vocab_size, bias=False)
        )
    elif model_opt.load_lm or model_opt.reset_decoder:
        generator = nn.Sequential(
            nn.Linear(model.config.d_model, model.config.vocab_size, bias=False),
            Cast(torch.float32),
            nn.LogSoftmax(dim=-1)
        )
        model2 = T5GenerationModel.from_pretrained(model_name)
        lm_head = {'0.weight': model2.state_dict()['lm_head.weight']}
        generator.load_state_dict(lm_head)
    elif model_opt.mode == 'mc':
        generator = nn.Sequential(
            nn.Linear(model.config.d_model, model_opt.num_choice, bias=False),
            Cast(torch.float32),
            nn.LogSoftmax(dim=-1)
        )
    else:
        generator = nn.Sequential(
            nn.Linear(model.config.d_model, model.config.vocab_size, bias=False),
            Cast(torch.float32),
            nn.LogSoftmax(dim=-1)
        )
    model.generator = generator

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'], strict=False)
            if evaluate:
                model.lm_head = generator
    elif model_opt.partial_load_path is not None:
        checkpoint = torch.load(model_opt.partial_load_path,
                                map_location=lambda storage, loc: storage)
        if model_opt.reset_decoder:
            model_state_dict = {k: v for k, v in checkpoint['model'].items()
                                if 'decoder' not in k}
            model.load_state_dict(model_state_dict, strict=False)      
        else:
            model.load_state_dict(checkpoint['model'], strict=False)
            if 'generator' in checkpoint and not model_opt.load_lm:
                generator.load_state_dict(checkpoint['generator'], strict=False)
    # must load t5

    if model_opt.fix_enc:
        for p in model.encoder.parameters():
            p.requires_grad = False
    model.to(device)
    return model


def build_model(model_opt, opt, checkpoint):
    logger.info('Building model...')
    if model_opt.architect == 't5' or model_opt.architect == 'longt5':
        model = build_base_model(model_opt, use_gpu(opt), checkpoint)
    elif model_opt.architect == 'bart':
        model = build_bart_model(model_opt, use_gpu(opt), checkpoint)
    else:
        model = build_small_model(model_opt, opt.target_vocab, use_gpu(opt), checkpoint)
    return model
