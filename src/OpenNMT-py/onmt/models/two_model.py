""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn
from transformers import T5Model, T5Tokenizer, T5ForConditionalGeneration, \
    BartTokenizer, BartForConditionalGeneration, BartConfig, \
    LongT5ForConditionalGeneration, AutoTokenizer

class LT2Model(LongT5ForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained('google/long-t5-tglobal-base')

class BT2Model(BartForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

class NMT2Model(T5ForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

