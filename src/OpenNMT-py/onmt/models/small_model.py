""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn


class SmallEncoder(nn.Module):
    def __init__(self, transformer):
        super(SmallEncoder, self).__init__()
        self.transformer = transformer

    def forward(self, src, mask):
        transformer_output = self.transformer(src, mask)
        hidden = transformer_output.last_hidden_state.transpose(0,1).contiguous()
        return hidden


class SmallDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(SmallDecoder, self).__init__()
        self.post_encoder = encoder
        self.post_decoder = decoder

    def forward(self, src, dec_in, hidden, lengths):
        enc_state, memory_bank, _ = self.post_encoder(hidden, lengths)
        self.post_decoder.init_state(src, memory_bank, enc_state)
        dec_out, _ = self.post_decoder(dec_in, memory_bank, memory_lengths=lengths)
        return dec_out

class SmallGen(nn.Module):
    def __init__(self, generator):
        super(SmallGen, self).__init__()
        self.gen = generator

    def forward(self, output):
        return self.gen(output)


class SmallModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, transformer, encoder, decoder, generator, tokenizer, tgt_vocab):
        super(SmallModel, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = SmallEncoder(transformer)
        self.decoder = SmallDecoder(encoder, decoder)
        self.generator = SmallGen(generator)
        self.tgt_itos = {i:w for i, w in enumerate(tgt_vocab)}
        self.tgt_stoi = {w:i for i, w in enumerate(tgt_vocab)}

    def forward(self, src, attn_mask, tgt, dattn_mask):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        lengths = attn_mask.sum(dim=1)
        hidden = self.encoder(src, attn_mask)
        src = src.unsqueeze(-1).transpose(0,1).contiguous()
        dec_in = tgt.unsqueeze(-1).transpose(0,1).contiguous()[:-1]

        dec_out = self.decoder(src, dec_in, hidden, lengths)
        bottled_output = dec_out.view(-1, dec_out.size(2))
        scores = self.generator(bottled_output)

        return scores

