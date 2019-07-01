"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, vectors, hidden_size, drop_prob=0., use_char=False, use_transformer=False, **kwargs):
        super(BiDAF, self).__init__()
        self.use_char = use_char
        self.use_transformer = use_transformer
        self.hidden_size = hidden_size

        if not use_char:
            self.emb = layers.Embedding(vectors=vectors,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob)
        else:
            self.char_size = vectors.shape[1]
            self.char_limit = kwargs['char_limit']
            self.emb = layers.Char2WordEmbedding(vectors=vectors,
                                                 hidden_size=hidden_size,
                                                 drop_prob=drop_prob,
                                                 char_limit=kwargs['char_limit'])
        if not use_transformer:
            self.enc = layers.RNNEncoder(input_size=hidden_size,
                                         hidden_size=hidden_size,  # output = 2*hidden_size
                                         num_layers=1,
                                         drop_prob=drop_prob)
            self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                         hidden_size=hidden_size,  # output = 2*hidden_size
                                         num_layers=2,
                                         drop_prob=drop_prob)
            self.out = layers.BiDAFOutput(hidden_size=2 * hidden_size, drop_prob=drop_prob,
                                          use_transformer=use_transformer)
        else:
            self.heads = kwargs['heads']
            self.inter_size = kwargs['inter_size']
            self.PE = layers.PositionalEncodings(hidden_size, .1)
            self.enc = layers.TransformerEncoderStack(
                N=1,
                heads=self.heads,
                input_size=hidden_size,
                output_size=hidden_size,
                inter_size=self.inter_size,
                drop_prob=.1
                )

            self.mod = layers.TransformerEncoderStack(
                N=1,
                heads=self.heads,
                input_size=4*hidden_size,
                output_size=hidden_size,
                inter_size=self.inter_size,
                drop_prob=.1
                )

            self.out = layers.BiDAFOutput(hidden_size=hidden_size, drop_prob=drop_prob,
                                          use_transformer=use_transformer,
                                          heads=self.heads, inter_size=self.inter_size)

        self.att = layers.BiDAFAttention(hidden_size=(1 if self.use_transformer else 2)*hidden_size,
                                         drop_prob=drop_prob)  # (batch_size, seq_len, 4*input_hidden_size)

    def forward(self, c_idxs, q_idxs):
        """
        :param c_idxs: A tuple of word and char indices for the context.
        :param q_idxs: A tuple of word and char indices for the question.
        :return:
        """
        cw_idxs, cc_idxs = c_idxs  # cc_idxs (batch_size, c_limit, char_limit)
        qw_idxs, qc_idxs = q_idxs  # qc_idxs (batch_size, q_limit, char_limit)
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs  # c_mask, cw_idxs (batch_size, c_limit)
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)  # (batch_size) Actual length
        c_limit, q_limit = cc_idxs.shape[1], qc_idxs.shape[1]

        if self.use_char:
            cc_idxs = cc_idxs.view(-1, self.char_limit)  # (batch_size*c_limit, char_limit)
            qc_idxs = qc_idxs.view(-1, self.char_limit)  # (batch_size*q_limit, char_limit)
            c_emb = self.emb(cc_idxs).view(-1, c_limit, self.hidden_size)  # (batch_size, c_limit, hidden_size)
            q_emb = self.emb(qc_idxs).view(-1, q_limit, self.hidden_size)  # (batch_size, q_limit, hidden_size)
        else:
            c_emb = self.emb(cw_idxs)         # (batch_size, c_limit, hidden_size)
            q_emb = self.emb(qw_idxs)         # (batch_size, q_limit, hidden_size)

        if self.use_transformer:
            c_emb = self.PE(c_emb)  # (batch_size, c_limit, hidden_size)
            q_emb = self.PE(q_emb)  # (batch_size, q_limit, hidden_size)

        c_enc = self.enc(c_emb, c_mask if self.use_transformer else c_len)  # (batch_size, c_limit, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_mask if self.use_transformer else q_len)  # (batch_size, q_limit, 2 * hidden_size)

        att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_limit, 8 * hidden_size)
        mod = self.mod(att, c_mask if self.use_transformer else c_len)  # (batch_size, c_limit, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_limit)

        return out
