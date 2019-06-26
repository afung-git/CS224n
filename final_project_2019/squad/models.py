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
    def __init__(self, vectors, hidden_size, drop_prob=0., use_char=False, **kwargs):
        super(BiDAF, self).__init__()
        self.use_char = use_char
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

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, c_idxs, q_idxs):
        cw_idxs, cc_idxs = c_idxs  # cc_idxs (batch_size, c_len, char_limit)
        qw_idxs, qc_idxs = q_idxs  # qc_idxs (batch_size, q_len, char_limit)
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        len_c, len_q = cc_idxs.shape[1], qc_idxs.shape[1]

        if self.use_char:
            cc_idxs = cc_idxs.view(-1, self.char_limit)  # (batch_size*c_len, char_limit)
            qc_idxs = qc_idxs.view(-1, self.char_limit)  # (batch_size*q_len, char_limit)
            c_emb = self.emb(cc_idxs).view(-1, len_c, self.hidden_size)         # (batch_size, c_len, hidden_size)
            q_emb = self.emb(qc_idxs).view(-1, len_q, self.hidden_size)         # (batch_size, q_len, hidden_size)
        else:
            c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
            q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
