"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

from . import layers
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
    def __init__(self, vectors, hidden_size, char_limit, use_transformer, use_GRU, drop_prob=0., **kwargs):
        super(BiDAF, self).__init__()
        self.use_transformer = use_transformer
        self.use_GRU = use_GRU
        self.hidden_size = hidden_size

        self.emb = layers.Embedding(vectors=vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    char_limit=char_limit)
        if not use_transformer:
            self.enc = layers.RNNEncoder(input_size=hidden_size,
                                         hidden_size=hidden_size,  # output = 2*hidden_size
                                         num_layers=1,
                                         drop_prob=drop_prob,
                                         use_GRU=use_GRU)
            self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                         hidden_size=hidden_size,  # output = 2*hidden_size
                                         num_layers=2,
                                         drop_prob=drop_prob,
                                         use_GRU=use_GRU)
            self.out = layers.BiDAFOutput(hidden_size=2 * hidden_size, drop_prob=drop_prob,
                                          use_transformer=use_transformer)
        else:
            self.heads = kwargs['heads']
            self.inter_size = kwargs['inter_size']
            self.enc = layers.TransformerEncoderStack(
                N=1,
                heads=self.heads,
                input_size=hidden_size,
                output_size=hidden_size,
                inter_size=self.inter_size,
                num_conv=4,
                drop_prob=.1
                )
            self.squeeze = layers.InitializedLayer(4*hidden_size, hidden_size, bias=False)
            self.mod = layers.TransformerEncoderStack(
                        N=3,
                        heads=self.heads,
                        input_size=hidden_size,
                        output_size=hidden_size,
                        inter_size=self.inter_size,
                        num_conv=2,
                        drop_prob=.1
                        )
            self.out = layers.QAOutput(2*hidden_size)

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

        cc_idxs = cc_idxs.view(-1, cc_idxs.shape[-1])  # (batch_size*c_limit, char_limit)
        qc_idxs = qc_idxs.view(-1, qc_idxs.shape[-1])  # (batch_size*q_limit, char_limit)
        c_idxs, q_idxs = (cw_idxs, cc_idxs), (qw_idxs, qc_idxs)
        c_emb = self.emb(c_idxs)  # (batch_size, c_limit, hidden_size)
        q_emb = self.emb(q_idxs)  # (batch_size, q_limit, hidden_size)

        c_enc = self.enc(c_emb, c_mask if self.use_transformer else c_len)  # (batch_size, c_limit, hidden_size)
        q_enc = self.enc(q_emb, q_mask if self.use_transformer else q_len)  # (batch_size, q_limit, hidden_size)

        att = self.att(c_enc, q_enc, c_mask, q_mask)  # (batch_size, c_limit, 4 * hidden_size)

        if self.use_transformer:
            att = self.squeeze(att)  # (batch_size, c_limit, hidden_size)
            start, end = layers.QAModel(self.mod, att, c_mask)  # 2 x (batch_size, c_limit, hidden_size)
            out = self.out(start, end, c_mask)  # 2 tensors, each (batch_size, c_limit)

        else:
            mod = self.mod(att, c_len)  # (batch_size, c_limit, 2 * hidden_size)
            out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_limit)

        return out
