"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from cnn import CNN


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(vectors)
        self.proj = nn.Linear(vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class Char2WordEmbedding(Embedding):
    """
    Assignment 5's Character to Word Embeddings
    A 1D Conv followed by Highway Network refines the output word embeddings
    Args:
        vectors (torch.Tensor): Pre-trained char vectors
        hidden_size (int): Output word vector size.
        drop_prob (float): Probability of zero-ing out activations

    """
    def __init__(self, vectors, hidden_size, drop_prob, char_limit, kernel_width=5):
        super(Char2WordEmbedding, self).__init__(vectors, hidden_size, drop_prob)
        self.proj = CNN(vectors.shape[1], hidden_size, char_limit, kernel_width)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size*seq_len, char_limit, char_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = emb.transpose(1, 2)  # (batch_size*len, char_size, char_limit)
        emb = self.proj(emb)  # (batch_size*seq_len, word_size)
        emb = self.hwy(emb)   # (batch_size*seq_len, word_size)
        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


def self_attention(query, key, value, mask=None):
    """
    :param query: Query tensor (batch x heads x seq_len x d_k)
    :param key: Key tensor (batch x heads x seq_len x d_k)
    :param value: Value tensor (batch x heads x seq_len x d_k)
    :param mask: Optional mask, same for all heads (batch x heads x seq_len x seq_len)
    :return: output, scores (batch x heads x seq_len x d_k), (batch x heads x seq_len x seq_len)
    """
    logits = torch.matmul(query, key.transpose(-1, -2))/math.sqrt(key.shape[-1])
    if mask is not None:
        logits = logits.masked_fill(mask==0, 1e-9)
    scores = F.softmax(logits, dim=-1)
    return torch.matmul(scores, value), scores


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads, hidden_size, drop_prob=0.):
        """
        :param heads: Number of attention heads to use
        :param hidden_size: Dimension of input/output vectors
        :param drop_prob: Dropout rate
        """
        super(MultiHeadSelfAttention, self).__init__()

        assert hidden_size % heads == 0, "hidden_size not a multiple of heads"

        self.d_k = hidden_size // heads
        self.heads = heads
        self.Linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, q, k, v, mask=None):
        """
        :param q: Query tensor (batch_size x seq_len x hidden_size)
        :param k: Key tensor (batch_size x seq_len x hidden_size)
        :param v: Value tensor (batch_size x seq_len x hidden_size)
        :param mask: Optional mask (batch_size x seq_len x seq_len)
        :return: o: output tensor (batch_size x seq_len x hidden_size)
        """
        batch_size = q.shape[0]

        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size x 1 x seq_len x seq_len)

        # Get the Q, K, V in multiple-heads form after linear layers
        q, k, v = [l(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
                   for l, x in zip(self.Linears, (q, k, v))]
        o, self.attn = self_attention(q, k, v, mask)  # (batch_size, heads, seq_len, d_k)
        o = self.dropout(o).transpose(1, 2).contiguous().view(batch_size, -1, self.heads*self.d_k)

        return self.Linears[-1](o)


class FeedForward(nn.Module):
    def __init__(self, input_size, output_size, inter_size, drop_prob=0.):
        """
        :param input_size: Dimension of input vectors
        :param output_size: Dimension of output vectors
        :param inter_size: Dimension of intermediate vectors
        :param drop_prob: Dropout rate
        """
        super(FeedForward, self).__init__()
        self.FF = nn.Sequential(
            nn.Linear(input_size, inter_size),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(inter_size, output_size)
        )

    def forward(self, x):
        return self.FF(x)


class Sublayer(nn.Module):
    def __init__(self, size, drop_prob=0.):
        """
        :param size: Size of input to Layernorm
        :param drop_prob: Dropout rate
        """
        super(Sublayer, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, sub):
        """
        :param x: Input (batch x seq_len x hidden_size)
        :param sub: Sublayer (Feedforward, MultiHeadSelfAttention etc.)
        :return: Normalize, Sublayer, Dropout
        """
        return self.dropout(sub(self.norm(x)))


def make_mask(masks, decode=False):
    """
    :param masks: 0 for pad, 1 for non-pad (batch x seq_len)
    :param decode: decoders are Auto-Regressive (can't see future words)
    :return: mask: (batch x seq_len x seq_len)
    """
    masks = torch.bmm(masks.unsqueeze(2).float(), masks.unsqueeze(1).float())
    if decode:
        masks = torch.from_numpy(np.tril(masks))
    return masks.long()


class TransformerEncoder(nn.Module):
    def __init__(self, heads, input_size, output_size, inter_size, drop_prob=0.):
        """
        :param heads: Number of heads in multi-layer self-attention
        :param input_size: Input hidden state size
        :param output_size: Output hidden state size
        :param inter_size: Dimension of intermediate feed-forward layers
        :param seq_len: Maximum sequence length
        :param drop_prob: Dropout rate
        """
        super(TransformerEncoder, self).__init__()
        self.MHSA = MultiHeadSelfAttention(heads, input_size, drop_prob)
        self.FF = FeedForward(input_size, output_size, inter_size, drop_prob)
        self.layers = nn.ModuleList([Sublayer(input_size, drop_prob) for _ in range(2)])
        self.downsample = nn.Linear(input_size, output_size) if input_size > output_size else None

    def forward(self, x, masks=None):
        """
        :param x: Input (batch, seq_len, hidden_size)
        :param masks: (batch, seq_len)
        :return: o: Output (batch, seq_len, hidden_size)
        """
        if masks is not None:
            masks = make_mask(masks)
        x = x + self.layers[0](x, lambda x: self.MHSA(x, x, x, masks))  # Need lambda due to mask
        return (x if self.downsample is None else self.downsample(x)) + self.layers[1](x, self.FF)


class PositionalEncodings(nn.Module):
    def __init__(self, d, drop_prob=0., max_len=5000):
        """
        :param d: Dimension of embedding
        :param drop_prob: Dropout Rate
        :param max_len: Maximum length of a sequence
        """
        super(PositionalEncodings, self).__init__()
        self.dropout = nn.Dropout(p=drop_prob)

        PE = torch.zeros((max_len, d))  # (L, d)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div = torch.exp(torch.arange(0., d, 2,)/d*math.log(1e4))  # (d/2)
        PE[:, ::2] = torch.sin(pos/div)  # (L, d/2)
        PE[:, 1::2] = torch.cos(pos/div)  # (L, d/2)
        self.PE = nn.Parameter(PE.unsqueeze(0), requires_grad=False)  # (1, L, d)

    def forward(self, x):
        """
        :param x: Input (batch, seq_len, d)
        :return: x + PE
        """
        return self.dropout(x + self.PE[0, x.shape[1]])


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 4 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob, use_transformer, **kwargs):
        super(BiDAFOutput, self).__init__()
        self.use_transformer = use_transformer

        self.att_linear_1 = nn.Linear(4 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(hidden_size, 1)

        if not use_transformer:
            self.enc = RNNEncoder(input_size=hidden_size,  # hidden_size= 2h
                                  hidden_size=int(hidden_size/2),  # hidden_size= 2h
                                  num_layers=1,
                                  drop_prob=drop_prob)
        else:
            heads = kwargs['heads']
            inter_size = kwargs['inter_size']
            self.enc = TransformerEncoder(heads=heads,
                input_size=hidden_size,  # hidden_size= h
                output_size=hidden_size,  # hidden_size= h
                inter_size=inter_size,
                drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(4 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        if not self.use_transformer:
            mod_2 = self.enc(mod, mask.sum(-1))
        else:
            mod_2 = self.enc(mod, mask)
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
