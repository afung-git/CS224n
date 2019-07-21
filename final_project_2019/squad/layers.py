"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
import math


class InitializedLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel=None, act=None, bias=False, use_dsc=False):
        """
        Module for Linear, Convolution, or Depthwise Separable Convolution (Chollet: https://arxiv.org/abs/1610.02357)
        Includes initialization depending on activation
        :param input_size: Input dimension
        :param output_size: Output dimension
        :param kernel: Convolution kernel size
        :param act: To use ReLU or not
        :param bias: To add bias or not (For DSC, only the pointwise)
        :param use_dsc: To use DSC or not
        """
        super(InitializedLayer, self).__init__()
        self.act = act
        self.ff = nn.Conv1d(input_size, output_size, kernel_size=kernel, padding=kernel//2, bias=bias) \
            if kernel and not use_dsc else nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel, padding=kernel//2, groups=input_size, bias=False),
            nn.Conv1d(input_size, output_size, 1, bias=bias)) if kernel and use_dsc \
            else nn.Linear(input_size, output_size, bias=bias)

        def weight_init(x):
            nn.init.kaiming_normal_(x, nonlinearity='relu') if act else nn.init.xavier_uniform_(x)

        [weight_init(ff.weight) for ff in self.ff] if use_dsc else weight_init(self.ff.weight)

        nn.init.constant_((self.ff[1].bias if use_dsc else self.ff.bias), 0.) if bias else None

    def forward(self, x):
        return self.act(self.ff(x)) if self.act else self.ff(x)


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        vectors (torch.Tensor): Pre-trained word & char vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
        char_limit (int): Maximum limit of characters per word
    """
    def __init__(self, vectors, c2w_size, hidden_size, drop_prob, char_limit):
        super(Embedding, self).__init__()
        word_vectors, char_vectors = vectors
        self.char_embed = nn.Embedding.from_pretrained(char_vectors)
        self.c2w_size = c2w_size
        self.char2word = nn.Sequential(InitializedLayer(char_vectors.shape[1], c2w_size, 5, act=F.relu, bias=True),
            nn.MaxPool1d(char_limit - 5 + 1))
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = InitializedLayer(word_vectors.shape[1] + c2w_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)
        self.drop_prob = drop_prob

    def forward(self, x):
        x_w, x_c = x
        w_emb = self.word_embed(x_w)  # (batch_size, seq_len, w_embed_size)

        c_emb = self.char_embed(x_c)  # (batch_size*seq_len, char_limit, c_embed_size)
        c_emb = c_emb.transpose(1, 2)  # (batch_size*seq_len, c_embed_size, char_limit)
        c2w_emb = self.char2word(c_emb)  # (batch_size*seq_len, w_embed_size)
        c2w_emb = c2w_emb.reshape(list(w_emb.shape[:-1]) + [self.c2w_size])  # (batch_size, seq_len, c2w_size)

        emb = torch.cat((w_emb, c2w_emb), dim=-1)  # (batch_size, seq_len, w_embed_size + c2w_size)
        emb = F.dropout(emb, self.drop_prob, self.training)

        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

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
        self.transforms = nn.ModuleList([InitializedLayer(hidden_size, hidden_size, act=F.relu, bias=True)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([InitializedLayer(hidden_size, hidden_size, act=torch.sigmoid, bias=True)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = gate(x)
            t = transform(x)
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
                 drop_prob=0.,
                 use_GRU=True):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        if use_GRU:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                               batch_first=True,
                               bidirectional=True,
                               dropout=drop_prob if num_layers > 1 else 0.)
        else:
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


def self_attention(query, key, value, mask=None, bias=None, dp=None):
    """
    :param query: Query tensor (batch x heads x seq_len x d_k)
    :param key: Key tensor (batch x heads x seq_len x d_k)
    :param value: Value tensor (batch x heads x seq_len x d_k)
    :param mask: Optional mask, same for all heads (batch x heads x seq_len x seq_len)
    :param bias: Bias (optional)
    :param dp: Dropout layer
    :return: output, scores (batch x heads x seq_len x d_k), (batch x heads x seq_len x seq_len)
    """
    logits = torch.matmul(query, key.transpose(-1, -2))/(key.shape[-1]**.5)
    if bias:
        logits += bias
    if mask is not None:
        logits = logits.masked_fill(mask == 0, -1e9)  # NOT 1e-9. Softmax(1e-9) is still 1.
    scores = F.softmax(logits, dim=-1)
    if dp is not None:
        scores = dp(scores)
    return torch.matmul(scores, value), scores


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads, hidden_size, drop_prob=.1):
        """
        :param heads: Number of attention heads to use
        :param hidden_size: Dimension of input/output vectors
        :param drop_prob: Dropout rate
        """
        super(MultiHeadSelfAttention, self).__init__()

        assert hidden_size % heads == 0, "hidden_size not a multiple of heads"

        self.d_k = hidden_size // heads
        self.heads = heads
        self.Linears = nn.ModuleList([InitializedLayer(hidden_size, hidden_size) for _ in range(4)])

        self.bias = nn.Parameter(torch.zeros(1))

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
        o, self.attn = self_attention(q, k, v, mask, self.bias, self.dropout)  # (batch_size, heads, seq_len, d_k)
        o = o.transpose(1, 2).contiguous().view(batch_size, -1, self.heads*self.d_k)

        return self.Linears[-1](o)


class FeedForward(nn.Module):
    def __init__(self, input_size, output_size, inter_size, drop_prob=.1):
        """
        :param input_size: Dimension of input vectors
        :param output_size: Dimension of output vectors
        :param inter_size: Dimension of intermediate vectors
        :param drop_prob: Dropout rate
        """
        super(FeedForward, self).__init__()
        self.FF = nn.Sequential(
            InitializedLayer(input_size, inter_size, act=F.relu, bias=True),
            nn.Dropout(p=drop_prob),
            InitializedLayer(inter_size, output_size, bias=True)
        )

    def forward(self, x):
        return self.FF(x)


class Sublayer(nn.Module):
    def __init__(self, size, layer_list, drop_prob=.1):
        """
        :param size: Size of input to Layernorm
        :param drop_prob: Dropout rate
        """
        super(Sublayer, self).__init__()
        layer_list[0] = layer_list[0] + 1
        self.id = layer_list[0]  # 1st layer id = 1
        self.dropout = nn.Dropout(p=drop_prob if self.id % 2 == 1 else 0.)  # Alternate layers have dropout
        self.norm = nn.LayerNorm(size)  # Only Norm the last dimension (seq_len differs from batch to batch)

    def forward(self, x, sub, swap=False):
        """
        :param x: Input (batch x seq_len x hidden_size)
        :param sub: Sublayer (Feedforward, MultiHeadSelfAttention etc.)
        :param swap: Flag to transpose layers (especially for conv)
        :return: Normalize, Sublayer, Dropout
        """
        return self.dropout(sub(self.norm(x).transpose(1, 2)).transpose(1, 2) if swap else
                            sub(self.norm(x)))


def make_mask(masks, decode=False):
    """
    :param masks: 0 for pad, 1 for non-pad (batch x seq_len)
    :param decode: decoders are Auto-Regressive (can't see future words)
    :return: mask: (batch x seq_len x seq_len / batch x 1 x seq_len)
    """
    masks = masks.unsqueeze(-2)  # Pad words should not be zeroed across their whole rows
    if decode:
        masks = masks & torch.from_numpy(np.tril(np.ones(masks.shape[-1]))).byte()
    return masks.long()


class TransformerEncoder(nn.Module):
    def __init__(self, heads, input_size, output_size, inter_size, num_conv, layers_count, drop_prob=.1, p_sdd=.9):
        """
        :param heads: Number of heads in multi-layer self-attention
        :param input_size: Input hidden state size
        :param output_size: Output hidden state size
        :param inter_size: Dimension of intermediate feed-forward layers
        :param num_conv: Number of convolutional layers before MHSA
        :param drop_prob: Dropout rate
        """
        super(TransformerEncoder, self).__init__()
        self.p_sdd = p_sdd
        self.PE = PositionalEncodings(input_size, drop_prob)
        self.convs = nn.ModuleList([InitializedLayer(input_size, input_size, 7, act=F.relu, bias=True, use_dsc=True)
                                    for _ in range(num_conv)])
        self.MHSA = MultiHeadSelfAttention(heads, input_size, drop_prob)
        self.FF = FeedForward(input_size, output_size, inter_size, drop_prob)

        self.layers = nn.ModuleList([Sublayer(input_size, layers_count, drop_prob) for _ in range(2 + num_conv)])
        self.layers_count = layers_count

        # match in/output shapes
        self.morph = InitializedLayer(input_size, output_size, bias=True) if input_size != output_size else None

    def sdd(self, l, x, args):
        """
        Perform Stochastic Depth Dropout
        :param l: Sublayer object
        :param x: Residual input
        :param args: inputs to Sublayer, including operation type
        :return: x or x + l(args)
        """
        if self.training:
            L = self.layers_count[0]
            p = 1. - (l.id/L) * (1. - self.p_sdd)  # min 0.9. doesn't dropout much
            b = torch.bernoulli(torch.tensor(p)) == 0
            return x if b else x + l(*args)
        else:
            return x

    def forward(self, x, masks):
        """
        :param x: Input (batch, seq_len, hidden_size)
        :param masks: (batch, seq_len)
        :return: o: Output (batch, seq_len, hidden_size)
        """
        x = self.PE(x)

        for i, conv in enumerate(self.convs):
            x = self.sdd(self.layers[i], x, [x, conv, True])

        x = self.sdd(self.layers[len(self.convs)],
                     x, [x, lambda y: self.MHSA(y, y, y, make_mask(masks))])  # Need lambda due to mask

        return self.sdd(self.layers[len(self.convs)+1],
                        x if self.morph is None else self.morph(x), [x, self.FF])


class PositionalEncodings(nn.Module):
    def __init__(self, d, drop_prob=.1, max_len=5000):
        """
        :param d: Dimension of embedding
        :param drop_prob: Dropout Rate
        :param max_len: Maximum length of a sequence
        """
        super(PositionalEncodings, self).__init__()
        self.dropout = nn.Dropout(p=drop_prob)

        PE = torch.zeros((max_len, d))  # (L, d)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div = torch.exp(torch.arange(0., d, 2)/d*math.log(1e4))  # (d/2)
        PE[:, ::2] = torch.sin(pos/div)  # (L, d/2)
        PE[:, 1::2] = torch.cos(pos/div)  # (L, d/2)
        self.register_buffer('PE', PE)  # (L, d)

    def forward(self, x):
        """
        :param x: Input (batch, seq_len, d)
        :return: x + PE
        """
        return self.dropout(x + self.PE[:x.shape[1]])  # You added the same PE sinusoid to all positions


class TransformerEncoderStack(nn.Module):
    def __init__(self, N, heads, input_size, output_size, inter_size, num_conv, drop_prob=.1, p_sdd=.9):
        """
        :param layers: Transformer layer/block
        :param N: Number of layers/blocks to stack
        """
        super(TransformerEncoderStack, self).__init__()
        self.layers_count = [0]  # A list with a single element. A counter for (sub)layers
        self.layers = nn.ModuleList([TransformerEncoder(heads=heads,
                                    input_size=input_size,
                                    output_size=input_size,
                                    inter_size=inter_size,
                                    num_conv=num_conv,
                                    layers_count=self.layers_count,
                                    drop_prob=drop_prob, p_sdd=p_sdd) for _ in range(N-1)])
        self.last = TransformerEncoder(heads=heads,
                                       input_size=input_size,
                                       output_size=output_size,
                                       inter_size=inter_size,
                                       num_conv=num_conv,
                                       layers_count=self.layers_count,
                                       drop_prob=drop_prob, p_sdd=p_sdd)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x, mask=None):
        """
        :param x: Input (batch x seq_len x input_size)
        :param mask: mask (batch x seq_len)
        :return: Output (batch x seq_len x output_size)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(self.last(x, mask))


def QAModel(modelstack, x, mask):
    """
    Perform QANet modelling
    :param modelstack: Transformer Encoder Stack
    :param x: Input tensor (B x L x H) from BiDAF Attention
    :param mask: Mask for pads
    :return: start=[M1, M2], end=[M1, M3] (B x L x 2H)
    """
    M1 = modelstack(x, mask)
    M2 = modelstack(M1, mask)
    M3 = modelstack(M2, mask)
    return torch.cat((M1, M2), dim=2), torch.cat((M1, M3), dim=2)


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
    def __init__(self, hidden_size, drop_prob=.1):
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
            self.enc = TransformerEncoderStack(N=1,
                heads=heads,
                input_size=hidden_size,  # hidden_size= h
                output_size=hidden_size,  # hidden_size= h
                inter_size=inter_size,
                drop_prob=.1)

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


class QAOutput(nn.Module):
    """
    QANet Output layer. Takes a start tensor and end tensor,
    Feeds each into a Feed-Forward layer, then Masked-Softmax.
    Returns probability of start and end positions
    """
    def __init__(self, input_size):
        super(QAOutput, self).__init__()
        self.startFF = InitializedLayer(input_size, 1, bias=True)
        self.endFF = InitializedLayer(input_size, 1, bias=True)

    def forward(self, start, end, mask):
        logits_1 = self.startFF(start)
        logits_2 = self.startFF(end)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
