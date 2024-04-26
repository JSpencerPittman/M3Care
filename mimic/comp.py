from torch import nn, Tensor
from typing import List
from utils import clones
import torch.nn.functional as F
import torch
import math

SEP_TOKEN = '[SEP]'


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int,
                 hidden_dim: List[int],
                 out_dim: int,
                 bias: bool = False,
                 relu: bool = False,
                 norm: bool = False):
        super(MLP, self).__init__()

        self.model = nn.Sequential()
        for lay_idx in range(len(hidden_dim)+1):
            lay_in = in_dim if lay_idx == 0 else hidden_dim[lay_idx-1]
            lay_out = out_dim if lay_idx == len(
                hidden_dim) else hidden_dim[lay_idx]
            self.model.append(nn.Linear(lay_in, lay_out, bias=bias))

            if relu:
                self.model.append(nn.ReLU())

            if norm:
                self.model.append(nn.BatchNorm1d(lay_out))

    def forward(self, x):
        return self.model(x)


class WordEmbedder(nn.Module):
    def __init__(self, vocab, vocab_size, embed_dim, d_model):
        super(WordEmbedder, self).__init__()

        # Create idx -> embed_idx mapping
        top_words = vocab.top_tokens(vocab_size+1)
        top_words.remove(SEP_TOKEN)
        self.emb_idx_map = {
            vocab.tok2id[tok]: idx for idx, tok in enumerate(top_words, start=1)}
        self.sep_tok_id = vocab.tok2id[SEP_TOKEN]
        self.vocab_size = vocab_size

        # SEP[0] VOCAB[1,V] UNKNOWN[V+1] PADDING[V+2]
        self.embedder = nn.Embedding(vocab_size+3, embed_dim)
        self.linear = nn.Linear(embed_dim, d_model)

    def forward(self, x: Tensor):
        x = torch.LongTensor([[self.embed_idx(idx) for idx in seq]
                             for seq in x]).to(self.embedder.weight.device)
        x = self.embedder(x)
        return F.relu(self.linear(x))

    def embed_idx(self, idx):
        if idx == self.sep_tok_id:
            return 0
        elif idx in self.emb_idx_map:
            return self.emb_idx_map[idx]
        elif idx == 0:
            return self.vocab_size+2
        else:
            return self.vocab_size+1


class Time2Vec(nn.Module):
    def __init__(self, max_len=5000):
        super(Time2Vec, self).__init__()

        self.omega = nn.Parameter(torch.randn(max_len))
        self.phi = nn.Parameter(torch.randn(max_len))

    def forward(self, tau):
        seq_len = tau.size(0)
        zero_start = bool(tau[0] == 0)
        tau = (tau*self.omega[:seq_len]) + self.phi[:seq_len]
        tau[zero_start:] = torch.sin(tau[zero_start:])
        return tau


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(
            in_features, out_features).float())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).float())
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj, x):
        y = torch.mm(x.float(), self.weight.float())
        output = torch.mm(adj.float(), y.float())
        if self.bias is not None:
            return output + self.bias.float()
        else:
            return output


def attention_head(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        print("Q, K, V found")
        print(f"Q {query.shape} K {key.shape} V {value.shape}")

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention_head(query, key, value, mask=mask,
                                      dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    "Construct a layernorm module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
