from torch.nn import functional as F
from torch import (nn, Tensor)
from typing import List
import torch
import math

### --- Static Tabular Modality --- ###


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

    def forward(self, x: Tensor):
        return self.model(x)

### --- Time-Series Tabular Modality --- ###


class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 nhead: int = 8):
        super(TimeSeriesTransformer, self).__init__()

        self.mlp = MLP(d_input, [int((d_input+d_model)/2)],
                       d_model, bias=True, relu=True)

        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)

        self.tran_enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model*4, dropout, batch_first=True)
        self.tran_enc = nn.TransformerEncoder(self.tran_enc_layer, 1)

    def forward(self, x: Tensor, mask: Tensor):
        pad_mask = ~mask.bool()

        x = self.mlp(x)
        x = self.pos_enc(x)
        x = self.tran_enc(x, src_key_padding_mask=pad_mask)
        return x

### --- Static Notes Modality --- ###


SEP_TOKEN = '[SEP]'


class NLEmbedder(nn.Module):
    def __init__(self, vocab, embed_dim, model_dim, tran_heads, tran_dff, vocab_size=10000, dropout=0.3):
        super(NLEmbedder, self).__init__()

        self.word_embed = WordEmbedder(vocab, vocab_size, embed_dim, model_dim)
        self.pos_encode = PositionalEncoding(model_dim, dropout)
        self.enc_layer = nn.TransformerEncoderLayer(
            model_dim, tran_heads, tran_dff, dropout, batch_first=True)

    def forward(self, x, msk=None):
        x = self.word_embed(x)
        x = self.pos_encode(x)
        x = self.enc_layer(x)
        x = x.mean(dim=1)
        if msk is not None:
            x = self.fill_to_mask(x, msk)
        return x

    def fill_to_mask(self, x: Tensor, msk: Tensor):
        batch_size, feat_size = msk.size(0), x.size(-1)
        filled = torch.zeros((batch_size, feat_size),
                             dtype=x.dtype).to(next(self.parameters()).device)

        f_idx = 0
        for idx, m in enumerate(msk):
            if not m.item():
                filled[idx] = x[f_idx]
                f_idx += 1

        return filled


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


### --- Time-Series Notes Modality --- ###


class TimeSeriesNLEmbedder(nn.Module):
    def __init__(self, vocab, embed_dim, model_dim, tran_heads, tran_dff, vocab_size=10000, dropout=0.3):
        super(TimeSeriesNLEmbedder, self).__init__()

        self.nl_embedder = NLEmbedder(
            vocab, embed_dim, model_dim, tran_heads, tran_dff, vocab_size, dropout)

        self.time2vec = Time2Vec()

    # def forward(self, times, cats, notes):
    def forward(self, x, msk):
        for pat_x in x:
            emb_x = self.forward_patient(pat_x)

    def forward_patient(self, x):
        times, cats, notes = x

        times = torch.from_numpy(times).to(next(self.parameters()).device)
        cats = torch.from_numpy(cats).to(next(self.parameters()).device)
        notes = torch.from_numpy(notes).to(next(self.parameters()).device)

        notes = self.nl_embedder(notes)

        times = self.time2vec(times)

        return torch.cat([times.unsqueeze(1), cats, notes], dim=1)


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
