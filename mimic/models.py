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
        pad_mask = ~mask

        x = self.mlp(x)
        x = self.pos_enc(x)
        x = self.tran_enc(x, src_key_padding_mask=pad_mask)
        return x, mask

### --- Static Notes Modality --- ###


SEP_TOKEN = '[sep]'


class NLEmbedder(nn.Module):
    def __init__(self, vocab, embed_dim, model_dim, tran_heads, tran_dff, vocab_size=10000, dropout=0.3):
        super(NLEmbedder, self).__init__()

        self.word_embed = WordEmbedder(vocab, vocab_size, embed_dim, model_dim)
        self.pos_encode = PositionalEncoding(model_dim, dropout)
        self.enc_layer = nn.TransformerEncoderLayer(
            model_dim, tran_heads, tran_dff, dropout, batch_first=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, mask: Tensor):
        pad_mask = ~mask

        x = self.word_embed(x)
        x = self.pos_encode(x)
        x = self.enc_layer(x, src_key_padding_mask=pad_mask)
        x = x.nan_to_num()
        x = self.avg_pool(x.transpose(1, 2)).squeeze()

        mask = (mask.int().sum(axis=-1) > 0).bool().unsqueeze(-1)

        return x, mask


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

    def forward(self, x: Tensor, mask: Tensor):
        # x: BxTxS, m: BxTxS
        batch_size, time_steps, seq_len = x.shape

        x = x.view((batch_size*time_steps, -1))
        mask_nle = mask.view((batch_size*time_steps, -1))

        x, mask_nle = self.nl_embedder(x, mask_nle)

        x = x.view((batch_size, time_steps, -1))
        mask = (mask.int().sum(axis=2) > 0).bool()

        return x, mask
