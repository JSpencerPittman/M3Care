from torch import nn, Tensor
from typing import List
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


class IrregularTimeNLP(nn.Module):
    def __init__(self, vocab, embed_dim, model_dim, tran_heads, tran_dff, vocab_size=10000, dropout=0.3):
        super(IrregularTimeNLP, self).__init__()

        self.word_embed = WordEmbedder(vocab, vocab_size, embed_dim, model_dim)
        self.pos_encode = PositionalEncoding(model_dim, dropout)
        self.enc_layer = nn.TransformerEncoderLayer(
            model_dim, tran_heads, tran_dff, dropout, batch_first=True)

        self.time2vec = Time2Vec()

    def forward(self, times, cats, notes):
        notes = self.word_embed(notes)
        notes = self.pos_encode(notes)
        notes = self.enc_layer(notes)
        notes = notes.mean(dim=1)

        times = self.time2vec(times)

        return torch.cat([times.unsqueeze(1), cats, notes], dim=1)


class NLEmbedder(nn.Module):
    def __init__(self):
        self.char_embed = CharacterEmbedding()
        self.pos_encode = PositionalEncoding()

        tran_enc_lay = nn.TransformerEncoderLayer()
        self.tran_enc = nn.TransformerEncoder()

    def forward(self, x):
        x = self.char_embed(x)
        x = self.pos_encode(x)
        return self.tran_enc(x)
