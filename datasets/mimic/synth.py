from comp import (WordEmbedder, PositionalEncoding,
                  Time2Vec, MultiHeadedAttention,
                  PositionwiseFeedForward, SublayerConnection)
from torch import nn
import torch


class NLEmbedder(nn.Module):
    def __init__(self, vocab, embed_dim, model_dim, tran_heads, tran_dff, vocab_size=10000, dropout=0.3):
        super(NLEmbedder, self).__init__()

        self.word_embed = WordEmbedder(vocab, vocab_size, embed_dim, model_dim)
        self.pos_encode = PositionalEncoding(model_dim, dropout)
        self.enc_layer = nn.TransformerEncoderLayer(
            model_dim, tran_heads, tran_dff, dropout, batch_first=True)

    def forward(self, x):
        x = self.word_embed(x)
        x = self.pos_encode(x)
        x = self.enc_layer(x)
        return x.mean(dim=1)


class TimeSeriesNLEmbedder(nn.Module):
    def __init__(self, vocab, embed_dim, model_dim, tran_heads, tran_dff, vocab_size=10000, dropout=0.3):
        super(TimeSeriesNLEmbedder, self).__init__()

        self.nl_embedder = NLEmbedder(
            vocab, embed_dim, model_dim, tran_heads, tran_dff, vocab_size, dropout)

        self.time2vec = Time2Vec()

    def forward(self, times, cats, notes):
        notes = self.nl_embedder(notes)

        times = self.time2vec(times)

        return torch.cat([times.unsqueeze(1), cats, notes], dim=1)


class MultiModalTransformer(nn.Module):
    def __init__(self, input_dim, d_model,  MHD_num_head, d_ff, output_dim, keep_prob=0.5):
        super(MultiModalTransformer, self).__init__()

        # hyperparameters
        self.input_dim = input_dim
        self.d_model = d_model
        self.MHD_num_head = MHD_num_head
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.keep_prob = keep_prob

        # layers
        self.embed = nn.Linear(self.input_dim, self.d_model)

        self.PositionalEncoding = PositionalEncoding(
            self.d_model, dropout=0, max_len=5000)

        self.MultiHeadedAttention = MultiHeadedAttention(
            self.MHD_num_head, self.d_model)
        self.SublayerConnection = SublayerConnection(
            self.d_model, dropout=1 - self.keep_prob)

        self.PositionwiseFeedForward = PositionwiseFeedForward(
            self.d_model, self.d_ff, dropout=0.1)
        self.output = nn.Linear(self.d_model, self.output_dim)

        self.dropout = nn.Dropout(p=1 - self.keep_prob)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input, mask):
        # input shape [batch_size, timestep, feature_dim]
        feature_dim = input.size(2)
        assert (feature_dim == self.input_dim)
        assert (self.d_model % self.MHD_num_head == 0)

        input = self.embed(input)
        input = self.PositionalEncoding(input)  # b t d_model

        contexts = self.SublayerConnection(input, lambda x: self.MultiHeadedAttention(
            input, input, input, mask))  # b t d_model

        contexts = self.SublayerConnection(
            contexts, lambda x: self.PositionwiseFeedForward(contexts))  # b t d_model

        return contexts  # b t h
