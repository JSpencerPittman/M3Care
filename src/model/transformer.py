from src.embed.char_embed import CharacterEmbedding
from typing import List, Tuple
import torch.nn as nn
import torch
import copy

from src.component.attention import MultiHeadedAttention
from src.component.layer import PositionwiseFeedForward, SublayerConnection
from src.embed.pos_encode import PositionalEncoding
from src.component.encoder import Encoder, EncoderLayer


class NatLangTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, devicename, dropout_rate=0.2):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NatLangTransformer, self).__init__()
        self.model_embeddings = CharacterEmbedding(
            vocab, 50, hidden_size, 5, 0.3)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.devicename = devicename

        c = copy.deepcopy
        attn = MultiHeadedAttention(8, self.hidden_size)
        ff = PositionwiseFeedForward(
            self.hidden_size, self.hidden_size*4, self.dropout_rate)
        self.position = PositionalEncoding(embed_size, dropout_rate)
        self.encoder = Encoder(EncoderLayer(
            hidden_size, c(attn), c(ff), dropout_rate), 1)

        self.opt = nn.Linear(
            in_features=(hidden_size), out_features=1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, source: List[List[str]]) -> torch.Tensor:

        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        total_src_padded = self.vocab.to_input_tensor(
            source, device=self.devicename)   # Tensor: (src_len, b)

        enc_hiddens, first_hidden = self.encode(
            total_src_padded)

        return enc_hiddens, source_lengths

    def encode(self, source_padded: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """

        enc_hiddens, _ = None, None

        source_padded = source_padded.permute(1, 0)  # b t
        src_mask = (source_padded != 0).unsqueeze(-2)  # b t 1

        X = self.model_embeddings(source_padded)

        enc_hiddens = self.encoder(X, src_mask)  # b t h
        first_hidden = enc_hiddens[:, 0, :]

        return enc_hiddens, first_hidden


class MultiModalTransformer(nn.Module):
    def __init__(self, input_dim, d_model,  MHD_num_head, d_ff, output_dim, keep_prob=0.5):
        super(MultiModalEncoder, self).__init__()

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
