import math

import torch
from torch import nn

from src import tensortypes as tt
from mimic.models.vector import MLP


class PositionalEncoding(nn.Module):
    """
    Apply positional encoding to embedded sequences.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Constructor for PositionalEncoding.

        Args:
            embed_dim (int): Dimensionality of the embedded token.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            max_len (int, optional): The maximum sequence length. Defaults to 5000.
        """

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2)
                             * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, emb_seqs: tt.BatSeqEmbTensor) -> tt.BatSeqEmbTensor:
        """
        Apply positional encoding to embedded sequences.

        Args:
            emb_seqs (BatSeqEmbTensor): The embedded sequences to add positional
                encoding information to.

        Returns:
            BatSeqEmbTensor: Positionally encoded embedded sequences.
        """

        emb_seqs = emb_seqs + self.pe[:, :emb_seqs.size(1)]
        return self.dropout(emb_seqs)


class SequentialEmbedder(nn.Module):
    """
    Embeds sequences of vectors.
    """

    def __init__(self,
                 feat_dim: int,
                 embed_dim: int,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 num_heads: int = 8):
        """
        Constructor for SequenceEmbedder.

        Args:
            feat_dim (int): Dimensionality of the feature vector for each timestep.
            embed_dim (int): Dimensionality for each embedded feature vector.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            max_len (int, optional): The maximum sequence length. Defaults to 5000.
            num_heads (int, optional): Number of heads in the transformer. Defaults to 8.
        """

        super().__init__()

        self.mlp = MLP(feat_dim,
                       [int((feat_dim+embed_dim)/2)],
                       embed_dim,
                       bias=True,
                       relu=True)

        self.pos_enc = PositionalEncoding(embed_dim, dropout, max_len)

        self.tran_enc_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, embed_dim*4, dropout, batch_first=True)
        self.tran_enc = nn.TransformerEncoder(self.tran_enc_layer, 1)

    def forward(self, seqs: tt.BatSeqFeatTensor, mask: tt.BatSeqTensor
                ) -> tuple[tt.BatSeqEmbTensor, tt.BatSeqTensor]:
        """
        Embed the sequence.

        Args:
            seqs (BatSeqFeatTensor): The batch of sequences each containing feature
                vectors.
            mask (BatSeqTensor): The mask for the passed in sequences.

        Returns:
            tuple[BatSeqEmbTensor, BatSeqTensor]: The embedded batch of sequences with
                each feature vector embedded and its corresponding mask.
        """
        pad_mask = (~mask).float()

        seqs = self.mlp(seqs)
        seqs = self.pos_enc(seqs)
        emb = self.tran_enc(seqs, src_key_padding_mask=pad_mask)

        return emb, mask
