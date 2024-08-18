import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from general import tensortypes as tt


class GraphConvolution(nn.Module):
    """
    Graph Convolution.

    This class is used to calcualte auxillary information for a certain modality for
    each sample. This information can serve as a replacement for the missing modality
    or compliment any existent information.

    The graph is fully connected, however, the strength of the connection is
    proportional to how similar two samples are. The stronger edges/connected nodes
    carry a greater influence on the computed auxillary information for the modality.
    """

    def __init__(self, embedded_dim: int, bias: bool = True):
        """
        Constructor for graph convolution.

        Args:
            embedded_dim (int): The embedded dimension.
            bias (bool, optional): Add bias to the linear projections.
                Defaults to True.
        """

        super().__init__()

        self.linear_layers = nn.ModuleList([nn.Linear(embedded_dim,
                                                      embedded_dim,
                                                      bias=bias) for _ in range(2)])

    def forward(self,
                emb: tt.EmbeddedStaticTensor,
                filt_sim_mat: tt.BatchSimilarityTensor) -> tt.EmbeddedStaticTensor:
        """
        Forward graph propagation.

        Args:
            emb (EmbeddedStaticTensor): The embedded representation of the modality for
                each sample.
            filt_sim_mat (BatchSimilarityTensor): The similarity matrix between samples
                across all modalities.

        Returns:
            EmbeddedStaticTensor: Computed auxillary information.
        """

        auxillary = F.relu(
            self.linear_layers[0](filt_sim_mat @ emb)
            )
        auxillary = F.relu(
            self.linear_layers[1](filt_sim_mat @ auxillary)
        )
        return auxillary


class PositionalEncoding(nn.Module):
    """
    Positional encoder.

    Encodes the sequential positions into the tensor.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Constructor for PositionalEncoding.

        Args:
            d_model (int): Dimensionality of the model.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            max_len (int, optional): Max length of the sequence. Defaults to 5000.
        """

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
        Add position information into sequence.

        Arguments:
            x Tensor: Sequential tensor of shape
                (batch_size x seq_len x embedding_dim).

        Returns:
            Tensor: Positionally encoded tensor.
        """

        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiModalTransformer(nn.Module):
    def __init__(self,
                 embedded_dim: int,
                 num_heads: Sequence[int],
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()

        self.pos_encode = PositionalEncoding(embedded_dim, dropout, max_len)

        self.num_trans = len(num_heads)
        self.num_heads = num_heads
        tran_enc_lays = [nn.TransformerEncoderLayer(d_model=embedded_dim,
                                                    nhead=nhead,
                                                    dim_feedforward=embedded_dim*4,
                                                    dropout=dropout,
                                                    activation='relu',
                                                    batch_first=True)
                         for nhead in num_heads]
        self.norm = nn.ModuleList([nn.LayerNorm(embedded_dim)
                                   for _ in range(self.num_trans)])
        self.tran_encs = nn.ModuleList([nn.TransformerEncoder(tran_enc_lay,
                                                              1,
                                                              norm=self.norm[tran_idx])
                                        for tran_idx, tran_enc_lay
                                        in enumerate(tran_enc_lays)])

    def forward(self,
                mm: tt.MultiModalTensor,
                mask: tt.MultiModalMaskTensor) -> tt.MultiModalTensor:
        mm = self.pos_encode(mm)

        for tran_idx in range(self.num_trans):
            mask_scaled = (~mask).repeat_interleave(self.num_heads[tran_idx], dim=0) \
                .float()
            mm = self.tran_encs[tran_idx](mm, mask=mask_scaled)
        return mm
