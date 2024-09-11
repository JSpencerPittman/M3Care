import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn.conv import MessagePassing  # type:ignore
from torch_geometric.typing import Adj  # type:ignore
from torch_geometric.utils import softmax  # type:ignore
from torch_scatter import scatter

from raindrop import tensortypes as tt


class ObservationPropagation(MessagePassing):
    def __init__(self,
                 num_sensors: int,
                 obs_dim: int,
                 time_steps: int,
                 num_heads: int,
                 out_dim: int,
                 dropout: float = 0.3):
        self.num_sensors = num_sensors
        self.obs_dim = obs_dim
        self.time_steps = time_steps
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.dropout = dropout

        assert out_dim % num_heads == 0
        self.head_dim = out_dim / num_heads

        self.lin_key = nn.Linear(obs_dim, out_dim)
        self.lin_query = nn.Linear(obs_dim, out_dim)
        self.lin_value = nn.Linear(obs_dim, out_dim)
        self.s = nn.Linear(time_steps, 1)

    def forward(self, x: tt.TimeSenObsTensor, edge_index: Adj, pe: tt.TimePeTensor):
        x = x.permute(1, 0, 2)  # (Se x T x Obs)
        x = (x, x)

        out = self.propagate(edge_index, x=x, size=None)

        return out

    def message(self,
                x_i: tt.EdgeTimeObsTensor,
                x_j: tt.EdgeTimeObsTensor,
                index: Tensor) -> tt.EdgeTimeTensor:
        # (E, H, T, d_head)
        key = self.lin_key(x_j).view(*x_j.shape[-1:],
                                     self.num_heads,
                                     self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # (E, H, T, d_head)
        query = self.lin_query(x_i).view(*x_i.shape[-1:],
                                         self.num_heads,
                                         self.head_dim)
        query = query.permute(0, 2, 1, 3)

        # (E, H, T, T)
        beta = (query @ key.T) / math.sqrt(self.out_dim)
        # (E, H, T)
        beta = self.s(beta).squeeze()
        beta = softmax(beta, index)
        beta = F.dropout(beta, p=self.dropout, training=self.training)

        return beta

    def aggregate(self, inputs: tt.EdgeTimeTensor, index: Tensor):
        return scatter(inputs, index, reduce=self.aggr)
