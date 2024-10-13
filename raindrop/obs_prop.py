import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax


class ObservationPropagation(MessagePassing):
    _alpha_store: OptTensor

    def __init__(self,
                 d_inp: int,
                 d_out: int,
                 num_heads: int = 1,
                 d_edge: Optional[int] = None,
                 dropout: float = 0.,
                 concat_heads: bool = True,
                 use_skip: bool = True,
                 use_beta: bool = False,
                 bias: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ObservationPropagation, self).__init__(node_dim=0, **kwargs)

        self.d_inp = d_inp
        self.d_out = d_out
        self.num_heads = num_heads
        self.d_edge = d_edge
        self.dropout = nn.Dropout(dropout)
        self.concat_heads = concat_heads
        self.use_skip = use_skip
        self.use_beta = use_beta
        self.bias = bias

        self.lin_key = nn.Linear(d_inp, num_heads * d_out)
        self.lin_query = nn.Linear(d_inp, num_heads * d_out)
        self.lin_value = nn.Linear(d_inp, num_heads * d_out)

        self.use_edge = d_edge is not None
        if self.use_edge:
            self.lin_edge = nn.Linear(d_edge, num_heads * d_out, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if self.use_skip:
            self.lin_skip = nn.Linear(d_inp,
                                      d_out * (num_heads if self.concat_heads else 1),
                                      bias=bias)
        else:
            self.lin_skip = self.register_parameter('lin_skip', None)

        if self.use_beta:
            assert self.use_skip
            self.lin_beta = \
                nn.Linear(3 * d_out * (num_heads if self.concat_heads else 1),
                          1,
                          bias=False)
        else:
            self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_attr: OptTensor = None,
                ret_attn_weights: bool = False):
        # x (N, I)

        if self.use_edge:
            assert edge_attr is not None

        # out (N, H, O)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        alpha = self._alpha_store
        self._alpha_store = None

        if self.concat_heads:
            # out (N, HO)
            out = out.view(-1, self.num_heads * self.d_out)
        else:
            # out (N, O)
            out = out.mean(dim=1)

        if self.use_skip:
            # x_skip (N, O | OH)
            x_skip = self.lin_skip(x)

            if self.use_beta:
                beta = self.lin_beta(torch.cat([out, x_skip, out - x_skip], dim=-1))
                beta = F.sigmoid(beta)
                out = beta * x_skip + (1 - beta) * out
            else:
                out = out + x_skip

        if ret_attn_weights:
            return out, (edge_index, alpha)
        else:
            return out

    def message(self,
                x_i: Tensor,
                x_j: Tensor,
                edge_weights: OptTensor,
                edge_attr: OptTensor,
                index: Tensor,
                ptr: OptTensor,
                size_i: Optional[int]
                ) -> Tensor:
        # x_i (E, I)
        # x_j (E, I)
        # edge_attr (E, D_edge)

        # query (E, H, O)
        query = self.lin_query(x_i).view(-1, self.num_heads, self.d_out)
        # key (E, H, O)
        key = self.lin_key(x_j).view(-1, self.num_heads, self.d_out)
        # out (E, H, O)
        value = self.lin_value(x_j).view(-1, self.num_heads, self.d_out)

        edge: OptTensor = None
        if self.use_edge:
            # edge (E, H, O)
            edge = self.lin_edge(edge_attr).view(-1, self.num_heads, self.d_out)
            key = key + edge
            value = value + edge

        # alpha (E, H)
        alpha = (query @ key.T) / math.sqrt(self.d_out)
        if edge_weights is not None:
            alpha = alpha * edge_weights.unsqueeze(-1)

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha_store = alpha
        alpha = self.dropout(alpha)

        # (E, H, O)
        return value * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
