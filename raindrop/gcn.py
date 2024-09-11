import torch
from torch import nn
from torch_geometric.nn import MessagePassing  # type:ignore
from torch_geometric.utils import add_self_loops, degree
from tensortypes import TensorType


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))

    def forward(self, x, edge_index):
        # x: (NxI)
        # edge_index: (2xE)

        # (2x(E+N))
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # NxO
        x = self.lin(x)

        # (E+N), (E+N)
        row, col = edge_index
        # (N)
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # (2x(E+N))
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)    
        out = out + self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1)