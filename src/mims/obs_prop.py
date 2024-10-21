from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot
from torch.nn import init
import math
from typing import Optional
from torch_geometric.typing import Adj, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter


class ObservationProgation(MessagePassing):
    _alpha: OptTensor

    def __init__(self,
                 d_feat: int,
                 num_nodes: int,
                 heads: int = 1,
                 concat: bool = True,
                 dropout: float = 0.,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.d_feat = d_feat
        self.num_nodes = num_nodes
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.lin_value = Linear(d_feat, heads * d_feat)

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weights: Tensor = None,
                edge_attr: OptTensor = None,
                ret_attn_weights: bool = False):
        """
        Args:
            x (Tensor): The input tensor of observations that will be propagated
                throughout the graph. Shape of (N, F).
        """

        assert x.shape[0] == self.num_nodes

        # Propagate x
        out = self.propagate(edge_index,
                             x=x,
                             edge_weights=edge_weights,
                             edge_attr=edge_attr)

        # Load calculated attention weights
        alpha = self._alpha_store
        self._alpha_store = None

        # Aggregate heads
        if self.concat:
            out = out.view(-1, self.heads * self.d_feat)
        else:
            out = out.mean(dim=1)

        if ret_attn_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self,
                x_i: Tensor,
                edge_weights: Tensor,
                edge_index_i: Tensor) -> Tensor:
        """
        Args:
            x_i (Tensor): Source nodes for each edge (E, F).
            edge_weights (Tensor): Weights of each edge (E).
            edge_index_i (Tensor): The index of each edge's source node (E).
        
        Returns:
            Tensor: A tensor with the applied attention weights (E, H, F).
        """
      
        alpha = edge_weights.unsqueeze(-1)
        self._alpha_store = alpha # use-beta == False

        # Emphasize the strongest edges and randomly cut (p*E) edges
        alpha = softmax(alpha, index=edge_index_i, num_nodes=self.num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Project the source nodes to multiple heads
        out = F.relu(self.lin_value(x_i)).view(-1, self.heads, self.d_feat)
    
        # Apply attention weights
        out = out * alpha.unsqueeze(1)

        return out

    def aggregate(self,
                  inputs: Tensor,
                  index: Tensor) -> Tensor:
        return scatter(inputs, index, dim=self.node_dim, dim_size=self.num_nodes,
                           reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.d_feat,
                                             self.d_feat, self.heads)
