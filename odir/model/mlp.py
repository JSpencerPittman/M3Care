from typing import List
import torch.nn as nn


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
