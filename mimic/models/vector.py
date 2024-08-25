from torch import nn, Tensor
from typing import Optional


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: list[int],
                 out_dim: int,
                 bias: bool = False,
                 relu: bool = False,
                 norm: bool = False):
        """
        Creates a multi-layer perceptron neural newtwork.

        Args:
            in_dim (int): Input vector's dimensionality, I.
            hidden_dim (list[int]): Number of neurons in each hidden layer.
            out_dim (int): Output vector's dimensionality, O.
            bias (bool, optional): Apply bias to each layer. Defaults to False.
            relu (bool, optional): Use the relu activation function after each layer.
                Defaults to False.
            norm (bool, optional): Apply normalization after each layer. Defaults to
                False.
        """

        super().__init__()

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

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # X: I -> O
        x = self.model(x)
        if mask is None:
            return x

        x *= mask.unsqueeze(-1)
        return x, mask
