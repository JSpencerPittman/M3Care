from torch import nn, Tensor
import torch


class PositionalEncodingTF(nn.Module):
    """
    Encodes timestamps according to method in the appendix A.1 of RAINDROP.

    frequency(f) = 10^4
    expected dimensions (eps)
    p^t_{2k} = sin(t/f^(2k/eps))
    p^t_{2k+1} = cos(t/f^(2k/eps))
    """

    def __init__(self, pe_dim: int, max_timesteps: int = 500):
        """
        Constructor for RaindropPositionalEncoder.

        Args:
            pe_dim (int): The dimensions of the model or the embedding dimension.
            max_timesteps (int, optional): The max number of timesteps. Defaults to 500.
        """

        super().__init__()
        self.pe_dim = pe_dim
        self.max_timesteps = max_timesteps

        self.timescales = max_timesteps ** torch.linspace(0, 1, self.pe_dim//2).cuda()

    def forward(self, times: Tensor) -> Tensor:
        """
        Encodes the provided timestamps

        Args:
            times (Tensor): timestamps where each timestamp is a continuous
                value (T, B).

        Returns:
            Tensor: Encoded timestamps (T, B).
        """

        scaled_times = times.unsqueeze(-1) / self.timescales[None, None, :]
        return torch.concat([torch.sin(scaled_times), torch.cos(scaled_times)], dim=-1)
