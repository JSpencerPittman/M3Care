import torch
from torch import nn
from torch.nn import functional as F

from raindrop import tensortypes as tt
from raindrop.models import RaindropPositionalEncoder

# MASK-> 1:Present, 0:Missing

class Raindrop(nn.Module):
    def __init__(self,
                 num_sensors: int,
                 obs_dim: int,
                 obs_embed_dim: int,
                 pe_emb_dim: int,
                 dropout: float = 0.3,
                 init_range: float = 1e-10,
                 device: str = 'cpu'):
        super().__init__()

        # Save parameters
        self.num_sensors = num_sensors
        self.obs_dim = obs_dim
        
        self.obs_embed_dim = obs_embed_dim
        self.pe_emb_dim = pe_emb_dim
        
        self.device = device

        # State tracking
        self.first: bool = True

        # General modules
        self.dropout = nn.Dropout(dropout)

        # Observation embedding
        self.obs_emb_weights = nn.Parameter(
            torch.zeros(num_sensors, obs_dim, obs_embed_dim)
            )
        self.pos_encoder = RaindropPositionalEncoder(pe_emb_dim, device=device)

        # Inter-sensor attention
        self.inter_sensor_attn_dim = 10
        self.inter_sensor_attn_weights = nn.Parameter(
            torch.zeros(num_sensors, self.inter_sensor_attn_dim)
        )
        self.inter_sensor_recv_map = nn.Linear(self.obs_embed_dim,
                                               self.inter_sensor_attn_dim + pe_emb_dim)
        self.inter_sensor_bidir_weights = nn.Parameter(
            torch.zeros(self.num_sensors, self.obs_embed_dim)
                                                      )

        # self._init_weights(init_range)

    def forward(self,
                x: tt.BatTimeSenObsTensor,
                times: tt.BatTimeTensor,
                mask: tt.BatTimeSenTensor):
        batch_size, time_steps, _, _ = x.shape

        # Exclude batch_size
        h = self._embed_observation(x)  # (B, T, Se, ObsEmb)
        h = h * mask
        pe = self.pos_encoder(times)  # (B, T, Pe)

        alpha = self._calc_inter_sensor_attn_weights(h, pe)  # (B, T, Se, Se)
        adj_graph = self._init_adjacency_graph(batch_size)  # (B, Se, Se)

    def _embed_observation(self, x: tt.BatTimeSenObsTensor
                           ) -> tt.BatTimeSenObs_EmbTensor:
        return F.relu(torch.einsum("ijkl,klm->ijkm", x, self.obs_emb_weights))

    def _calc_inter_sensor_attn_weights(self,
                                        h: tt.BatTimeSenObs_EmbTensor,
                                        pe: tt.BatTimePeTensor,
                                        ) -> tt.BatTimeSenSenTensor:
        batch_size, time_steps, _, _ = h.shape
        pe = pe.unsqueeze(2).repeat(1, 1, self.num_sensors, 1)
        attn_weights = \
            self.inter_sensor_attn_weights[None, None, :, :].repeat(batch_size,
                                                                    time_steps,
                                                                    1, 1)
        alpha = torch.concat([attn_weights, pe], dim=-1)  # (B, T, Se, d_r+d_t)
        h = self.inter_sensor_recv_map(h)  # (B, T, Se, d_r+d_t)

        alpha = h @ alpha.T  # (B, T, Se, Se)
        return F.relu(alpha)

    def _propagate_message(self,
                           h: tt.BatTimeSenObs_EmbTensor,
                           alpha: tt.BatTimeSenSenTensor,
                           adj: tt.BatSenSenTensor):
        batch_size, time_steps, _, _ = h.shape
        bidir = \
            (self.inter_sensor_bidir_weights @ self.inter_sensor_bidir_weights.T)
        # (B, T, Se, Se)
        bidir = bidir[None, :, :].repeat(batch_size, 1, 1)
        adj = adj.unsqueeze(1).repeat(1, time_steps, 1, 1)  # (B, T, Se, Se)
        

    def _init_adjacency_graph(self, batch_size: int) -> tt.BatSenSenTensor:
        return torch.ones(batch_size,
                          self.num_sensors,
                          self.num_sensors,
                          device=self.device)

    # def _init_weights(self, init_range: float = 1e-10):
    #     # Observation embedding
    #     nn.init.uniform_(self.obs_emb_weights, -init_range, init_range)
