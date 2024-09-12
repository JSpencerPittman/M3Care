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
                 timesteps: int,
                 out_dim: int,
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

        # Temporal self-attention
        self.temp_attn_dim = 10
        self.temp_attn_query_map = nn.Linear(obs_embed_dim + pe_emb_dim,
                                             self.temp_attn_dim)
        self.temp_attn_key_map = nn.Linear(obs_embed_dim + pe_emb_dim,
                                           self.temp_attn_dim)
        self.temp_attn_s_map = nn.Linear(timesteps, 1)

        # Sensor Embeddings
        self.sensor_embed_map = nn.Linear(obs_embed_dim + pe_emb_dim, out_dim)

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
        adj_l1 = self._init_adjacency_graph(batch_size)  # (B, Se, Se)

        h = self._propagate_message(h, alpha, adj_l1)

        adj_l2 = self._init_next_layer(adj_l1, alpha, mask)

        h = self._propagate_message(h, alpha, adj_l2)

        H = torch.concat([h, pe.unsqueeze(2).repeat(1, 1, self.num_sensors, 1)], dim=-1)
        H = H.permute(0, 2, 1, 3)  # (B, Se, T, ObsEmb+Pe)

        beta = self._calc_temporal_self_attention(H)

        return self._generate_sensor_embedding(H, beta)

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
                           adj: tt.BatSenSenTensor) -> tt.BatTimeSenObs_EmbTensor:
        bidir = \
            (self.inter_sensor_bidir_weights @ self.inter_sensor_bidir_weights.T)
        bidir = bidir[None, None, :, :]  # (1, 1, Se, Se)
        adj = adj[:, None, :, :]  # (B, 1, Se, Se)
        final_inter_sensor_weights = bidir * alpha * adj  # (B, T, Se, Se)
        h_prop = h.unsqueeze(2) * final_inter_sensor_weights  # (B, T, Se, Se, ObsEmd)
        h_prop = h_prop.sum(3) # (B, T, Se, ObsEmb)
        return F.relu(h_prop)

    def _init_adjacency_graph(self, batch_size: int) -> tt.BatSenSenTensor:
        return torch.ones(batch_size,
                          self.num_sensors,
                          self.num_sensors,
                          device=self.device)

    def _init_next_layer(self,
                         adj_prev: tt.BatSenSenTensor,
                         alpha: tt.BatTimeSenSenTensor,
                         mask: tt.BatTimeSenTensor) -> tt.BatSenSenTensor:
        return adj_prev * (alpha.sum(1) / mask.sum(1).unsqueeze(-1))

    def _calc_temporal_self_attention(self, H: tt.BatSenTimeObs_Pe_EmbTensor
                                      ) -> tt.BatSenTimeTensor:
        Q = self.temp_attn_query_map(H)  # (B, Se, T, dk)
        K = self.temp_attn_key_map(H)  # (B, Se, T, dk)

        beta = (Q @ K.T) / torch.sqrt(self.obs_embed_dim + self.pe_emb_dim)
        beta = self.temp_attn_s_map(beta).squeeze()  # (B, Se, T)
        return F.softmax(beta)

    def _generate_sensor_embedding(self, H, beta: tt.BatSenTimeTensor):
        return self.sensor_embed_map((beta * H).sum(2))

    # def _init_weights(self, init_range: float = 1e-10):
    #     # Observation embedding
    #     nn.init.uniform_(self.obs_emb_weights, -init_range, init_range)
