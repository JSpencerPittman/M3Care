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
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 init_range: float = 1e-10,
                 device: str = 'cpu',
                 inter_sensor_attn_dim: int = 10,
                 temp_attn_dim: int = 10):
        super().__init__()

        # Parameters
        self.num_sensors = num_sensors
        self.obs_dim = obs_dim
        self.obs_embed_dim = obs_embed_dim
        self.pe_emb_dim = pe_emb_dim
        self.timesteps = timesteps
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.init_range = init_range
        self.device = device
        self.inter_sensor_attn_dim = inter_sensor_attn_dim
        self.temp_attn_dim = temp_attn_dim

        # Observation Embedding
        self.obs_emb_weights = nn.Parameter(
            torch.zeros(num_sensors, obs_dim, obs_embed_dim)
            )
        self.pos_encoder = RaindropPositionalEncoder(pe_emb_dim, device=device)

        # Inter-Sensor Attention
        self.inter_sensor_attn_weights = nn.Parameter(
            torch.zeros(num_layers, num_sensors, self.inter_sensor_attn_dim)
        )
        self.inter_sensor_proj_map = nn.Linear(self.obs_embed_dim,
                                               self.inter_sensor_attn_dim + pe_emb_dim)
        self.inter_sensor_bidir_weights = nn.Parameter(
            torch.zeros(self.num_sensors, self.obs_embed_dim)
                                                      )

        # Temporal Self-Attention
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
        batch_size = x.shape[0]

        # Exclude batch_size
        h = self._embed_observation(x) * mask
        pe = self.pos_encoder(times)
        adj = self._init_adjacency_graph(batch_size)

        for lay_idx in range(self.num_layers):
            alpha = self._calc_inter_sensor_attention(h, pe, lay_idx)

            if lay_idx:
                adj = self._calc_next_adj_graph_layer(adj, alpha, mask)

            h = self._propagate_message(h, alpha, adj)

        H = torch.concat([h, pe.unsqueeze(2).repeat(1, 1, self.num_sensors, 1)], dim=-1)
        H = H.permute(0, 2, 1, 3)

        beta = self._calc_temporal_self_attention(H)

        return self._generate_sensor_embedding(H, beta)

    def _embed_observation(self, x: tt.BatTimeSenObsTensor
                           ) -> tt.BatTimeSenObs_EmbTensor:
        return F.relu(torch.einsum("ijkl,klm->ijkm", x, self.obs_emb_weights))

    def _calc_inter_sensor_attention(self,
                                     h: tt.BatTimeSenObs_EmbTensor,
                                     pe: tt.BatTimePeTensor,
                                     lay_idx: int,
                                     ) -> tt.BatTimeSenSenTensor:
        batch_size, time_steps, _, _ = h.shape
        pe = pe.unsqueeze(2).repeat(1, 1, self.num_sensors, 1)
        attn_weights = self.inter_sensor_attn_weights[lay_idx]
        attn_weights = attn_weights[None, None, :, :].repeat(batch_size,
                                                             self.timesteps,
                                                             1,
                                                             1)                                                 1, 1)
        alpha = torch.concat([attn_weights, pe], dim=-1)
        h = self.inter_sensor_proj_map(h)

        alpha = h @ alpha.T
        return F.relu(alpha)

    def _propagate_message(self,
                           h: tt.BatTimeSenObs_EmbTensor,
                           alpha: tt.BatTimeSenSenTensor,
                           adj: tt.BatSenSenTensor) -> tt.BatTimeSenObs_EmbTensor:
        bidir = \
            (self.inter_sensor_bidir_weights @ self.inter_sensor_bidir_weights.T)
        bidir = bidir[None, None, :, :]
        adj = adj[:, None, :, :]
        final_inter_sensor_weights = bidir * alpha * adj
        h_prop = h.unsqueeze(2) * final_inter_sensor_weights
        h_prop = h_prop.sum(3)
        return F.relu(h_prop)

    def _init_adjacency_graph(self, batch_size: int) -> tt.BatSenSenTensor:
        return torch.ones(batch_size,
                          self.num_sensors,
                          self.num_sensors,
                          device=self.device)

    def _calc_next_adj_graph_layer(self,
                                   adj_prev: tt.BatSenSenTensor,
                                   alpha: tt.BatTimeSenSenTensor,
                                   mask: tt.BatTimeSenTensor) -> tt.BatSenSenTensor:
        return adj_prev * (alpha.sum(1) / mask.sum(1).unsqueeze(-1))

    def _calc_temporal_self_attention(self, H: tt.BatSenTimeObs_Pe_EmbTensor
                                      ) -> tt.BatSenTimeTensor:
        Q = self.temp_attn_query_map(H)
        K = self.temp_attn_key_map(H)

        beta = (Q @ K.T) / torch.sqrt(self.obs_embed_dim + self.pe_emb_dim)
        beta = self.temp_attn_s_map(beta).squeeze()
        return F.softmax(beta)

    def _generate_sensor_embedding(self,
                                   H: tt.BatSenTimeObs_Pe_EmbTensor,
                                   beta: tt.BatSenTimeTensor
                                   ) -> tt.BatSenOutTensor:
        return self.sensor_embed_map((beta * H).sum(2))

    # def _init_weights(self, init_range: float = 1e-10):
    #     # Observation embedding
    #     nn.init.uniform_(self.obs_emb_weights, -init_range, init_range)
