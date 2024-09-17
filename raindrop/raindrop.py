import math
from typing import Optional
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
                 num_heads: int = 4,
                 inter_sensor_attn_dim: int = 12,
                 temporal_attn_dim: int = 12,
                 dropout: float = 0.3,
                 init_range: float = 1e-10,
                 prune_rate: float = 0.5,
                 device: str = 'cpu',
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
        self.num_heads = num_heads
        self.inter_sensor_attn_dim = inter_sensor_attn_dim
        self.temp_attn_dim = temporal_attn_dim
        self.dropout = nn.Dropout(dropout)
        self.init_range = init_range
        self.prune_rate = prune_rate
        self.device = device
        self.temp_attn_dim = temp_attn_dim

        # Observation Embedding
        assert obs_embed_dim % num_heads == 0

        self.obs_emb_weights = nn.Parameter(
            torch.zeros(num_sensors, obs_dim, obs_embed_dim * num_heads)
            )
        self.pos_encoder = RaindropPositionalEncoder(pe_emb_dim, device=device)

        # Inter-Sensor Attention
        self.inter_sensor_attn_weights = nn.Parameter(
            torch.zeros(num_layers, num_heads, num_sensors, self.inter_sensor_attn_dim)
        )
        self.inter_sensor_proj_map = nn.Linear(self.obs_embed_dim,
                                               self.inter_sensor_attn_dim + pe_emb_dim)
        self.inter_sensor_bidir_weights = nn.Parameter(
            torch.zeros(num_layers, num_heads, num_sensors, self.inter_sensor_attn_dim))

        # Temporal Self-Attention
        self.temp_attn_query_map = nn.Linear(obs_embed_dim + pe_emb_dim,
                                             self.temp_attn_dim)
        self.temp_attn_key_map = nn.Linear(obs_embed_dim + pe_emb_dim,
                                           self.temp_attn_dim)
        self.temp_attn_s_map = nn.Linear(timesteps, 1)

        # Sensor Embeddings
        assert out_dim // num_heads == 0
        self.sensor_embed_map = nn.Linear(obs_embed_dim + pe_emb_dim,
                                          out_dim // num_heads)

        self._init_weights(init_range)

    def forward(self,
                x: tt.BatTimeSenObsTensor,
                times: tt.BatTimeTensor,
                mask: tt.BatTimeSenTensor):
        batch_size = x.shape[0]
        h = self._embed_observation(x) * mask
        pe = self.pos_encoder(times)
        adj = self._init_adj_graph(batch_size)
        prune_msk: Optional[tt.BatHeadSenSenTensor] = None

        for lay_idx in range(self.num_layers):
            alpha = self._calc_inter_sensor_attention(h, pe, lay_idx)

            if lay_idx:
                adj = self._calc_next_adj_graph_layer(adj, alpha, mask)
                if prune_msk is None:
                    prune_msk = self._generate_prune_mask(adj)
                adj *= prune_msk

            h = self._propagate_message(h, alpha, adj)

        pe = pe[:, None, :, None, :].repeat(1, self.num_heads, 1, self.num_sensors, 1)
        H = torch.concat([h, pe], dim=-1).permute(0, 1, 3, 2, 4)

        beta = self._calc_temporal_self_attention(H)

        return self._generate_sensor_embedding(H, beta)

    def _embed_observation(self,
                           x: tt.BatTimeSenObsTensor
                           ) -> tt.BatHeadTimeSenObs_EmbTensor:
        h = F.relu(torch.einsum("ijkl,klm->ijkm", x, self.obs_emb_weights))
        h = h.view(h.shape[:3], self.num_heads, self.obs_embed_dim)
        return h.permute(0, 3, 1, 2, 4)

    def _calc_inter_sensor_attention(self,
                                     h: tt.BatHeadTimeSenObs_EmbTensor,
                                     pe: tt.BatTimePeTensor,
                                     lay_idx: int,
                                     ) -> tt.BatHeadTimeSenSenTensor:
        batch_size = h.shape[0]
        pe = pe[:, None, :, None, :].repeats(1, self.num_heads, 1, self.num_sensors, 1)
        attn_weights = self.inter_sensor_attn_weights[lay_idx]
        attn_weights = attn_weights[None, :, None, :, :].repeat(batch_size,
                                                                1,
                                                                self.timesteps,
                                                                1,
                                                                1)
        alpha = torch.concat([attn_weights, pe], dim=-1)
        alpha = alpha.view(alpha.shape[:2], self.num_heads, self.head_dim)
        alpha = alpha.permute(0, 3, 1, 2, 4)
        h = self.inter_sensor_proj_map(h)

        alpha = h @ alpha.T
        return F.relu(alpha)

    def _propagate_message(self,
                           h: tt.BatHeadTimeSenObs_EmbTensor,
                           alpha: tt.BatHeadTimeSenSenTensor,
                           adj: tt.BatHeadSenSenTensor,
                           lay_idx: int) -> tt.BatHeadTimeSenObs_EmbTensor:
        bidir = \
            (self.inter_sensor_bidir_weights[lay_idx] @
             self.inter_sensor_bidir_weights[lay_idx].T)
        bidir = bidir[None, :, None, :, :]
        adj = adj[:, :, None, :, :]
        final_inter_sensor_weights = bidir * alpha * adj
        h_prop = h.unsqueeze(3) * final_inter_sensor_weights.unsqueeze(-1)
        h_prop = h_prop.sum(4)
        return F.relu(h_prop)

    def _init_adj_graph(self, batch_size: int) -> tt.BatHeadSenSenTensor:
        return torch.ones(batch_size,
                          self.num_heads,
                          self.num_sensors,
                          self.num_sensors,
                          device=self.device)

    def _calc_next_adj_graph_layer(self,
                                   adj_prev: tt.BatHeadSenSenTensor,
                                   alpha: tt.BatHeadTimeSenSenTensor,
                                   mask: tt.BatTimeSenTensor
                                   ) -> tt.BatHeadSenSenTensor:
        mask = mask.sum(1)[:, None, :, None]
        return adj_prev * (alpha.sum(2) / mask)

    def _generate_prune_mask(self, adj: tt.BatHeadSenSenTensor
                             ) -> tt.BatHeadSenSenTensor:
        adj = adj.view(adj.shape[:2], self.num_sensors**2)
        pruned_edges = math.floor(self.num_sensors**2 * (1-self.prune_rate))
        retain_idxs = torch.argsort(adj)[:, :, pruned_edges:]
        prune_msk = torch.zeros_like(adj).scatter(dim=2, index=retain_idxs, value=1)
        return prune_msk.view(adj.shape[:2], self.num_sensors, self.num_sensors)

    def _calc_temporal_self_attention(self, H: tt.BatHeadSenTimeObs_Pe_EmbTensor
                                      ) -> tt.BatHeadSenTimeTensor:
        Q = self.temp_attn_query_map(H)
        K = self.temp_attn_key_map(H)

        beta = (Q @ K.T) / torch.sqrt(self.obs_embed_dim + self.pe_emb_dim)
        beta = self.temp_attn_s_map(beta).squeeze()
        return F.softmax(beta)

    def _generate_sensor_embedding(self,
                                   H: tt.BatHeadSenTimeObs_Pe_EmbTensor,
                                   beta: tt.BatHeadSenTimeTensor
                                   ) -> tt.BatSenOutTensor:
        batch_size = H.shape[0]
        out = self.sensor_embed_map((beta.unsqueeze(-1) * H).sum(3))
        return out.permute(0, 2, 1, 3).view(batch_size, self.num_sensors, -1)

    def _init_weights(self, init_range: float = 1e-10):
        nn.init.uniform_(self.obs_emb_weights, -init_range, init_range)
        nn.init.uniform_(self.inter_sensor_attn_weights, -init_range, init_range)
        nn.init.uniform_(self.inter_sensor_bidir_weights, -init_range, init_range)
