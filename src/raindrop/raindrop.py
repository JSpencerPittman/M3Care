import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.raindrop import tensortypes as tt  # type: ignore
from src.raindrop.models import RaindropPositionalEncoder

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
        self.prune_rate = prune_rate
        self.device = device
        self.temp_attn_dim = temp_attn_dim

        # Observation Embedding
        assert obs_embed_dim % num_heads == 0

        self.scale_inp_bn = nn.BatchNorm2d(timesteps)
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
        self.temp_attn_softmax = nn.Softmax(dim=-1)
        self.temp_attn_ln = nn.LayerNorm((num_sensors, timesteps))

        # Sensor Embeddings
        assert out_dim % num_heads == 0
        self.sensor_embed_ln = nn.LayerNorm((num_sensors, out_dim // num_heads))
        self.sensor_embed_map = nn.Linear(obs_embed_dim + pe_emb_dim,
                                          out_dim // num_heads)

        self._init_weights()

    def forward(self,
                x: tt.BatTimeSenObsTensor,
                times: tt.BatTimeTensor,
                mask: tt.BatTimeSenTensor) -> tuple[tt.BatSenOutTensor, float]:
        batch_size = x.shape[0]
        x = self.scale_inp_bn(x)
        h = self._embed_observation(x) * mask[:, None, :, :, None]
        pe = self.pos_encoder(times)
        adj = self._init_adj_graph(batch_size)
        prune_msk: Optional[tt.BatHeadSenSenTensor] = None

        h = self.dropout(h)

        adj_graph_sim = self._calc_adj_graph_sim(adj)

        for lay_idx in range(self.num_layers):
            alpha = self._calc_inter_sensor_attention(h, pe, lay_idx)

            if lay_idx:
                adj = self._calc_next_adj_graph_layer(adj, alpha, mask)
                if prune_msk is None:
                    prune_msk = self._generate_prune_mask(adj)
                adj *= prune_msk
                adj_graph_sim += self._calc_adj_graph_sim(adj)

            h = self._propagate_message(h, alpha, adj, lay_idx)

        pe = pe[:, None, :, None, :].repeat(1, self.num_heads, 1, self.num_sensors, 1)
        H = torch.concat([h, pe], dim=-1).permute(0, 1, 3, 2, 4)

        beta = self._calc_temporal_self_attention(H)

        sens_emb = self._generate_sensor_embedding(H, beta)

        return sens_emb, adj_graph_sim

    def _embed_observation(self,
                           x: tt.BatTimeSenObsTensor
                           ) -> tt.BatHeadTimeSenObs_EmbTensor:
        h = F.leaky_relu(torch.einsum("ijkl,klm->ijkm", x, self.obs_emb_weights))
        h = h.view(*h.shape[:3],
                   self.num_heads,
                   self.obs_embed_dim
                   ).permute(0, 3, 1, 2, 4)
        return h

    def _calc_inter_sensor_attention(self,
                                     h: tt.BatHeadTimeSenObs_EmbTensor,
                                     pe: tt.BatTimePeTensor,
                                     lay_idx: int,
                                     ) -> tt.BatHeadTimeSenSenTensor:
        batch_size = h.shape[0]

        pe = pe[:, None, :, None, :].repeat(1, self.num_heads, 1, self.num_sensors, 1)
        attn_weights = self.inter_sensor_attn_weights[lay_idx]
        attn_weights = attn_weights[None, :, None, :, :].repeat(batch_size,
                                                                1,
                                                                self.timesteps,
                                                                1,
                                                                1)

        alpha = torch.concat([attn_weights, pe], dim=-1).swapaxes(-1, -2)
        h = self.inter_sensor_proj_map(h)

        return F.leaky_relu(h @ alpha)

    def _propagate_message(self,
                           h: tt.BatHeadTimeSenObs_EmbTensor,
                           alpha: tt.BatHeadTimeSenSenTensor,
                           adj: tt.BatHeadSenSenTensor,
                           lay_idx: int) -> tt.BatHeadTimeSenObs_EmbTensor:
        bidir = \
            F.leaky_relu(self.inter_sensor_bidir_weights[lay_idx] @
                         self.inter_sensor_bidir_weights[lay_idx].swapaxes(-1, -2))
        bidir = bidir[None, :, None, :, :]
        adj = adj[:, :, None, :, :]
        final_inter_sensor_weights = F.leaky_relu(bidir * alpha * adj)
        h_prop = F.leaky_relu(h.unsqueeze(3) * final_inter_sensor_weights.unsqueeze(-1))
        return F.leaky_relu(h_prop.sum(4))

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
        return F.leaky_relu(adj_prev * (alpha.sum(2) / mask))

    def _generate_prune_mask(self, adj: tt.BatHeadSenSenTensor
                             ) -> tt.BatHeadSenSenTensor:
        adj = adj.view(*adj.shape[:2], self.num_sensors**2)
        pruned_edges = math.floor(self.num_sensors**2 * (1-self.prune_rate))
        retain_idxs = torch.argsort(adj)[:, :, pruned_edges:]
        prune_msk = torch.zeros_like(adj).scatter(dim=2, index=retain_idxs, value=1)
        return prune_msk.view(*adj.shape[:2], self.num_sensors, self.num_sensors)

    def _calc_adj_graph_sim(self, adj: tt.BatHeadSenSenTensor) -> float:
        batch_size = adj.shape[0]
        sim = 0.0
        for sample_idx in range(batch_size-1):
            sim += ((adj[sample_idx] - adj[sample_idx+1:])**2).sum() / (batch_size-1)**2
        return sim / self.num_sensors**2

    def _calc_temporal_self_attention(self, H: tt.BatHeadSenTimeObs_Pe_EmbTensor
                                      ) -> tt.BatHeadSenTimeTensor:
        Q = self.temp_attn_query_map(H)
        K = self.temp_attn_key_map(H)
        # norm = math.sqrt(self.obs_embed_dim + self.pe_emb_dim)
        beta = (Q @ K.swapaxes(-1, -2))  # / norm
        beta = self.temp_attn_s_map(beta).squeeze(dim=-1)
        beta = self.temp_attn_ln(beta)
        # return self.temp_attn_softmax(beta)
        return beta

    def _generate_sensor_embedding(self,
                                   H: tt.BatHeadSenTimeObs_Pe_EmbTensor,
                                   beta: tt.BatHeadSenTimeTensor
                                   ) -> tt.BatSenOutTensor:
        batch_size = H.shape[0]

        out = F.leaky_relu((beta.unsqueeze(-1) * H).sum(3))
        out = F.leaky_relu(self.sensor_embed_map(out))
        out = self.sensor_embed_ln(out)
        out = out.permute(0, 2, 1, 3)

        return torch.reshape(out, (batch_size, self.num_sensors, self.out_dim))

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.obs_emb_weights)
        nn.init.kaiming_uniform_(self.inter_sensor_attn_weights)
        nn.init.kaiming_uniform_(self.inter_sensor_bidir_weights)
