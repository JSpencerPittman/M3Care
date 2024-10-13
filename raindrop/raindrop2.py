import math
from typing import Optional

import numpy as np
import torch
from models import RaindropPositionalEncoder
from obs_prop import ObservationPropagation
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn.inits import glorot


class Raindrop2(nn.Module):
    """
    Implement the raindrop stratey one by one. Transformer model with context
        embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_static = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length
        MAX  = positional encoder MAX parameter
        n_classes = number of classes
    """

    def __init__(self,
                 d_inp: int = 36,
                 num_sensors: int = 36,
                 sensor_dim: int = 1,
                 d_pe: int = 36,
                 d_static: int = 9,
                 d_model: int = 64,
                 timesteps: int = 500,
                 num_heads_obsprop: int = 1,
                 num_heads_tran_enc: int = 1,
                 d_hid_tran_enc: int = 1,
                 num_obs_prop_layers: int = 2,
                 num_tran_enc_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.3,
                 sensor_wise_mask: bool = False,
                 masked_agg: bool = True,
                 init_range: float = 1e-10,
                 device: str = 'cpu'):
        super(Raindrop, self).__init__()

        self.d_inp = d_inp
        self.num_sensors = num_sensors
        self.sensor_dim = sensor_dim
        self.d_pe = d_pe
        self.d_static = d_static
        self.d_static = d_static
        self.timesteps = timesteps
        self.num_heads_obsprop = num_heads_obsprop
        self.num_heads_tran_enc = num_heads_tran_enc
        self.d_hid_tran_enc = d_hid_tran_enc
        self.num_obs_prop_layers = num_obs_prop_layers
        self.num_tran_enc_layers = num_tran_enc_layers
        self.sensor_wise_mask = sensor_wise_mask
        self.masked_agg = masked_agg
        self.device = device

        self.R_u = nn.Parameter(torch.Tensor(1, self.num_sensors * self.sensor_dim))
        self.pos_encoder = RaindropPositionalEncoder(d_pe, timesteps, device)
        self.lin_static_embedder = nn.Linear(d_static, d_static)

        self.obs_prop = \
            nn.ModuleList([ObservationPropagation(timesteps*sensor_dim,
                                                  timesteps*sensor_dim,
                                                  num_heads_obsprop)
                           for _ in range(num_obs_prop_layers)])

        d_encoder_inp = num_sensors * sensor_dim
        d_encoder_inp += d_pe * (num_sensors if sensor_wise_mask else 0)
        tran_enc_layer = nn.TransformerEncoderLayer(d_encoder_inp,
                                                    num_heads_tran_enc,
                                                    d_hid_tran_enc,
                                                    dropout)
        self.transformer_encoder = nn.TransformerEncoder(tran_enc_layer,
                                                         num_tran_enc_layers)

        self.mlp = nn.Sequential(
            nn.Linear(d_encoder_inp+d_static, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

        self.init_weights(init_range)

    def init_weights(self, init_range: float):
        glorot(self.R_U)

    def forward(self, src, static, times, lengths, miss_mask):
        # src (T, B, Se)
        # static (B, D_static)
        # times (T, B)
        # lengths (B)
        # miss_mask (T, B, Se)

        # batch_size (1)
        batch_size = src.shape[1]

        # x,h (T, B, Se*D_ob)
        x = torch.repeat_interleave(src, self.sensor_dim, dim=-1)
        h = F.relu(x * self.R_u)
        h = self.dropout(h)

        # pe (T, B, D_pe)
        pe = self.pos_encoder(times).permute(1, 0, 2)

        # emb (B, d_static)
        static_emb = self.lin_static_embedder(static)

        # mask (B, T)
        mask = torch.arange(self.timesteps, device=self.device)[None, :]
        mask = mask >= lengths[:, None]

        # adj (Se, Se)
        adj = torch.ones((self.num_sensors, self.num_sensors), device=self.device)
        # edge_index (2, E)
        edge_index = torch.nonzero(adj).T
        # edge_weights (E)
        edge_weights = adj[edge_index[0], edge_index[1]]
        # output (T, B, Se*D_se)
        output = torch.zeros((self.timesteps,
                              batch_size,
                              self.num_sensors*self.sensor_dim),
                             device=self.device)

        num_edges = edge_index.shape[1]
        # alpha (E, B)
        alpha = torch.zeros((num_edges, batch_size), device=self.device)

        for sample_idx in range(batch_size):
            # s_x (T, Se*D_ob)
            s_x = x[:, sample_idx, :]
            # p_t (T, D_pe)
            p_t = pe[:, sample_idx, :]

            # s_x (Se, T, D_se)
            s_x = s_x.reshape((self.timesteps,
                               self.num_sensors,
                               self.sensor_dim)).permute(1, 0, 2)
            # s_x (Se, T*D_se)
            s_x = s_x.reshape((self.num_sensors, self.timesteps*self.sensor_dim))

            lay_edge_index, lay_edge_weights = edge_index, edge_weights
            for lay_idx in range(self.num_obs_prop_layers):
                # s_x (Se, T*D_se)
                # edge_index (2, E)
                # attn_weights (E, H)
                s_x, (lay_edge_index, lay_edge_weights) = \
                    self.obs_prop[lay_idx](s_x,
                                           edge_index=lay_edge_index,
                                           edge_weights=lay_edge_weights,
                                           edge_attr=None,
                                           ret_attn_weights=True)

            # s_x (Se, T, D_se)
            s_x = s_x.view((self.num_sensors, self.timesteps, self.sensor_dim))
            # s_x (T, Se, D_se)
            s_x = s_x.permute(1, 0, 2)
            # s_x (T, Se*D_se)
            s_x = s_x.reshape((-1, self.num_sensors * self.sensor_dim))

            output[:, sample_idx, :] = s_x
            alpha[:, sample_idx] = lay_edge_weights

        distance = torch.mean(torch.cdist(alpha.T, alpha.T))

        if self.sensor_wise_mask:
            # extend_output (T, B, Se, D_se)
            extend_output = output.view(-1,
                                        batch_size,
                                        self.num_sensors,
                                        self.sensor_dim)
            # extended_pe (T, B, Se, D_pe)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.num_sensors, 1])
            # output (T, B, Se, D_se+D_pe)
            output = torch.cat([extend_output, extended_pe], dim=-1)
            # output (T, B, Se * (D_se+D_pe))
            output = output.view(-1,
                                 batch_size,
                                 self.num_sensors*(self.sensor_dim+self.d_pe))
        else:
            # output (T, B, Se*D_se + D_pe)
            output = torch.cat([output, pe], dim=-1)

        # r_out (T, B, Se * (D_se+D_pe)) - if sensor-wise mask
        r_out = self.transformer_encoder(output, src_key_padding_mask=mask)

        if self.masked_agg:
            # lengths2 (B, 1)
            lengths2 = lengths.unsqueeze(-1)
            # mask2 (T, B, 1)
            mask2 = mask.permute(1, 0).unsqueeze(-1).long()

            if self.sensor_wise_mask:
                # output (B, Se, D_se+D_pe)
                output = torch.zeros([batch_size, self.num_sensors,
                                      self.sensor_dim+self.d_pe],
                                     device=self.device)
                # extended_miss_mask (T, B, Se)
                extended_miss_mask = miss_mask

                for se_idx in range(self.num_sensors):
                    # r_out  (T, B, Se, D_se+D_pe)
                    r_out = r_out.view(-1,
                                       batch_size,
                                       self.num_sensors,
                                       self.sensor_dim + self.d_pe)
                    # out (T, B, D_se+D_pe)
                    out = r_out[:, :, se_idx, :]
                    # len (B, 1)
                    len = torch.sum(extended_miss_mask[:, :, se_idx], dim=0).unsqueeze(-1)
                    # out_sensor (B, D_se+D_pe)
                    out_sensor = \
                        torch.sum(out *
                                  (1 - extended_miss_mask[:, :, se_idx].unsqueeze(-1)),
                                  dim=0) / (len + 1)
                    output[:, se_idx, :] = out_sensor
                # output (B, Se * (D_se+D_pe))
                output = output.view([-1, self.num_sensors*(self.sensor_dim+self.d_pe)])
            else:
                # output (B, Se*D_se + D_pe)
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        else:
            # output (B, Se * (D_se+D_pe))
            output = r_out[-1, :, :]

        output = torch.cat([output, static_emb], dim=-1)
        output = self.mlp(output)

        return output, distance, None
