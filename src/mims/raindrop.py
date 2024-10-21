from torch import nn
from src.mims.pos_encoder import PositionalEncodingTF
import torch
from src.mims.obs_prop import ObservationProgation
from torch_geometric.nn.inits import glorot
from torch.nn import functional as F


class Raindrop(nn.Module):
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
    """

    def __init__(self,
                 num_sensors: int = 36,
                #  d_ob: int = 1,
                #  d_ob_emb: int = 4,
                #  d_pe: int = 16,
                #  timesteps: int = 60,
                 d_model=64,
                 num_heads: int = 4,
                 nhid=128, nlayers=2,
                 dropout: float = 0.3,
                 max_len: int = 215,
                 d_static: int = 9,
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2, sensor_wise_mask=False,
                 use_static: bool =True):
        super().__init__()

        # TEMPORARY FILLERS
        d_ob = 1
        d_ob_emb = 4
        d_pe = 16
        timesteps = 60
        # END

        self.num_sensors = num_sensors
        self.d_ob = d_ob
        self.d_ob_emb = d_ob_emb
        self.d_pe = d_pe
        self.timesteps = timesteps
        self.dropout = nn.Dropout(dropout)
        self.d_static = d_static
        self.use_static = use_static

        # Time
        self.pos_encoder = PositionalEncodingTF(d_pe, timesteps, MAX)

        # Static
        if use_static:
            self.static_emb_map = nn.Linear(d_static, num_sensors)

        # Adjacency matrix
        self.global_structure = torch.ones(num_sensors, num_sensors)

        # second dimension required for glorot initialization
        self.R_u = nn.Parameter(torch.Tensor(1, self.num_sensors*self.d_ob_emb)).cuda()

        ## BOUNDARY ##

        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.global_structure = torch.ones(num_sensors, num_sensors)
        self.sensor_wise_mask = sensor_wise_mask

        d_enc = num_sensors

        self.d_model = d_model
       
        # self.d_ob = int(d_model/num_sensors)

        self.encoder = nn.Linear(num_sensors*self.d_ob_emb, self.num_sensors*self.d_ob_emb)

        if self.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(self.num_sensors*(self.d_ob_emb+16), num_heads, nhid, dropout)
        else:
            encoder_layers = TransformerEncoderLayer(d_model+16, num_heads, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.adj = torch.ones([self.num_sensors, self.num_sensors]).cuda()

        self.ob_propagation =  ObservationProgation(
                                 d_feat=timesteps*self.d_ob_emb,
                                 heads=1,
                                 num_nodes=num_sensors)

        self.ob_propagation_layer2 = ObservationProgation(
                                        d_feat=timesteps*self.d_ob_emb,
                                        heads=1,
                                        num_nodes=num_sensors)

        if use_static == False:
            d_final = d_model + d_pe
        else:
            d_final = d_model + d_pe + num_sensors

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        self.aggreg = aggreg
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.use_static:
            self.static_emb_map.weight.data.uniform_(-initrange, initrange)
        glorot(self.R_u)

    def forward(self, src, static, times, lengths):
        """
        src (T, B, Se, Obs)
        """
        # """Input to the model:
        # src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        # static = Pstatic: [128, 9]: this one doesn't matter; static features
        # times = Ptime: [215, 128]: the timestamps
        # lengths = lengths: [128]: the number of nonzero recordings.
        # """
        
        timesteps, batch_size = src.shape[:2]
        
        # Time embedding
        pe = self.pos_encoder(times)

        # Static embedding
        if static is not None:
            emb = self.static_emb_map(static)

        # Split mask from masked data
        sensor_mask = src[:, :, self.num_sensors:]
        x = src[:, :, :self.num_sensors]

        x = torch.repeat_interleave(x, self.d_ob_emb, dim=-1)

        h = F.relu(x*self.R_u)
        h = self.dropout(h)

        # Mask describing time steps of samples with no sensor readings
        time_mask = (torch.arange(timesteps)[None, :] >= (lengths.cpu()[:, None])).cuda()

        # Initalize edges
        adj = self.global_structure.cuda()
        adj[torch.eye(self.num_sensors).byte()] = 1
        edge_index = torch.nonzero(adj).T
        edge_weights = adj[edge_index[0], edge_index[1]]

        # Initalize output (T, B, Se * D_ob_emb)
        output = torch.zeros([timesteps, batch_size, self.num_sensors*self.d_ob_emb]).cuda()

        # Initialize attention weights (E, B)
        alpha_all = torch.zeros([edge_index.shape[1],  batch_size]).cuda()

        for smp_idx in range(batch_size):
            smp_h = h[:, smp_idx, :]

            smp_h = smp_h.reshape([timesteps,
                                   self.num_sensors,
                                   self.d_ob_emb]).permute(1, 0, 2)
            smp_h = smp_h.reshape(self.num_sensors, timesteps*self.d_ob_emb)

            smp_h, (edge_index_layer2, alpha_layer2) = self.ob_propagation(smp_h,
                                                      edge_index=edge_index,
                                                      edge_weights=edge_weights,
                                                      edge_attr=None,
                                                      ret_attn_weights=True)
        
            alpha_layer2 = alpha_layer2.squeeze(-1)

            smp_h, (_, alpha_final) = self.ob_propagation_layer2(smp_h,
                                                             edge_index=edge_index_layer2,
                                                             edge_weights=alpha_layer2,
                                                             edge_attr=None,
                                                             ret_attn_weights=True)

            # Save propagated nodes and alphas
            smp_h = smp_h.view([self.num_sensors, timesteps, self.d_ob_emb])
            smp_h = smp_h.permute([1, 0, 2])
            smp_h = smp_h.reshape([-1, self.num_sensors*self.d_ob_emb])
            output[:, smp_idx, :] = smp_h
            alpha_all[:, smp_idx] = alpha_final.squeeze(-1)

        distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
        distance = torch.mean(distance)

        # Give each sensor its own positional encoding
        if self.sensor_wise_mask:
            extend_output = output.view(-1, batch_size, self.num_sensors, self.d_ob_emb)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.num_sensors, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size, self.num_sensors*(self.d_ob_emb+self.d_pe))
        else:
            output = torch.cat([output, pe], axis=2)

        step2 = True
        if step2 == True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=time_mask)
        elif step2 == False:
            r_out = output

        sensor_wise_mask = self.sensor_wise_mask

        masked_agg = True
        if masked_agg == True:
            lengths2 = lengths.unsqueeze(1)
            mask2 = time_mask.permute(1, 0).unsqueeze(2).long()
            if sensor_wise_mask:
                output = torch.zeros([batch_size,self.num_sensors, self.d_ob_emb+16]).cuda()
                extended_missing_mask = sensor_mask.view(-1, batch_size, self.num_sensors)
                for se in range(self.num_sensors):
                    r_out = r_out.view(-1, batch_size, self.num_sensors, (self.d_ob_emb+16))
                    out = r_out[:, :, se, :]
                    len = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                    out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (len + 1)
                    output[:, se, :] = out_sensor
                output = output.view([-1, self.num_sensors*(self.d_ob_emb+16)])
            elif self.aggreg == 'mean':
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg == False:
            output = r_out[-1, :, :].squeeze(0)

        if static is not None:
            output = torch.cat([output, emb], dim=1)
        output = self.mlp_static(output)

        return output, distance, None