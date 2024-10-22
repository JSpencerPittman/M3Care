import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric.nn.inits import glorot
from typing import Optional

from src.mims.obs_prop import ObservationProgation
from src.mims.pos_encoder import PositionalEncodingTF


class Raindrop(nn.Module):
    """
    A stripped version of the Raindrop model implemented by Harvard's Raindrop paper
    at "https://github.com/mims-harvard/Raindrop".
    """

    def __init__(self,
                 num_sensors: int = 36,
                 timesteps: int = 60,
                 num_classes: int = 2, 
                 d_ob: int = 1,
                 d_ob_emb: int = 4,
                 d_pe: int = 16,
                 d_static: int = 6,
                 d_static_emb: int = 128, 
                 use_static: bool = True,
                 num_tran_heads: int = 4,
                 num_tran_layers: int = 2,
                 d_trans_hid: int = 128,
                 dropout: float = 0.3,
                 init_range: float = 1e-10):
        """
        Constructor for Raindrop.

        Args:
            num_sensors (int, optional): Number of sensors (nodes) in the graph.
                Defaults to 36.
            timesteps (int, optional): Max number of timesteps. Defaults to 60.
            num_classes (int, optional): Number of classes being predicted.
                Defaults to 2.
            d_ob (int, optional): Dimensionality of the input observation. 
                Defaults to 1.
            d_ob_emb (int, optional): Dimension of an embedded observation.
                Defaults to 4.
            d_pe (int, optional): Dimension of the positional encoding. Defaults to 16.
            d_static (int, optional): Dimension of the input static tensor.
                Defaults to 6.
            d_static_emb (int, optional): Dimension of the embedded static tensor.
                Defaults to 128.
            use_static (bool, optional): Is a static tensor being used?
                Defaults to True.
            num_tran_heads (int, optional): Number of heads used in the transformer.
                Defaults to 4.
            num_tran_layers (int, optional): Number of layers in the transformer.
                Defaults to 2.
            d_trans_hid (int, optional): Dimension of the feedforward layer in the
                transformer. Defaults to 128.
            dropout (float, optional): Dropout to be used. Defaults to 0.3.
            init_range (float, optional): Range used for initializing weights. 
                Defaults to 1e-10. 
        """

        super().__init__()

        # general dimensions
        self.num_sensors = num_sensors
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.d_ob = d_ob
        self.d_ob_emb = d_ob_emb
        self.d_pe = d_pe
        self.d_static = d_static
        self.d_static_emb = d_static_emb
        self.use_static = use_static
        # transformer
        self.num_tran_heads = num_tran_heads
        self.num_tran_layers = num_tran_layers
        self.d_trans_hid = d_trans_hid
        # miscellaneous
        self.dropout = nn.Dropout(dropout)

        # time embedding
        self.pos_encoder = PositionalEncodingTF(d_pe, timesteps)

        # static embedding
        d_static_emb = num_sensors * d_ob_emb
        self.d_static_emb = d_static_emb
        if use_static:
            self.static_emb_map = nn.Linear(d_static, d_static_emb)

        # adjacency matrix
        self.adj = torch.ones(num_sensors, num_sensors).cuda()

        # second dimension required for glorot initialization
        self.R_u = nn.Parameter(torch.Tensor(1, self.num_sensors*self.d_ob_emb)).cuda()

        # propagation of observation
        self.ob_propagation =  ObservationProgation(
                                 d_feat=timesteps*self.d_ob_emb,
                                 heads=1,
                                 num_nodes=num_sensors)

        self.ob_propagation_layer2 = ObservationProgation(
                                        d_feat=timesteps*self.d_ob_emb,
                                        heads=1,
                                        num_nodes=num_sensors)

        # calculation of r_out
        encoder_layers = nn.TransformerEncoderLayer(num_sensors * d_ob_emb + d_pe,
                                                    num_tran_heads, 
                                                    d_trans_hid,
                                                    dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                         num_tran_layers)

        # logits for the 2 class predictions
        d_final = num_sensors * d_ob_emb + d_pe
        if use_static:
            d_final += d_static_emb

        self.mlp = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, num_classes),
        )

        self.init_weights(init_range)

    def init_weights(self, init_range: float):
        """
        Initialize weights.

        Args:
            init_range (float): Range used to initialize weights.
        """

        if self.use_static:
            self.static_emb_map.weight.data.uniform_(-init_range, init_range)
        glorot(self.R_u)

    def forward(self,
                src: Tensor,
                times: Tensor,
                lengths: Tensor,
                static: Optional[Tensor] = None) -> Tensor:
        """
        Classify the passed in time series.

        Args:
            src (Tensor): The time series along with the concatenated sensorwise mask
                tensor (T, B, 2*Se).
            times (Tensor): The timestamp for each sample at each time step. 0 is
                equivalent to the sensor not having a value at that timestep (T, B). 
            lengths (Tensor): The number of nonzero recordings (B). This is at the
                scope of a timestep not a sensor.
            static (Optional[Tensor], optional): The static tensor to be embedded into
                the time series (B, S).

        Returns:
            Tensor: The logits of the classification of the time series and optionally
                passed in static tensor.
        """
        
        self._verify_inputs(src, times, lengths, static)

        timesteps, batch_size = src.shape[:2]
        
        # Time embedding
        pe = self.pos_encoder(times)

        # Static embedding
        if static is not None:
            static_emb = self.static_emb_map(static)

        # Format x to the correct dimensions
        x = src[:, :, :self.num_sensors]
        x = torch.repeat_interleave(x, self.d_ob_emb, dim=-1)

        # Extract nodes
        h = F.relu(x*self.R_u)
        h = self.dropout(h)

        # Mask describing time steps of samples with no sensor readings
        time_mask = (torch.arange(timesteps)[None, :] >= (lengths.cpu()[:, None])).cuda()

        # Initalize edges
        adj = self.adj
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

        output = torch.cat([output, pe], axis=2) # (T, B, Se * D_ob + D_pe)

        r_out = self.transformer_encoder(output, src_key_padding_mask=time_mask)

        lengths2 = lengths.unsqueeze(1) # (B, 1)
        mask2 = time_mask.permute(1, 0).unsqueeze(2).long() # (T, B, 1)
        output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)

        if static is not None:
            output = torch.cat([output, static_emb], dim=1)
        output = self.mlp(output)

        return output
    
    def _verify_inputs(self,
                       src: Tensor,
                       times: Tensor,
                       lengths: Tensor,
                       static: Optional[Tensor] = None):
        """
        Verify the inputs in regards to required, type, dimensionality, and shape.
        """

        # Required
        assert not self.use_static or static is not None

        # Type
        assert isinstance(src, Tensor)
        assert isinstance(times, Tensor)
        assert isinstance(lengths, Tensor)
        if self.use_static:
            assert isinstance(static, Tensor)

        # Dimensionality
        assert src.ndim == 3
        assert times.ndim == 2
        assert lengths.ndim == 1
        if self.use_static:
            assert static.ndim == 2

        # Shape
        batch_size = src.shape[1]
        assert batch_size > 0

        assert src.shape == (self.timesteps, batch_size, self.num_sensors*2)
        assert times.shape == (self.timesteps, batch_size)
        assert lengths.shape == (batch_size,)
        if self.use_static:
            assert static.shape == (batch_size, self.d_static)
