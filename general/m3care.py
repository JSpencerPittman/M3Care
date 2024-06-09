from torch.nn import functional as F
from general.util import (clones, init_weights, guassian_kernel)
from general.model import (
    GraphConvolution, MultiModalTransformer, PositionalEncoding)
from torch import nn, Tensor
from typing import List
import torch

GK_MUL = 2.0
GK_NUM = 3
MM_TRAN_NHEADS = [4, 1]


class M3Care(nn.Module):
    def __init__(self,
                 unimodal_models: nn.ModuleList,
                 missing_modals: List[bool],
                 time_modals: List[bool],
                 timesteps_modals: List[int],
                 mask_modals: List[bool],
                 hidden_dim: int,
                 output_dim: int,
                 device: str,
                 keep_prob=1):
        super(M3Care, self).__init__()

        assert len(unimodal_models) == len(missing_modals)
        assert len(unimodal_models) == len(time_modals)

        # General parameters
        self.num_modals = len(unimodal_models)
        self.num_miss_modals = sum(missing_modals)
        self.hidden_dim = hidden_dim
        self.device = device

        # Unimodal extractions
        self.unimodal_models = unimodal_models
        self.modal_full_idxs = [
            i for i, b in enumerate(missing_modals) if not b]
        self.modal_miss_idxs = [i for i, b in enumerate(missing_modals) if b]

        self.modal_time_idxs = [i for i, b in enumerate(time_modals) if b]

        self.modal_msk_idxs = [i for i, b in enumerate(mask_modals) if b]
        self.num_msk_modals = sum(mask_modals)

        self.modal_timesteps = timesteps_modals
        self.num_embeddings = sum(self.modal_timesteps) + \
            self.num_modals - sum(time_modals)

        # Modality similarity calculation
        self.relu = nn.ReLU()
        self.simi_proj = clones(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            self.relu,
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            self.relu,
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        ), self.num_modals)
        self.simi_bn = nn.BatchNorm1d(hidden_dim)
        self.simi_eps = nn.Parameter(torch.ones((self.num_modals)+1))

        # Aggregated Auxillary Information
        self.dissimilar_thresh = nn.Parameter(torch.ones((1))+1)
        self.graph_net = clones(nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim, bias=True),
            GraphConvolution(hidden_dim, hidden_dim, bias=True)
        ]), self.num_miss_modals)

        # Adaptive Modality Imputation
        self.adapt_self = clones(
            nn.Linear(hidden_dim, 1), self.num_miss_modals)
        self.adapt_other = clones(
            nn.Linear(hidden_dim, 1), self.num_miss_modals)

        # Multimodal Interaction Capture
        self.modal_type_embeddings = nn.Embedding(self.num_modals, hidden_dim)
        self.modal_type_embeddings.apply(init_weights)

        self.mm_tran = nn.ModuleList([
            MultiModalTransformer(d_input=hidden_dim, d_model=hidden_dim,
                                  d_ff=4*hidden_dim, num_heads=MM_TRAN_NHEADS[0]),
            MultiModalTransformer(d_input=hidden_dim, d_model=hidden_dim,
                                  d_ff=4*hidden_dim, num_heads=MM_TRAN_NHEADS[1])
        ])

        self.pos_encode = PositionalEncoding(hidden_dim, (1-keep_prob))

        # Final Prediction
        self.output_fcs = nn.ModuleList([
            nn.Linear(hidden_dim*self.num_embeddings, hidden_dim*2),
            nn.Linear(hidden_dim*2, output_dim)
        ])
        self.dropout = nn.Dropout(1-keep_prob)

    def forward(self, *inputs):
        # [Modalities], [masks], batch_size
        assert len(inputs) == self.num_modals + self.num_msk_modals + 1

        modals_inp = inputs[:self.num_modals]
        modals_inp_msks = inputs[self.num_modals:-1]
        batch_size = inputs[-1]

        ### ----- Unimodal Feature Extraction ----- ###

        modal_embs_orig = []  # Each: Static -> B x Dmodel ; TS -> B x T x Dmodel
        modal_msks_orig = []  # Each: Static -> B ; TS -> B x T
        modal_embs = []  # Each: B x Dmodel
        modal_msks = []  # Each: B x 1

        for modal_idx in range(self.num_modals):
            modal_emb: None | Tensor = None
            modal_msk: None | Tensor = None

            modal_inp = modals_inp[modal_idx]

            # Determine embedded representation and its mask
            if modal_idx in self.modal_msk_idxs:
                modal_inp_msk = modals_inp_msks[self.modal_msk_idxs.index(
                    modal_idx)]
                modal_emb, modal_msk = self.unimodal_models[modal_idx](
                    modal_inp, modal_inp_msk)

            else:
                modal_emb = self.unimodal_models[modal_idx](modal_inp)
                if modal_idx in self.modal_time_idxs:
                    modal_msk = torch.ones(
                        (batch_size, modal_emb.shape[1]), dtype=torch.bool, device=self.device)
                else:
                    modal_msk = torch.ones(
                        (batch_size, 1), dtype=torch.bool, device=self.device)

            modal_embs_orig.append(modal_emb)
            modal_msks_orig.append(modal_msk)

            # For time series ensure the first timestep is selected
            if modal_idx in self.modal_time_idxs:
                modal_embs.append(modal_emb[:, 0, :])
                modal_msks.append(modal_msk[:, 0].unsqueeze(-1))
            else:
                modal_embs.append(modal_emb)
                modal_msks.append(modal_msk)

        modal_msks = torch.stack(modal_msks)  # M x B x 1

        ### ----- Missing Modality Matrices ----- ###

        modal_mask_mats = modal_msks * modal_msks.transpose(2, 1)  # M x B x B

        ### ----- Calculate Similarity between Modalities ----- ###

        sim_mats = []  # Each: B x B

        for modal_idx, modal_emb in enumerate(modal_embs):
            sim_wgk = guassian_kernel(self.simi_bn(F.relu(self.simi_proj[modal_idx](modal_emb))),
                                      kernel_mul=GK_MUL, kernel_num=GK_NUM)
            sim_gk = guassian_kernel(self.simi_bn(modal_emb),
                                     kernel_mul=GK_MUL, kernel_num=GK_NUM)

            sim_mat = ((1-F.sigmoid(self.simi_eps[modal_idx])) * sim_wgk +
                       F.sigmoid(self.simi_eps[modal_idx])) * sim_gk

            sim_mat *= modal_mask_mats[modal_idx]

            sim_mats.append(sim_mat)

        sim_mats = torch.stack(sim_mats)  # M x B x B

        ### ----- Stabilize learned representations --- ###

        lstab = 0.0

        for modal_idx, modal_emb in enumerate(modal_embs):
            lstab += torch.abs(torch.norm(
                self.simi_proj[modal_idx](modal_emb)) - torch.norm(modal_emb))

        ### ----- Filtered Similarity Matrix ----- ###

        agg_sim_mat = sim_mats.sum(dim=0) / modal_mask_mats.sum(dim=0)
        agg_sim_mat *= agg_sim_mat > F.sigmoid(self.dissimilar_thresh)  # B x B

        ### --- Calculate Aggregated Auxillary Information --- ###

        modal_auxs = []  # Each: B x Dmodel

        for miss_idx, modal_idx in enumerate(self.modal_miss_idxs):
            modal_aux = F.relu(self.graph_net[miss_idx][0](
                agg_sim_mat, modal_embs[modal_idx]))
            modal_aux = F.relu(
                self.graph_net[miss_idx][1](agg_sim_mat, modal_aux))

            modal_auxs.append(modal_aux)

        ### --- Adaptive Modality Imputation --- ###

        for miss_idx, modal_idx in enumerate(self.modal_miss_idxs):
            modal_self_info = F.sigmoid(
                self.adapt_self[miss_idx](modal_embs[modal_idx]))
            modal_other_info = F.sigmoid(
                self.adapt_self[miss_idx](modal_auxs[miss_idx]))

            modal_self_info = modal_self_info / \
                (modal_self_info + modal_other_info)
            modal_other_info = 1 - modal_self_info

            modal_impute = (
                modal_self_info * modal_embs[modal_idx]) + (modal_other_info * modal_auxs[miss_idx])
            modal_impute = (
                modal_impute * modal_msks[modal_idx]) + (~modal_msks[modal_idx] * modal_auxs[miss_idx])

            modal_embs[modal_idx] = modal_impute

        for modal_idx in range(self.num_modals):
            if modal_idx in self.modal_time_idxs:
                modal_embs_orig[modal_idx][:, 0, :] = modal_embs[modal_idx]
            else:
                modal_embs_orig[modal_idx] = modal_embs[modal_idx].unsqueeze(1)
                modal_msks_orig[modal_idx] = modal_msks_orig[modal_idx]

        ### ----- Multimodal Interaction Capture ----- ###

        for modal_idx in range(self.num_modals):
            modal_embs_orig[modal_idx] += self.modal_type_embeddings(
                torch.IntTensor([modal_idx]).to(self.device))

            if modal_idx in self.modal_time_idxs:
                modal_embs_orig[modal_idx] = self.pos_encode(
                    modal_embs_orig[modal_idx])

        # modal_embs_orig: B x T x D_model

        z0 = torch.concat(modal_embs_orig, dim=1)  # B x T(All) x D_model

        z_mask = torch.concat(
            modal_msks_orig, dim=1).unsqueeze(-1)  # B x T(All) x 1
        z_mask = z_mask * z_mask.transpose(-1, -2)  # B x T(All) x T(All)

        # B x H x T(All) x T(All)
        z_mask0 = torch.concat([z_mask]*MM_TRAN_NHEADS[0], dim=1)
        z_mask1 = torch.concat([z_mask]*MM_TRAN_NHEADS[1], dim=1)

        # (B*H) x T(All) x T(All)
        z_mask0 = z_mask0.view(-1, z_mask0.size(-1), z_mask0.size(-1))
        z_mask1 = z_mask1.view(-1, z_mask1.size(-1), z_mask1.size(-1))

        z1 = F.relu(self.mm_tran[0](z0, z_mask0))  # B x T(All) x Dmodel
        z2 = F.relu(self.mm_tran[1](z1, z_mask1))  # B x T(All) x Dmodel

        comb_fin = z2.view(batch_size, -1)  # B x (T(All) * Dmodel)

        y_init = self.dropout(F.relu(self.output_fcs[0](comb_fin)))
        y_res = self.output_fcs[1](y_init).squeeze(-1)

        return y_res, lstab
