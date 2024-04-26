from util import (clones, init_weights, guassian_kernel)
from torch import nn
from components import (GraphConvolution, MultiModalTransformer)
from typing import List
import torch.nn.functional as F
import torch

GK_MUL = 2.0
GK_NUM = 3


class M3Care(nn.Module):
    def __init__(self,
                 unimodal_models: nn.ModuleList,
                 missing_modals: List[bool],
                 hidden_dim: int,
                 output_dim: int,
                 device: str,
                 keep_prob=1):
        super(M3Care, self).__init__()

        assert len(unimodal_models) == len(missing_modals)

        # General parameters
        self.num_modals = len(unimodal_models)
        self.num_miss_modals = sum(unimodal_models)
        self.hidden_dim = hidden_dim
        self.device = device

        # Unimodal extractions
        self.unimodal_models = unimodal_models
        self.modal_full_idxs = [
            i for i, b in enumerate(missing_modals) if not b]
        self.modal_miss_idxs = [i for i, b in enumerate(missing_modals) if b]

        # Modality similarity calculation
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
            MultiModalTransformer(input_dim=hidden_dim, d_model=hidden_dim,
                                  MHD_num_head=4, d_ff=hidden_dim*4, output_dim=1),
            MultiModalTransformer(input_dim=hidden_dim, d_model=hidden_dim,
                                  MHD_num_head=1, d_ff=hidden_dim*4, output_dim=1)
        ])

        # Final Prediction
        self.output_fcs = nn.ModuleList([
            nn.Linear(hidden_dim*self.num_modals, hidden_dim*2),
            nn.Linear(hidden_dim*2, output_dim)
        ])
        self.dropout = nn.Dropout(1-keep_prob)

    def forward(self, *inputs):
        # [Modalities], missing, batch_size
        assert len(inputs) == self.num_modals + 2

        modal_inputs = inputs[:-2]
        missing = inputs[-2]
        batch_size = inputs[-1]

        ### ----- Unimodal Feature Extraction ----- ###

        # Non-missing modalities
        modal_embs = []  # Each: B x Dmodel
        modal_msks = []  # Each: B x 1

        for modal_idx in range(self.num_modals):
            modal_emb = self.unimodal_models[modal_idx](
                modal_inputs[modal_idx])
            if modal_idx in self.modal_full_idxs:
                modal_msk = (torch.ones(batch_size, 1) == 1).to(self.device)
            else:
                miss_idx = self.modal_miss_idxs.index(modal_idx)
                modal_msk = torch.zeros(
                    missing[:, miss_idx]).unsqueeze(-1).to(self.device)

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
                modal_self_info * modal_embs[modal_idx]) + (modal_other_info * modal_embs[modal_idx])
            modal_impute = (
                modal_impute * modal_mask_mats[modal_idx]) + (~modal_mask_mats[modal_idx] * modal_auxs[miss_idx])

            modal_embs[modal_idx] = modal_impute

        ### ----- Multimodal Interaction Capture ----- ###

        for modal_idx in range(self.num_modals):
            modal_embs[modal_idx] += self.modal_type_embeddings[modal_idx](
                modal_idx * torch.ones((1, self.hidden_dim))).to(self.device)

            modal_embs[modal_idx].unsqueeze(1)

        # modal_embs: B x 1 x D_model

        z_mask = modal_msks.int().squeeze(-1).transpose(0, 1)  # B x M
        z0 = torch.cat(modal_embs, dim=1)  # B x M x Dmodel

        z1 = F.relu(self.mm_tran[0](z0, z_mask))  # B x M x Dmodel
        z2 = F.relu(self.mm_tran[1](z1, z_mask))  # B x M x Dmodel

        comb_fin = z2.view(batch_size, -1)

        y_init = self.dropout(F.relu(self.output_fcs[0](comb_fin)))
        y_res = self.output_fcs[1](y_init).squeeze(-1)

        return y_res, lstab
