from mimic.utils import (clones, init_weights, guassian_kernel)
from torch import nn
from comp import GraphConvolution
from synth import MultiModalTransformer
import torch.nn.functional as F
import torch

MODAL_NUM = 5
MODAL_DEM = 0
MODAL_VIT = 1
MODAL_ITV = 2
MODAL_NST = 3
MODAL_NTS = 4

MISSING_MODALS = [MODAL_NST, MODAL_NTS]
NUM_MISS_MODALS = len(MISSING_MODALS)
MODAL_MISS_NST = 0
MODAL_MISS_NTS = 1

GK_MUL = 2.0
GK_NUM = 3


class M3Care(nn.Module):
    def __init__(self,
                 dem_mdl: nn.Module,
                 vit_mdl: nn.Module,
                 itv_mdl: nn.Module,
                 nst_mdl: nn.Module,
                 nts_mdl: nn.Module,
                 hidden_dim: int,
                 output_dim: int,
                 device: str,
                 keep_prob=1):
        super(M3Care, self).__init__()

        # General utilities
        self.relu = nn.ReLU()

        # Store models for each modality
        self.dem_mdl = dem_mdl
        self.vit_mdl = vit_mdl
        self.itv_mdl = itv_mdl
        self.nst_mdl = nst_mdl
        self.nts_mdl = nts_mdl

        # Modality similarity calculation
        self.simi_proj = clones(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            self.relu,
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            self.relu,
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        ), MODAL_NUM)
        self.simi_bn = nn.BatchNorm1d(hidden_dim)
        self.simi_eps = nn.Parameter(torch.ones((MODAL_NUM)+1))

        # Aggregated Auxillary Information
        self.dissimilar_thresh = nn.Parameter(torch.ones((1))+1)
        self.graph_net = clones(nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim, bias=True),
            GraphConvolution(hidden_dim, hidden_dim, bias=True)
        ]), NUM_MISS_MODALS)

        # Adaptive Modality Imputation
        self.adapt_self = clones(
            nn.Linear(hidden_dim, 1), NUM_MISS_MODALS)
        self.adapt_other = clones(
            nn.Linear(hidden_dim, 1), NUM_MISS_MODALS)

        # Multimodal Interaction Capture
        self.modal_type_embeddings = nn.Embedding(MODAL_NUM, hidden_dim)
        self.modal_type_embeddings.apply(init_weights)

        self.mm_tran = nn.ModuleList([
            MultiModalTransformer(input_dim=hidden_dim, d_model=hidden_dim,
                                  MHD_num_head=4, d_ff=hidden_dim*4, output_dim=1),
            MultiModalTransformer(input_dim=hidden_dim, d_model=hidden_dim,
                                  MHD_num_head=1, d_ff=hidden_dim*4, output_dim=1)
        ])

        # Final Prediction
        self.output_fcs = nn.ModuleList([
            nn.Linear(hidden_dim*MODAL_NUM, hidden_dim*2),
            nn.Linear(hidden_dim*2, output_dim)
        ])
        self.dropout = nn.Dropout(1-keep_prob)

        # General parameters
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, dem, vit, itv, nst, nts, missing):
        batch_size = dem.size(0)

        ### ----- Unimodal Feature Extraction ----- ###

        # Demographic Modality
        dem_emb = self.dem_mdl(dem)
        dem_msk = (torch.ones(batch_size, 1) == 1).to(self.device)

        # Vitals Modality
        vit_emb = self.vit_mdl(vit)
        vit_msk = (torch.ones(batch_size, 1) == 1).to(self.device)

        # Interventions Modality
        itv_emb = self.itv_mdl(itv)
        itv_msk = (torch.ones(batch_size, 1) == 1).to(self.device)

        # Static Notes Modality
        nst_emb = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        nst_msk = ~torch.tensor(missing[:, 0]).unsqueeze(1).to(self.device)

        nst_emb_ptl = self.nst_mdl(nst)
        nst_real_cnt = 0
        for idx, real in enumerate(nst_msk):
            if real:
                nst_emb[idx] = nst_emb_ptl[nst_real_cnt]
                nst_real_cnt += 1

        # Time-Series Notes Modality
        nts_emb = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        nts_msk = ~torch.tensor(missing[:, 1]).unsqueeze(1).to(self.device)

        nts_emb_ptls = list()
        for pat_nts in nts:
            times, cats, notes = [torch.tensor(
                v).to(self.device) for v in pat_nts]
            nts_emb_ptls.append(self.nts_mdl(times, cats, notes))

        nts_real_cnt = 0
        for idx, real in enumerate(nts_msk):
            if real:
                nts_emb[idx] = nts_emb_ptls[nts_real_cnt]
                nts_real_cnt += 1

        ### ----- Missing Modality Matrices ----- ###

        # Demographic Notes Modality
        dem_miss_mat = dem_msk * dem_msk.permute(1, 0)

        # Vitals Modality
        vit_miss_mat = vit_msk * vit_msk.permute(1, 0)

        # Interventions Modality
        itv_miss_mat = itv_msk * itv_msk.permute(1, 0)

        # Static Notes Modality
        nst_miss_mat = nst_msk * nst_msk.permute(1, 0)

        # Time-Series Notes Modality
        nts_miss_mat = nts_msk * nts_msk.permute(1, 0)

        ### ----- Calculate Similarity between Modalities ----- ###

        # Demographic Modality
        dem_sim_wgk = gaussian_kernel(self.simi_bn(F.relu(self.simi_proj[MODAL_DEM](dem_emb))),
                                      kernel_mul=GK_MUL, kernel_num=GK_NUM)
        dem_sim_gk = gaussian_kernel(self.simi_bn(dem_emb),
                                     kernel_mul=GK_MUL, kernel_num=GK_NUM)

        dem_sim_mat = ((1-F.sigmoid(self.simi_eps[MODAL_DEM])) * dem_sim_wgk +
                       F.sigmoid(self.simi_eps[MODAL_DEM])) * dem_sim_gk

        # Vitals Modality
        vit_sim_wgk = gaussian_kernel(self.simi_bn(F.relu(self.simi_proj[MODAL_VIT](vit_emb))),
                                      kernel_mul=GK_MUL, kernel_num=GK_NUM)
        vit_sim_gk = gaussian_kernel(self.simi_bn(vit_emb),
                                     kernel_mul=GK_MUL, kernel_num=GK_NUM)

        vit_sim_mat = ((1-F.sigmoid(self.simi_eps[MODAL_VIT])) * vit_sim_wgk +
                       F.sigmoid(self.simi_eps[MODAL_VIT])) * vit_sim_gk

        # Interventions Modality
        itv_sim_wgk = gaussian_kernel(self.simi_bn(F.relu(self.simi_proj[MODAL_VIT](itv_emb))),
                                      kernel_mul=GK_MUL, kernel_num=GK_NUM)
        itv_sim_gk = gaussian_kernel(self.simi_bn(itv_emb),
                                     kernel_mul=GK_MUL, kernel_num=GK_NUM)

        itv_sim_mat = ((1-F.sigmoid(self.simi_eps[MODAL_ITV])) * itv_sim_wgk +
                       F.sigmoid(self.simi_eps[MODAL_ITV])) * itv_sim_gk

        # Static Notes Modality
        nst_sim_wgk = gaussian_kernel(self.simi_bn(F.relu(self.simi_proj[MODAL_NST](nst_emb))),
                                      kernel_mul=GK_MUL, kernel_num=GK_NUM)
        nst_sim_gk = gaussian_kernel(self.simi_bn(nst_emb),
                                     kernel_mul=GK_MUL, kernel_num=GK_NUM)

        nst_sim_mat = ((1-F.sigmoid(self.simi_eps[MODAL_NST])) * nst_sim_wgk +
                       F.sigmoid(self.simi_eps[MODAL_NST])) * nst_sim_gk
        nst_sim_mat *= nst_miss_mat

        # Time-series Notes Modality
        nts_sim_wgk = gaussian_kernel(self.simi_bn(F.relu(self.simi_proj[MODAL_NTS](nst_emb))),
                                      kernel_mul=GK_MUL, kernel_num=GK_NUM)
        nts_sim_gk = gaussian_kernel(self.simi_bn(nst_emb),
                                     kernel_mul=GK_MUL, kernel_num=GK_NUM)

        nts_sim_mat = ((1-F.sigmoid(self.simi_eps[MODAL_NTS])) * nts_sim_wgk +
                       F.sigmoid(self.simi_eps[MODAL_NTS])) * nts_sim_gk
        nts_sim_mat *= nts_miss_mat

        ### ----- Stabilize learned representations --- ###

        stab_diff_dem = torch.abs(torch.norm(
            self.simi_proj[MODAL_DEM](dem_emb)) - torch.norm(dem_emb))
        stab_diff_vit = torch.abs(torch.norm(
            self.simi_proj[MODAL_VIT](vit_emb)) - torch.norm(vit_emb))
        stab_diff_itv = torch.abs(torch.norm(
            self.simi_proj[MODAL_ITV](itv_emb)) - torch.norm(itv_emb))
        stab_diff_nst = torch.abs(torch.norm(
            self.simi_proj[MODAL_NST](nst_emb)) - torch.norm(nst_emb))
        stab_diff_nts = torch.abs(torch.norm(
            self.simi_proj[MODAL_NTS](nts_emb)) - torch.norm(nts_emb))

        lstab = stab_diff_dem + stab_diff_vit + \
            stab_diff_itv + stab_diff_nst + stab_diff_nts

        ### ----- Filtered Similarity Matrix ----- ###

        agg_sim_mat = dem_sim_mat + vit_sim_mat + \
            itv_sim_mat + nst_sim_mat + nts_sim_mat
        agg_sim_mat /= dem_miss_mat + vit_miss_mat + \
            itv_miss_mat + nst_miss_mat + nts_miss_mat

        agg_sim_mat *= agg_sim_mat > F.sigmoid(self.dissimilar_thresh)

        ### --- Calculate Aggregated Auxillary Information --- ###

        # Static Notes Modality
        nst_aux = F.relu(
            self.graph_net[MODAL_MISS_NST][0](agg_sim_mat, nst_emb))
        nst_aux = F.relu(
            self.graph_net[MODAL_MISS_NST][1](agg_sim_mat, nst_aux))

        # Time-series Notes Modality
        nts_aux = F.relu(
            self.graph_net[MODAL_MISS_NTS][0](agg_sim_mat, nts_emb))
        nts_aux = F.relu(
            self.graph_net[MODAL_MISS_NTS][1](agg_sim_mat, nts_aux))

        ### --- Adaptive Modality Imputation --- ###

        # Static Notes Modality
        nst_self_info = F.sigmoid(self.adapt_self[MODAL_MISS_NST](nst_emb))
        nst_other_info = F.sigmoid(self.adapt_self[MODAL_MISS_NST](nst_aux))

        nst_self_info = nst_self_info / (nst_self_info + nst_other_info)
        nst_other_info = 1 - nst_self_info

        nst_impute = nst_miss_mat * \
            (nst_self_info * nst_emb + nst_other_info * nst_aux)
        nst_impute += (~nst_miss_mat) * nst_aux

        # Time-series Notes Modality
        nts_self_info = F.sigmoid(self.adapt_self[MODAL_MISS_NTS](nts_emb))
        nts_other_info = F.sigmoid(self.adapt_self[MODAL_MISS_NTS](nts_aux))

        nts_self_info = nts_self_info / (nts_self_info + nts_other_info)
        nts_other_info = 1 - nts_self_info

        nts_impute = nts_miss_mat * \
            (nts_self_info * nts_emb + nts_other_info * nts_aux)
        nts_impute += (~nts_miss_mat) * nts_aux

        ### ----- Multimodal Interaction Capture ----- ###

        # Demographic Modality
        dem_emb = dem_emb + \
            self.token_type_embeddings(
                MODAL_DEM * torch.ones((1, self.hidden_dim))).to(self.device)

        # Vitals Modality
        vit_emb = vit_emb + \
            self.token_type_embeddings(
                MODAL_VIT * torch.ones((1, self.hidden_dim))).to(self.device)

        # Interventions Modality
        itv_emb = itv_emb + \
            self.token_type_embeddings(
                MODAL_ITV * torch.ones((1, self.hidden_dim))).to(self.device)

        # Static Notes Modality
        nst_impute = nst_impute + \
            self.token_type_embeddings(
                MODAL_NST * torch.ones((1, self.hidden_dim))).to(self.device)

        # Time-series Notes Modality
        nts_impute = nts_impute + \
            self.token_type_embeddings(
                MODAL_NTS * torch.ones((1, self.hidden_dim))).to(self.device)

        # Reshape into B X T X D_Model
        dem_emb = dem_emb.unsqueeze(1)
        vit_emb = vit_emb.unsqueeze(1)
        itv_emb = itv_emb.unsqueeze(1)
        nst_impute = nst_impute.unsqueeze(1)
        nts_impute = nts_impute.unsqueeze(1)

        # All together
        z_mask = torch.cat([dem_msk.int(), vit_msk.int(
        ), itv_msk.int(), nst_msk.int(), nts_msk.int()], dim=-1).int()

        z0 = torch.cat([dem_emb, vit_emb, itv_emb, nst_emb, nts_emb], dim=1)

        z1 = F.relu(self.mm_tran[0](z0, z_mask))
        z2 = F.relu(self.mm_tran[1](z1, z_mask))

        dem_fin = z2[:, MODAL_DEM, :]
        vit_fin = z2[:, MODAL_VIT, :]
        itv_fin = z2[:, MODAL_ITV, :]
        nst_fin = z2[:, MODAL_NST, :]
        nts_fin = z2[:, MODAL_NTS, :]

        comb_fin = torch.concat(
            [dem_fin, vit_fin, itv_fin, nst_fin, nts_fin], dim=-1)

        y_init = self.dropout(F.relu(self.output_fcs[0](comb_fin)))
        y_res = self.output_fcs[1](y_init).squeeze(-1)

        return y_res, lstab
