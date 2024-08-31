from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from m3care import tensortypes as tt
from m3care.model import (GraphConvolution, MultiModalTransformer,
                          PositionalEncoding)
from m3care.util import guassian_kernel, init_weights


class Modal(nn.Module):
    def __init__(self,
                 name: str,
                 model: nn.Module,
                 masked: bool = False,
                 time_dim: Optional[int] = None):
        super().__init__()

        self.name = name
        self.model = model
        self.masked = masked
        self.time_dim = time_dim

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class M3Care(nn.Module):
    GK_MUL = 2.0
    GK_NUM = 3

    def __init__(self,
                 modals: list[Modal],
                 embedded_dim: int,
                 output_dim: int,
                 device: str,
                 dropout: float = 0.0,
                 num_heads: Sequence[int] = [4, 1]):
        super().__init__()

        self.modals = nn.ModuleDict({modal.name: modal for modal in modals})
        self.modal_names = [modal.name for modal in modals]
        self.num_modals = len(modals)
        self.embedded_dim = embedded_dim
        self.device = device

        # Modality similarity calculation
        self.simi_proj = nn.ModuleDict({modal.name: nn.Sequential(
                nn.Linear(embedded_dim, embedded_dim, bias=True),
                nn.ReLU(),
                nn.Linear(embedded_dim, embedded_dim, bias=True),
                nn.ReLU(),
                nn.Linear(embedded_dim, embedded_dim, bias=True))
            for modal in modals})
        self.simi_bn = nn.BatchNorm1d(embedded_dim)
        self.simi_eps = nn.ParameterDict(
            {modal.name: nn.Parameter(torch.ones(1)) for modal in modals})

        self.dissimilar_thresh = nn.Parameter(torch.ones((1))+1)

        # Modality Auxillary calculation
        self.graph_net = nn.ModuleDict(
            {modal_name: GraphConvolution(embedded_dim, bias=True)
             for modal_name in self.modal_names})

        # Adaptive modality imputation
        self.adapt_self = nn.ModuleDict({modal_name: nn.Linear(embedded_dim, 1)
                                         for modal_name in self.modal_names})
        self.adapt_other = nn.ModuleDict({modal_name: nn.Linear(embedded_dim, 1)
                                          for modal_name in self.modal_names})

        number_of_embeddings = sum([(1 if modal.time_dim is None else modal.time_dim)
                                   for modal in self.modals.values()])

        # Multimodal interaction capture
        self.modal_type_embeddings = nn.Embedding(self.num_modals, embedded_dim)
        self.modal_type_embeddings.apply(init_weights)
        self.mm_tran = MultiModalTransformer(embedded_dim=embedded_dim,
                                             num_heads=num_heads,
                                             dropout=dropout,
                                             max_len=number_of_embeddings)

        self.pos_encode = PositionalEncoding(embedded_dim, dropout)

        # Final Prediction
        self.output_fcs = nn.ModuleList([
            nn.Linear(embedded_dim*number_of_embeddings, embedded_dim*2),
            nn.Linear(embedded_dim*2, output_dim)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                inputs: dict[str, Tensor],
                masks: dict[str, Tensor],
                batch_size: int):

        # ----- Unimodal Feature Extraction ----- #

        embs_orig: dict[str, tt.BatTimeEmbTensor] = {}
        emb_masks_orig: dict[str, tt.BatTimeTensor] = {}
        embs: dict[str, tt.BatEmbTensor] = {}
        emb_masks: dict[str, tt.BatTensor] = {}
        for name, modal in self.modals.items():
            emb_orig, emb_mask_orig, emb, emb_mask = \
                self._unimodal_feat_extract(modal,
                                            inputs[name],
                                            batch_size,
                                            masks[name] if modal.masked else None)
            embs_orig[name], emb_masks_orig[name] = emb_orig, emb_mask_orig
            embs[name], emb_masks[name] = emb, emb_mask

        # ----- Unimodal patient similarity matrices ----- #
        sim_mats: dict[str, tt.BatBatTensor] = {}
        sim_mat_masks: dict[str, tt.BatBatTensor] = {}

        for name in self.modal_names:
            sim_mats[name], sim_mat_masks[name] = \
                self._calc_intramodal_similarities(name, embs[name], emb_masks[name])

        # ----- Stabilize learned representations --- #

        lstab = self._calc_stabilization(embs)

        # ----- Filtered Similarity Matrix ----- #

        filt_sim_mat = self._calc_filt_sim_mat(sim_mats, sim_mat_masks, batch_size)

        # --- Calculate Aggregated Auxillary Information --- #

        auxillaries = {name: self._calc_modal_auxillary(name,
                                                        embs[name],
                                                        filt_sim_mat)
                       for name in self.modal_names}

        # --- Adaptive Modality Imputation --- #

        imputes = {name: self._adapative_imputation(name,
                                                    embs[name],
                                                    emb_masks[name],
                                                    auxillaries[name])
                   for name in self.modal_names}
        imputes_orig = {name: torch.concatenate(
            [imputes[name].unsqueeze(1), embs_orig[name][:, 1:, :]], dim=1)
            for name in self.modal_names}

        # --- Multimodal Interaction Capture --- #

        for idx, (name, modal) in enumerate(self.modals.items()):
            imputes_orig[name] = imputes_orig[name] + self.modal_type_embeddings(
                torch.IntTensor([idx]).to(self.device)
            )

            if modal.time_dim is not None:
                imputes_orig[name] = self.pos_encode(imputes_orig[name])

        # B x T(All) x D_model
        z0 = torch.concat([imputes_orig[name] for name in self.modal_names], dim=1)

        # B x T(All) x 1
        z_mask = torch.concat(
            [emb_masks_orig[name] for name in self.modal_names], dim=1
        ).unsqueeze(-1)

        z_mask = z_mask * z_mask.transpose(-1, -2)  # B x T(All) x T(All)
        zf = self.mm_tran(z0, z_mask)
        zf_comb = zf.view(batch_size, -1)
        y_init = self.dropout(F.relu(self.output_fcs[0](zf_comb)))
        y_res = self.output_fcs[1](y_init).squeeze(-1)
        return y_res, lstab

    def _unimodal_feat_extract(self,
                               modal: Modal,
                               x: Tensor,
                               batch_size: int,
                               mask: Optional[Tensor] = None
                               ) -> tuple[tt.BatTimeEmbTensor,
                                          tt.BatTimeTensor,
                                          tt.BatEmbTensor,
                                          tt.BatTensor]:
        """
        Unimodal feature extraction.

        Args:
            modal (Modal): Modal involved.
            x (Tensor): The input tensor.
            batch_size (int): Batch size.
            mask (Optional[Tensor], optional): The mask for the input tensor. Defaults
                to None.

        Returns:
          tuple[BatTimeEmbTensor, BatTimeTensor, BatEmbTensor, BatTensor]:
          1. (BatTimeEmbTensor): The originally embedded modality.
          2. (BatTimeTensor): Mask describing the embedded modality.
          3. (BatEmbTensor): The first position of the embedded modality.
            If the modality is static then this is the same as the original embedding.
          4. (BatTensor): Mask describing the first position of the embedded
            modality.
        """

        if modal.masked:
            emb_orig, emb_mask_orig = modal.model(x, mask)
            if modal.time_dim is None:
                emb_orig = emb_orig.unsqueeze(1)
                emb_mask_orig = emb_mask_orig.unsqueeze(1)
        else:
            emb_orig = modal.model(x).unsqueeze(1)
            emb_mask_shape = (batch_size,
                              1 if modal.time_dim is None else modal.time_dim)
            emb_mask_orig = torch.ones(*emb_mask_shape,
                                       dtype=torch.bool,
                                       device=self.device)
        emb = emb_orig[:, 0, :]
        emb_mask = emb_mask_orig[:, 0]

        return emb_orig, emb_mask_orig, emb, emb_mask

    def _calc_intramodal_similarities(self,
                                      modal_name: str,
                                      emb: tt.BatEmbTensor,
                                      emb_masks: tt.BatTensor
                                      ) -> tuple[tt.BatBatTensor,
                                                 tt.BatBatTensor]:
        """
        Calculates the similarities between each sample of the batch for a given
        modality.

        Args:
            modal_name (str): Name of the modality.
            emb (BatEmbTensor): The embedded representations of each sample.
            emb_masks (BatTensor): The masks for the embedded samples.

        Returns:
            BatBatTensor: A matrix describing the similarity between samples
                for the modality. Similarity is 0 if one of the two involved samples
                is missing.
        """

        emb_masks = emb_masks.unsqueeze(-1)
        missing_mat = emb_masks * emb_masks.T

        # Weighted gaussian similarity
        sim_wgk = guassian_kernel(
            self.simi_bn(
                F.relu(
                    self.simi_proj[modal_name](emb)
                    )
                ),
            kernel_mul=M3Care.GK_MUL,
            kernel_num=M3Care.GK_NUM)
        # Regular gaussian similarity
        sim_gk = guassian_kernel(
            self.simi_bn(
                F.relu(emb)
                ),
            kernel_mul=M3Care.GK_MUL,
            kernel_num=M3Care.GK_NUM)

        # Combine both gaussian similarities.
        sim_mat = ((1 - F.sigmoid(self.simi_eps[modal_name])) * sim_wgk +
                   F.sigmoid(self.simi_eps[modal_name])) * sim_gk

        # Zero out missing samples.
        sim_mat = sim_mat * missing_mat

        return sim_mat, missing_mat

    def _calc_stabilization(self,
                            embs: dict[str, tt.BatEmbTensor]) -> tt.Scalar:
        lstab = 0.0

        for modal_name, emb in embs.items():
            lstab = lstab + torch.abs(
                torch.norm(
                    self.simi_proj[modal_name](emb) - torch.norm(emb)
                )
            )

        return lstab

    def _calc_filt_sim_mat(self,
                           sim_mats: dict[str, tt.BatBatTensor],
                           sim_mat_masks: dict[str, tt.BatBatTensor],
                           batch_size: int) -> tt.BatBatTensor:
        """
        Calculate the filtered similarity matrix. This similarity matrix is the
        aggregation of each modal's individual similarity matrix into a single matrix
        describing each sample's similarity with other samples across multiple
        modalities.

        Args:
            sim_mats (dict[str, BatBatTensor]): A dictionary of each modals
                similarity matrix.
            sim_mat_masks (dict[str, BatBatTensor]): A dictionary of each
                modals similiarity matrix mask.
            batch_size (int): Size of the batch.

        Returns:
            BatBatTensor: A matrix describing the similarity between all
                samples in the batch across multiple modalities.
        """

        filt_sim_mat = torch.zeros(batch_size,
                                   batch_size,
                                   dtype=torch.float32,
                                   device=self.device)
        filt_sim_mat_mask = torch.zeros_like(filt_sim_mat, dtype=torch.bool)
        for sim_mat, sim_mat_mask in zip(sim_mats.values(), sim_mat_masks.values()):
            filt_sim_mat = filt_sim_mat + sim_mat
            filt_sim_mat_mask = filt_sim_mat_mask + sim_mat_mask

        filt_sim_mat = filt_sim_mat / filt_sim_mat_mask
        filt_sim_mat = \
            filt_sim_mat * (filt_sim_mat > F.sigmoid(self.dissimilar_thresh))

        return filt_sim_mat

    def _calc_modal_auxillary(self,
                              modal_name: str,
                              emb: tt.BatEmbTensor,
                              filt_sim_mat: tt.BatBatTensor
                              ) -> tt.BatEmbTensor:
        """
        Calculate auxillary information using graph convolutions.

        Args:
            modal_name (str): Name of the modal.
            emb (BatEmbTensor): Embedded modality.
            filt_sim_mat (BatBatTensor): Filtered similarity matrix.

        Returns:
            BatEmbTensor: Auxillary information to address missing modalities.
        """

        return F.relu(
            self.graph_net[modal_name](emb, filt_sim_mat)
            )

    def _adapative_imputation(self,
                              modal_name: str,
                              emb: tt.BatEmbTensor,
                              mask: tt.BatTensor,
                              aux: tt.BatEmbTensor) -> tt.BatEmbTensor:
        self_info = F.sigmoid(
            self.adapt_self[modal_name](emb)
        )
        other_info = F.sigmoid(
            self.adapt_other[modal_name](emb)
        )

        self_info = self_info / (self_info + other_info)
        other_info = 1 - self_info

        def logGrad(lbl: str, v: Tensor):
            print(f"{lbl} GRAD_FN ({v.grad_fn})\nCHILDREN: {[c[0] for c in v.grad_fn.next_functions]}\n{'-'*20}")
        logGrad('SELF_INFO', self_info)
        logGrad('EMB', emb)
        logGrad('OTHER_INFO', other_info)
        logGrad('AUX', aux)

        mask = mask.unsqueeze(-1)
        impute = (self_info * emb) + (other_info * aux)
        impute = (mask * impute) + (~mask * aux)

        return impute
