from torch import nn, Tensor


class M3Care(nn.Module):
    def __init__(self,
                 dem_mdl: nn.Module,
                 vit_mdl: nn.Module,
                 itv_mdl: nn.Module,
                 nst_mdl: nn.Module,
                 nts_mdl: nn.Module):
        super(M3Care, self).__init__()

        # Store models for each modality
        self.dem_mdl = dem_mdl
        self.vit_mdl = vit_mdl
        self.itv_mdl = itv_mdl
        self.nst_mdl = nst_mdl
        self.nts_mdl = nts_mdl

    def forward(self, dem, vit, itv, nst, nts):
        ### ----- Unimodal Feature Extraction ----- ###

        dem_emb = self.dem_mdl(dem)

        vit_emb = self.vit_mdl(vit)

        itv_emb = self.itv_mdl(itv)

        nst_emb = self.nst_mdl(nst)

        nts_emb = self.nts_mdl(nts)
