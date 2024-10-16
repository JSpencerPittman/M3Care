import torch
from torch import nn
from torch.nn import functional as F

from src.raindrop.raindrop import Raindrop


class RaindropClassifier(nn.Module):
    def __init__(self,
                 rd_model: Raindrop,
                 static_dim: int,
                 static_proj_dim: int,
                 cls_hidden_dim: int,
                 classes: int):
        super().__init__()
        self.static_dim = static_dim
        self.static_proj_dim = static_proj_dim
        self.cls_hidden_dim = cls_hidden_dim
        self.classes = classes

        self.rd_model = rd_model
        self.static_proj = nn.Linear(static_dim, static_proj_dim)

        rd_out_dim = self.rd_model.out_dim * self.rd_model.num_sensors
        self.cls = nn.Sequential(
            nn.Linear(rd_out_dim + static_proj_dim, cls_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(cls_hidden_dim, classes),
            nn.LeakyReLU())


    def forward(self, x_ts, times, mask, x_static):
        ts_emb, reg_loss = self.rd_model(x_ts, times, mask)
        ts_emb = ts_emb.view(ts_emb.shape[0], -1)

        static_emb = F.leaky_relu(self.static_proj(x_static))

        emb = torch.concat([ts_emb, static_emb], dim=-1) 
        return F.softmax(self.cls(emb), dim=-1), reg_loss

