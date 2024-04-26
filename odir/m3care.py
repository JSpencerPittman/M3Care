import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

from src.model.transformer import NatLangTransformer, MultiModalTransformer
from src.embed.pos_encode import PositionalEncoding
from src.component.graph import GraphConvolution
from src.utils import init_weights, clones, length_to_mask, guassian_kernel


class M3Care(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_size, vocab, device, output_dim=1, keep_prob=1):
        super(M3Care, self).__init__()

        # hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.keep_prob = keep_prob
        self.modal_num = 3
        self.device = device

        self.NLP_model = NatLangTransformer(embed_size=embed_size,
                                            hidden_size=hidden_dim,
                                            dropout_rate=1 - self.keep_prob,
                                            vocab=vocab, devicename=device).to(device)

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)

        self.res2hidden1 = nn.Linear(1000, self.hidden_dim)
        self.MM_model = MultiModalTransformer(input_dim=self.hidden_dim, d_model=self.hidden_dim,
                                              MHD_num_head=4, d_ff=self.hidden_dim*4, output_dim=1).to(device)
        self.MM_model2 = MultiModalTransformer(input_dim=self.hidden_dim, d_model=self.hidden_dim,
                                               MHD_num_head=1, d_ff=self.hidden_dim*4, output_dim=1).to(device)

        self.token_type_embeddings = nn.Embedding(6, self.hidden_dim)
        self.token_type_embeddings.apply(init_weights)

        self.sep_token_embeddings = nn.Embedding(6, self.hidden_dim)
        self.sep_token_embeddings.apply(init_weights)

        self.PositionalEncoding = PositionalEncoding(
            self.hidden_dim, dropout=0, max_len=5000)

        self.dropout = nn.Dropout(p=1 - self.keep_prob)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.proj1 = nn.Linear(self.hidden_dim * 5, self.hidden_dim*2)
        self.proj2 = nn.Linear(self.hidden_dim * 5, self.hidden_dim * 5)
        self.out_layer = nn.Linear(self.hidden_dim * 2, self.output_dim)

        self.threshold = nn.Parameter(torch.ones(size=(1,))+1)
        self.simiProj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.resnet18 = models.resnet18()

        self.selu = nn.SELU()

        self.bn = nn.BatchNorm1d(self.hidden_dim)

        self.simiProj = clones(torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            self.relu,
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            self.relu,
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
        ), self.modal_num)

        self.GCN = clones(GraphConvolution(
            self.hidden_dim, self.hidden_dim, bias=True), self.modal_num)
        self.GCN_2 = clones(GraphConvolution(
            self.hidden_dim, self.hidden_dim, bias=True), self.modal_num)
        self.GCN_3 = clones(GraphConvolution(
            self.hidden_dim, self.hidden_dim, bias=True), self.modal_num)

        self.eps0 = nn.Parameter(torch.ones(size=(1,))+1)
        self.eps1 = nn.Parameter(torch.ones(size=(1,))+1)
        self.eps2 = nn.Parameter(torch.ones(size=(1,))+1)

        self.weight1 = clones(nn.Linear(self.hidden_dim, 1), self.modal_num)
        self.weight2 = clones(nn.Linear(self.hidden_dim, 1), self.modal_num)

    def forward(self, tabular, left_fundus, right_fundus, left_diag, right_diag, l_r_masks):
        ### ----- Unimodal Feature Extraction ----- ###

        # Tabular Modality
        tabular_hidden = self.relu(self.linear1(tabular))  # b 1
        tabular_mask = length_to_mask(torch.ones((tabular_hidden.shape[0], 1)).int(
        ).squeeze()).unsqueeze(1).to(self.device).int()

        # Natural Language Modality
        left_diag_contexts, left_diag_lens = self.NLP_model(left_diag)  # b t h
        right_diag_contexts, right_diag_lens = self.NLP_model(
            right_diag)  # b t h

        left_diag_contexts = self.relu(left_diag_contexts)
        right_diag_contexts = self.relu(right_diag_contexts)

        left_diag_mask = length_to_mask(torch.from_numpy(
            np.array(left_diag_lens))).unsqueeze(1).to(self.device)
        right_diag_mask = length_to_mask(torch.from_numpy(
            np.array(right_diag_lens))).unsqueeze(1).to(self.device)

        # Image Modality
        left_image = self.relu(self.res2hidden1(self.resnet18(left_fundus)))
        right_image = self.relu(self.res2hidden1(self.resnet18(right_fundus)))

        left_image_mask = length_to_mask(
            torch.ones((tabular_hidden.shape[0], 1)).int().squeeze()).unsqueeze(1).to(self.device).int()
        right_image_mask = length_to_mask(torch.ones(
            (tabular_hidden.shape[0], 1)).int().squeeze()).unsqueeze(1).to(self.device).int()

        ### ----- Assimilate Unimodal Representations ----- ###
        # Tabular modality
        tabular_hidden00 = tabular_hidden

        # Textual modality
        left_diag_hidden00 = torch.zeros_like(left_diag_contexts[:, 0])
        right_diag_hidden00 = torch.zeros_like(left_diag_contexts[:, 0])
        for j in range(tabular_hidden00.shape[0]):
            left_diag_hidden00[j] = left_diag_contexts[j, 0]
            right_diag_hidden00[j] = right_diag_contexts[j, 0]

        # Image modality
        left_image_hidden00 = torch.zeros_like(left_diag_contexts[:, 0])
        right_image_hidden00 = torch.zeros_like(left_diag_contexts[:, 0])
        for j in range(tabular_hidden00.shape[0]):
            left_image_hidden00[j] = left_image[j]
            right_image_hidden00[j] = right_image[j]

        # Missing Modality matrix
        left_diag_mask_ = torch.from_numpy(np.array(l_r_masks[0])).to(
            self.device).unsqueeze(1)  # b 1
        right_diag_mask_ = torch.from_numpy(
            np.array(l_r_masks[1])).to(self.device).unsqueeze(1)

        left_diag_mask2 = left_diag_mask_ * left_diag_mask_.permute(1, 0)
        right_diag_mask2 = right_diag_mask_ * right_diag_mask_.permute(1, 0)

        ### ----- Calculate Similarity between Modalities ----- ###
        # Tabular modality
        tabular_hidden_mat = guassian_kernel(self.bn(
            self.relu(self.simiProj[0](tabular_hidden00))), kernel_mul=2.0, kernel_num=3)
        tabular_hidden_mat2 = guassian_kernel(
            self.bn(tabular_hidden00), kernel_mul=2.0, kernel_num=3)

        tabular_hidden_mat = ((1-self.sigmoid(self.eps0)) *
                              tabular_hidden_mat+self.sigmoid(self.eps0))*tabular_hidden_mat2

        # Natural Language modality
        left_diag_hidden_mat = guassian_kernel(self.bn(
            self.relu(self.simiProj[1](left_diag_hidden00))), kernel_mul=2.0, kernel_num=3)
        right_diag_hidden_mat = guassian_kernel(self.bn(self.relu(
            self.simiProj[1](right_diag_hidden00))), kernel_mul=2.0, kernel_num=3)

        right_diag_hidden_mat2 = guassian_kernel(
            self.bn(right_diag_hidden00), kernel_mul=2.0, kernel_num=3)
        left_diag_hidden_mat2 = guassian_kernel(
            self.bn(left_diag_hidden00), kernel_mul=2.0, kernel_num=3)

        left_diag_hidden_mat = ((1-self.sigmoid(self.eps1)) *
                                left_diag_hidden_mat+self.sigmoid(self.eps1))*left_diag_hidden_mat2
        right_diag_hidden_mat = (
            (1-self.sigmoid(self.eps1))*right_diag_hidden_mat+self.sigmoid(self.eps1))*right_diag_hidden_mat2

        left_diag_hidden_mat = left_diag_hidden_mat*left_diag_mask2
        right_diag_hidden_mat = right_diag_hidden_mat*right_diag_mask2

        # Image modality
        left_image_hidden_mat = guassian_kernel(self.bn(
            self.relu(self.simiProj[2](left_image_hidden00))), kernel_mul=2.0, kernel_num=3)
        right_image_hidden_mat = guassian_kernel(self.bn(
            self.relu(self.simiProj[2](right_image_hidden00))), kernel_mul=2.0, kernel_num=3)

        left_image_hidden_mat2 = guassian_kernel(
            self.bn(left_image_hidden00), kernel_mul=2.0, kernel_num=3)
        right_image_hidden_mat2 = guassian_kernel(
            self.bn(right_image_hidden00), kernel_mul=2.0, kernel_num=3)

        left_image_hidden_mat = ((1-self.sigmoid(self.eps2)) *
                                 left_image_hidden_mat+self.sigmoid(self.eps2))*left_image_hidden_mat2
        right_image_hidden_mat = ((1-self.sigmoid(self.eps2)) *
                                  right_image_hidden_mat+self.sigmoid(self.eps2))*right_image_hidden_mat2

        ### ----- Stabilize learned representations --- ###
        diff1 = torch.abs(torch.norm(self.simiProj[0](
            tabular_hidden00)) - torch.norm(tabular_hidden00))
        diff2 = torch.abs(torch.norm(self.simiProj[1](
            left_diag_hidden00)) - torch.norm(left_diag_hidden00))
        diff22 = torch.abs(torch.norm(self.simiProj[1](
            right_diag_hidden00)) - torch.norm(right_diag_hidden00))
        diff3 = torch.abs(torch.norm(self.simiProj[2](
            left_image_hidden00)) - torch.norm(left_image_hidden00))
        diff33 = torch.abs(torch.norm(self.simiProj[2](
            right_image_hidden00)) - torch.norm(right_image_hidden00))

        # Calculate L_stab
        sum_of_diff = diff1+diff2+diff22+diff3+diff33

        ### ----- Filtered Similarity Matrix ----- ###
        similar_score = (tabular_hidden_mat + left_diag_hidden_mat + right_diag_hidden_mat + left_image_hidden_mat +
                         right_image_hidden_mat) / \
            (torch.ones_like(right_diag_mask2) + torch.ones_like(right_diag_mask2) + torch.ones_like(right_diag_mask2)
             + right_diag_mask2 + left_diag_mask2)

        similar_score = self.relu(
            similar_score - self.sigmoid(self.threshold)[0])
        temp_thresh = self.sigmoid(self.threshold)[0]
        bin_mask = similar_score > 0
        similar_score = similar_score + bin_mask * temp_thresh.detach()

        ### ----- Use similar patients to impute natural language modality ----- ###
        # Run GCN to aggregate data from similar patients in the graph
        left_diag_hidden0 = self.relu(self.GCN[0](
            similar_score*left_diag_mask2, left_diag_hidden00))
        right_diag_hidden0 = self.relu(self.GCN[1](
            similar_score*right_diag_mask2, right_diag_hidden00))

        left_diag_hidden0 = self.relu(self.GCN_2[0](
            similar_score*left_diag_mask2, left_diag_hidden0))
        right_diag_hidden0 = self.relu(self.GCN_2[1](
            similar_score*right_diag_mask2, right_diag_hidden0))

        # Calculate the tradeoff of imputed vs original data
        left_diag_weight1 = torch.sigmoid(self.weight1[0](left_diag_hidden0))
        right_diag_weight1 = torch.sigmoid(self.weight1[1](right_diag_hidden0))

        left_diag_weight2 = torch.sigmoid(self.weight2[0](left_diag_hidden00))
        right_diag_weight2 = torch.sigmoid(
            self.weight2[1](right_diag_hidden00))

        left_diag_weight1 = left_diag_weight1 / \
            (left_diag_weight1+left_diag_weight2)
        right_diag_weight1 = right_diag_weight1 / \
            (right_diag_weight1+right_diag_weight2)

        left_diag_weight2 = 1-left_diag_weight1
        right_diag_weight2 = 1-right_diag_weight1

        # Compute hybrid imputation of natural language modality
        final_left_diag = left_diag_weight1*left_diag_hidden0 + \
            left_diag_weight2*left_diag_hidden00
        final_right_diag = right_diag_weight1*right_diag_hidden0 + \
            right_diag_weight2*right_diag_hidden00

        # if missing value only use the imputed otherwise hybrid
        left_diag_contexts_ = torch.zeros_like(left_diag_contexts)
        right_diag_contexts_ = torch.zeros_like(right_diag_contexts)

        for i in range(tabular_hidden00.shape[0]):
            left_diag_contexts_[i] = left_diag_contexts[i]
            right_diag_contexts_[i] = right_diag_contexts[i]

            if right_diag_mask_[i][0] != 1:
                right_diag_contexts_[i, 0] = right_diag_hidden0[i]
            else:
                right_diag_contexts_[i, 0] = final_right_diag[i]

            if left_diag_mask_[i][0] != 1:
                left_diag_contexts_[i, 0] = left_diag_hidden0[i]
            else:
                left_diag_contexts_[i, 0] = final_left_diag[i]

        ### ----- Context-aware multimodal interaction capture (sequence) ----- ###
        # Tabular modal
        values_hidden = values_hidden.unsqueeze(1) +  \
            self.token_type_embeddings(torch.zeros_like(
                tabular_mask.permute(0, 2, 1).squeeze(-1)).to(self.device).long())

        # Textual modal
        left_diag_contexts = self.  (left_diag_contexts_) +  \
            self.token_type_embeddings(torch.ones_like(
                left_diag_mask.permute(0, 2, 1).squeeze(-1)).to(self.device).long())

        right_diag_contexts = self.PositionalEncoding(right_diag_contexts_) +  \
            self.token_type_embeddings(
                2*torch.ones_like(right_diag_mask.permute(0, 2, 1).squeeze(-1)).to(self.device).long())

        # Image modal
        left_image_contexts = left_image.unsqueeze(1) +  \
            self.token_type_embeddings(
                3 * torch.ones_like(left_image_mask.permute(0, 2, 1).squeeze(-1)).to(self.device).long())

        right_image_contexts = right_image.unsqueeze(1) +  \
            self.token_type_embeddings(
                4 * torch.ones_like(right_image_mask.permute(0, 2, 1).squeeze(-1)).to(self.device).long())

        z0 = torch.cat([values_hidden, left_diag_contexts, right_diag_contexts, left_image_contexts,
                        right_image_contexts], dim=1)
        z0_mask = torch.cat([tabular_mask.int(), left_diag_mask.int(), right_diag_mask.int(), left_image_mask.int(),
                             right_image_mask.int()], dim=-1).int()

        z1 = self.relu(self.MM_model(z0, z0_mask))  # b 1

        z2 = self.relu(self.MM_model2(z1, z0_mask))  # b 1

        val_hidden = z2[:, 0, :]
        left_diag_hidden = []
        right_diag_hidden = []
        left_image_hidden = []
        right_image_hidden = []

        for j in range(z2.shape[0]):
            left_diag_hidden.append(z2[j, 1])
            right_diag_hidden.append(z2[j, 1 + left_diag_contexts.shape[1]])
            left_image_hidden.append(
                z2[j, 1 + left_diag_contexts.shape[1] + right_diag_contexts.shape[1]])
            right_image_hidden.append(z2[j, 1 + left_diag_contexts.shape[1] + right_diag_contexts.shape[1]
                                         + left_image_contexts.shape[1]])

        left_diag_hidden = torch.stack(left_diag_hidden)
        right_diag_hidden = torch.stack(right_diag_hidden)
        left_image_hidden = torch.stack(left_image_hidden)
        right_image_hidden = torch.stack(right_image_hidden)

        combined_hidden = torch.cat((val_hidden, left_diag_hidden, right_diag_hidden, left_image_hidden,
                                     right_image_hidden), -1)  # b n h

        last_hs_proj = self.dropout(F.relu(self.proj1(combined_hidden)))

        output = self.out_layer(last_hs_proj).squeeze(-1)

        return output, sum_of_diff
