import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, char_embed_size, num_filters):
        super(CNN, self).__init__()
        self.kernel_size = 5
        self.conv = nn.Conv1d(in_channels=char_embed_size, out_channels=num_filters, kernel_size=self.kernel_size)
        # self.maxpool = nn.MaxPool1d(kernel_size=max_word_length - self.kernel_size + 1)

    def forward(self, x):
        # x is of size (batch_size, char_embed_size, max_word_length)
        x_conv = self.conv(x)
        # x_conv is of size (batch_size, num_filters, max_word_length - kernel_size + 1)
        # x_conv_out = torch.squeeze(self.maxpool(F.relu(x_conv)), dim=2)
        x_conv_out = torch.max(F.relu(x_conv), dim=2)[0] # (batch_size, num_filters)
        return x_conv_out
