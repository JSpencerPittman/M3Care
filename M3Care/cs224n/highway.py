import torch.nn.functional as F
import torch.nn as nn
import torch

class Highway(nn.Module):
    def __init__(self, size):
        super(Highway, self).__init__()

        self.linear = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)

    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        nonlinear = F.relu(self.linear(x))
        output = gate * nonlinear + (1-gate) * x
        return output