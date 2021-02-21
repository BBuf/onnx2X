import torch
from torch import nn


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, input: torch.Tensor, shape):
        return torch.reshape(input, self.shape)
