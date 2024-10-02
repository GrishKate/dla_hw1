import torch
from torch import Tensor, nn


class Noise(nn.Module):
    def __init__(self, mean=0, std=0.02):
        super().__init__()
        self.noiser = torch.distributions.Normal(mean, std)

    def __call__(self, data: Tensor):
        return data + self.noiser.sample(data.size())