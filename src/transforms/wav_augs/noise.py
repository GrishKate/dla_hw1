import torch
from torch import Tensor, nn
import random


class Noise(nn.Module):
    def __init__(self, p=0.0, mean=0, std=0.02):
        super().__init__()
        self.p = p
        self.noiser = torch.distributions.Normal(mean, std)

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            return data + self.noiser.sample(data.size())
        return data
