from torch import Tensor, nn


class Normalize1d(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, data: Tensor):
        return (data - self.mean) / self.std
