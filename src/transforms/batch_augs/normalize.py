from torch import Tensor, nn


class Normalize1d(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, data: Tensor):
        return (data - self.mean) / self.std
