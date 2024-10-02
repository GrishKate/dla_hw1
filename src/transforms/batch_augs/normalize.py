from torch import Tensor


class Normalize1d:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data: Tensor):
        return (data - self.mean) / self.std
