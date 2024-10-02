import torch_audiomentations
from torchaudio.utils import download_asset
from torch import Tensor, nn


class LowPassFilter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.aug_ = torch_audiomentations.LowPassFilter(*args, **kwargs)


    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)