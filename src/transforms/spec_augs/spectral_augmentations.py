from torch import Tensor, nn
import torchaudio

class SpecAug(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        freq_mask = kwargs.get('freq_mask', 20)
        time_mask = kwargs.get('time_mask', 50)
        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask),
            torchaudio.transforms.TimeMasking(time_mask),
        )

    def __call__(self, data: Tensor):
        return self.specaug(data)