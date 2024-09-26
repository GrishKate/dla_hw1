import torch_audiomentations
from torch import Tensor, nn


class PitchShift(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.PitchShift(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)

'''
def preprocess_audio(self, wav):
    n = torch.randint(0, 4, 1)
    if n == 0:
        noiser = torch.distributions.Normal(0, 0.05)
        wav = wav + noiser.sample(wav.size())
    if n == 1:
        rate = torch.rand(1) * 1.8
        wav = librosa.effects.time_stretch(wav.numpy().squeeze(), rate=rate)
    if n == 2:
        n_steps = torch.rand(1) * 8 - 4
        wav = librosa.effects.pitch_shift(wav.numpy().squeeze(), sr=self.target_sr, n_steps=n_steps)
    return wav
'''