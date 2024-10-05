from torchaudio.sox_effects import apply_effects_tensor
from torch import Tensor, nn
import random


class Speed(nn.Module):
    def __init__(self, sample_rate=16000, speed_min=0.8, speed_max=1.2, p=0.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.speed_min = speed_min
        self.speed_max = speed_max

    def __call__(self, data: Tensor):
        if random.random() >= self.p:
            return data
        speed = random.random() / (self.speed_max - self.speed_min) + self.speed_min
        effects = [["speed", str(speed)]]
        data, _ = apply_effects_tensor(data, self.sample_rate, effects)
        return data
