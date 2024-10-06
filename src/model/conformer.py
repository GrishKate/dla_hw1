import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, dim, drop_p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        lst = []
        lst.append(nn.Conv1d(dim, dim * 2, kernel_size=1))
        lst.append(nn.GLU(dim=1))
        lst.append(nn.Conv1d(dim, dim, kernel_size=31, padding='same', groups=dim))
        lst.append(nn.BatchNorm1d(dim))
        lst.append(nn.SiLU())
        lst.append(nn.Conv1d(dim, dim, kernel_size=1))
        lst.append(nn.Dropout(p=drop_p))
        self.net = nn.Sequential(*lst)

    def forward(self, spectrogram):
        # batch, seq, dim
        out = self.ln(spectrogram).transpose(1, 2)
        return self.net(out).transpose(1, 2)


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop_p=0.1):
        super().__init__()
        lst = []
        lst.append(nn.LayerNorm(dim))
        lst.append(nn.Linear(dim, hidden_dim, bias=True))
        lst.append(nn.SiLU())
        lst.append(nn.Dropout(p=drop_p))
        lst.append(nn.Linear(hidden_dim, dim, bias=True))
        lst.append(nn.Dropout(p=drop_p))
        self.net = nn.Sequential(*lst)

    def forward(self, spectrogram):
        return self.net(spectrogram)


class RelativeAttn(nn.Module):
    def __init__(self, dim, n_heads=4, drop_p=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_head = dim // n_heads

        self.q_pr = nn.Linear(dim, dim)
        self.k_pr = nn.Linear(dim, dim)
        self.v_pr = nn.Linear(dim, dim)
        self.out_pr = nn.Linear(dim, dim)
        self.pos_pr = nn.Linear(dim, dim, bias=False)

        self.u = nn.Parameter(torch.Tensor(self.n_heads, self.dim_head))
        self.v = nn.Parameter(torch.Tensor(self.n_heads, self.dim_head))
        torch.nn.init.xavier_uniform_(self.u)
        torch.nn.init.xavier_uniform_(self.v)

        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=drop_p)

        length = 10000
        enc = torch.empty(length, dim)
        pos = torch.arange(0, length).float()[:, None] / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        enc[:, 0::2] = torch.sin(pos)
        enc[:, 1::2] = torch.cos(pos)
        self.register_buffer('enc', enc)

    def forward(self, spectrogram, mask=None):
        # batch, seq, dim
        b, seq, dim = spectrogram.shape
        out = self.ln(spectrogram)
        pos_emb = self.enc[:seq, :]  # seq, dim
        pos_emb = self.pos_pr(pos_emb).view(1, -1, self.n_heads, self.dim_head).permute(0, 2, 3, 1)

        q = self.q_pr(out).view(b, seq, self.n_heads, self.dim_head)
        k = self.k_pr(out).view(b, seq, self.n_heads, self.dim_head).permute(0, 2, 3, 1)
        v = self.v_pr(out).view(b, seq, self.n_heads, self.dim_head).permute(0, 2, 1, 3)

        qk = torch.matmul((q + self.u).transpose(1, 2), k)
        pos_emb = torch.matmul((q + self.v).transpose(1, 2), pos_emb)
        pos_emb = self.relative_shift(pos_emb)
        qk = (qk + pos_emb) / self.dim ** 0.5

        if mask is not None:
            qk.masked_fill_(mask.unsqueeze(1), -1e9)

        out = torch.matmul(F.softmax(qk, dim=-1), v).transpose(1, 2).reshape(b, -1, self.dim)
        out = self.dropout(self.out_pr(out))
        return out

    def relative_shift(self, pos):
        b, n_heads, s1, s2 = pos.shape
        zero = torch.zeros((b, n_heads, s1, 1), device=pos.device, dtype=pos.dtype)
        padded = torch.cat([zero, pos], dim=-1).reshape(b, n_heads, s2 + 1, s1)
        return padded[:, :, 1:].view_as(pos)


class Block(nn.Module):
    def __init__(self, dim, factor=4, drop_p=0.1):
        super().__init__()
        self.ffn1 = FFN(dim, dim * factor, drop_p)
        self.mhsa = RelativeAttn(dim)
        self.conv = ConvBlock(dim)
        self.ffn2 = FFN(dim, dim * factor, drop_p)
        self.ln = nn.LayerNorm(dim)

    def forward(self, inp):
        spectrogram, mask = inp
        out = spectrogram + 0.5 * self.ffn1(spectrogram)
        out = out + self.mhsa(out, mask=mask)
        out = out + self.conv(out)
        out = out + 0.5 * self.ffn2(out)
        out = self.ln(out)
        return out, mask


class Subsampling(nn.Module):
    def __init__(self, dim, factor=4, drop_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, stride=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2),
            nn.ReLU()
        )

    def forward(self, spectrogram):
        # batch freq time
        out = self.net(spectrogram.unsqueeze(1))  # batch dim freq/4 time/4
        batch, dim, freq, time = out.shape
        out = out.permute(0, 3, 1, 2).reshape(batch, time, -1)
        return out


class Conformer(nn.Module):

    def __init__(self, n_feats, n_tokens, n_blocks, dim=144, drop_p=0.1):
        super().__init__()

        self.conv = Subsampling(dim)
        n_feats = ((n_feats - 1) // 2 - 1) // 2
        self.fc1 = nn.Linear(dim * n_feats, dim)
        self.dropout = nn.Dropout(p=drop_p)
        blocks = [Block(dim) for _ in range(n_blocks)]
        self.net = nn.Sequential(*blocks)
        self.fc_last = nn.Linear(dim, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        # (batch freq_dim time)
        output = self.conv(spectrogram)
        output = self.dropout(self.fc1(output))
        output, _ = self.net((output, None))
        output = self.fc_last(output)
        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return ((input_lengths -1) / 2-1).int()

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
