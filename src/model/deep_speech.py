import torch
from torch import nn
import torch.nn.functional as F


class CReLU(nn.Module):
    """
    Clipped ReLU
    """

    def __init__(self, maximum=20):
        super().__init__()
        self.maximum = maximum

    def forward(self, x):
        return torch.clamp(F.relu(x), min=0, max=self.maximum)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)):
        super().__init__()
        self.padding = padding[1]
        self.dilation = 1
        self.kernel_size = kernel_size[1]
        self.stride = float(stride[1])
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = CReLU()

    def forward(self, inp):
        # input (batch, ch, freq_dim, time_seq)
        spec, seq_len = inp
        seq_len = ((seq_len.float() + 2 * self.padding - self.dilation *
                    (self.kernel_size - 1) - 1) / self.stride + 1).int()
        return self.act(self.bn(self.conv(spec))), seq_len


class RecurrentBlock(nn.Module):
    def __init__(self, inp_dim, outp_dim, dropout=0.1):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()
        self.rnn = nn.GRU(inp_dim, outp_dim, num_layers=1, bidirectional=True,
                          bias=True, dropout=dropout, batch_first=True)
        self.bn = nn.BatchNorm1d(inp_dim)
        self.act = CReLU()

    def forward(self, inp):
        # spec of shape (batch, seq, dim)
        spec, spec_lengths = inp
        length = spec.shape[1]
        out = self.act(self.bn(spec.transpose(1, 2))).transpose(1, 2)
        out = nn.utils.rnn.pack_padded_sequence(out, spec_lengths.cpu(), batch_first=True,
                                                enforce_sorted=False)
        out, _ = self.rnn(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=length, batch_first=True)
        return out, spec_lengths


class DeepSpeech2(nn.Module):
    """
    Deep Speech v2 model
    """

    def __init__(self, n_feats, n_tokens, n_recurrent=10):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()
        self.factor = 8
        modules_list = []
        # conv (batch ch freq_dim time)
        modules_list.append(ConvBlock(1, 16, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)))
        modules_list.append(ConvBlock(16, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)))
        modules_list.append(ConvBlock(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)))
        self.conv = nn.Sequential(*modules_list)
        dim = (n_feats // 8) * 32
        modules_list = []
        hidden = dim
        for i in range(n_recurrent):
            modules_list.append(RecurrentBlock(hidden, dim))
            hidden = 2 * dim
        dim = hidden
        self.recurrent = nn.Sequential(*modules_list)
        self.bn = nn.BatchNorm1d(dim)
        self.fc = nn.Linear(dim, n_tokens)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor) (batch, inp_dim, seq): input spectrogram.
            spectrogram_length (Tensor) (batch): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        # spectrogram (batch freq_dim time) -> (batch 1 freq_dim time)
        out_spec, out_len = self.conv((spectrogram.unsqueeze(1), spectrogram_length))
        b, ch, freq_dim, time_seq = out_spec.shape
        out_spec = out_spec.permute(0, 3, 1, 2).reshape(b, time_seq, ch * freq_dim)
        # mask
        out_spec, out_len = self.recurrent((out_spec, out_len))
        out_spec = self.bn(out_spec.transpose(1, 2)).transpose(1, 2)
        output = self.fc(out_spec)
        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = out_len  # self.transform_input_lengths(spectrogram_length)
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
        return input_lengths // self.factor

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
