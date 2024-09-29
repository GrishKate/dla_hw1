import torch
import torch.nn as nn
from src.model import DeepSpeech2

n_feats = 80
seq_len = 100
batch_size = 2
n_tokens = 10

criterion = nn.CTCLoss(blank=3, zero_infinity=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepSpeech2(n_feats=n_feats, n_tokens=n_tokens, n_recurrent=3).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-04)

for i in range(5):
    spectrogram = torch.randn(batch_size, n_feats, seq_len)
    lengths = torch.tensor([80, 51])
    target = torch.randint(low=0, high=n_tokens, size=(batch_size, n_tokens)).to(device)
    target_len = torch.tensor([4, 3])
    res_dict = model(spectrogram, lengths)
    outp, outp_len = res_dict['log_probs'], res_dict['log_probs_length']
    print(outp.shape, outp_len)
    loss = criterion(outp.transpose(0, 1), target[:, 1:], outp_len, target_len)
    loss.backward()
    optimizer.step()
    print(loss)
