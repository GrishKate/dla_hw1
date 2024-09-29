import torch
import pickle
from src.text_encoder.ctc_text_encoder import CTCTextEncoder

encoder = CTCTextEncoder()
'''
n_tokens = len(encoder.vocab)
seq = 20
logprobs = torch.randn(seq, n_tokens)
result = encoder.ctc_beam_decode(logprobs, k=3, beam_len=3)
print(result)
'''

# Load precomputed CTC output
with open('./lj_batch.pickle', 'rb') as f:
    batch = pickle.load(f)
# log probabilities of softmax layers [batch_size, T, vocab_size]
log_probs = batch["log_probs"]
# Dictionary with index to character mapping
ind2char = batch["ind2char"]
true_texts = batch["text"]
encoder.EMPTY_TOK = 0
encoder.ind2char = ind2char
# print(log_probs.shape)
for i in range(log_probs.shape[0]):
    print(encoder.ctc_decode(torch.argmax(log_probs[i], dim=-1).numpy()))
    result = encoder.ctc_beam_decode(torch.tensor(log_probs[i].exp()), k=3, beam_len=10)
    print(result)
    print()
