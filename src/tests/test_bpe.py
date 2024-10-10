import os
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

EMPTY_TOK = ""
data_file = '../../translations.txt'
sp_model_prefix = 'bpe_file'
vocab_size = 250
normalization_rule_name = 'nmt_nfkc_cf'
model_type = 'bpe'
if not os.path.isfile(sp_model_prefix + '.model'):
    # train tokenizer if not trained yet
    SentencePieceTrainer.train(
        input=data_file, vocab_size=vocab_size,
        model_type=model_type, model_prefix=sp_model_prefix,
        normalization_rule_name=normalization_rule_name,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )
    # load tokenizer from file
sp_model = SentencePieceProcessor(model_file=sp_model_prefix + '.model')
alphabet = [sp_model.id_to_piece(id).replace('▁', ' ') for id in range(sp_model.get_piece_size())]
for i in range(4):
    alphabet[i] = ''
vocab = list(alphabet) + [EMPTY_TOK]
ind2char = dict(enumerate(vocab))
char2ind = {v: k for k, v in ind2char.items()}
print(ind2char)
pad_id, unk_id, bos_id, eos_id = sp_model.pad_id(), sp_model.unk_id(), sp_model.bos_id(), sp_model.eos_id()
text = 'how are you'
inds = sp_model.encode(text)
print(inds)
print("".join([ind2char[int(ind)] for ind in inds]).strip())
