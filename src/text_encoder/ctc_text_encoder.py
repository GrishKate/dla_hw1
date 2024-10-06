import re
import os
from string import ascii_lowercase
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

import torch


# TODO add CTC decode - done
# TODO add BPE - done, LM, Beam Search support - done
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        self.use_bpe = kwargs.get('use_bpe', False)

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")
        if self.use_bpe:
            vocab_size = kwargs.get('vocab_size', 100)
            data_file = kwargs.get('data_file', 'translations.txt')
            if data_file is None:
                raise Exception('Need data file for bpe')
            self.train_bpe(data_file=data_file, vocab_size=vocab_size)
        else:
            self.alphabet = alphabet
            self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
            self.ind2char = dict(enumerate(self.vocab))
            self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            if not self.use_bpe:
                return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
            else:
                return torch.Tensor(self.sp_model.encode(text)).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char_ind = self.EMPTY_TOK
        for ind in inds:
            if last_char_ind == ind:
                continue
            if ind != self.EMPTY_TOK:
                decoded.append(self.ind2char[ind])
            last_char_ind = ind
        if last_char_ind != self.EMPTY_TOK:
            decoded.append(self.ind2char[ind])
        return ''.join(decoded)

    def get_string(self, string, ind):
        new_token = self.ind2char[ind]
        if len(string) == 1 and string == self.EMPTY_TOK:
            new_str = new_token
        elif new_token == self.EMPTY_TOK:
            new_str = string
        else:
            sim = similar(string, new_token)
            new_str = string + new_token[sim:]
        return new_str

    def ctc_beam_decode(self, probs, k=3, beam_len=10):
        # probs (seq, classes)
        best_string = ''
        start = 0
        current = torch.topk(probs[start, :], k)
        cur_ind = current.indices.cpu().numpy()
        decoded_strings = {self.get_string(best_string, cur_ind[i]): current.values[i]
                           for i in range(len(cur_ind))}
        for i in range(1, probs.shape[0]):
            current = torch.topk(probs[i, :], k)
            cur_ind = current.indices.cpu().numpy()
            new_decoded_strings = {}
            for j in range(k):
                for string in decoded_strings.keys():
                    new_str = self.get_string(string, cur_ind[j])
                    new_prob = decoded_strings[string] * current.values[j]
                    if new_str not in new_decoded_strings.keys():
                        new_decoded_strings[new_str] = 0
                    new_decoded_strings[new_str] += new_prob
            decoded_strings = truncate_path(new_decoded_strings, beam_len)
        # find string with maximum probability
        cur_max = -1
        for string in decoded_strings.keys():
            if decoded_strings[string] > cur_max:
                cur_max = decoded_strings[string]
                best_string = string
        return best_string

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def train_bpe(self, data_file: str, sp_model_prefix: str = 'bpe_file',
                  vocab_size: int = 250, normalization_rule_name: str = 'nmt_nfkc_cf',
                  model_type: str = 'bpe'):
        if not os.path.isfile(sp_model_prefix + '.model'):
            # train tokenizer if not trained yet
            SentencePieceTrainer.train(
                input=data_file, vocab_size=vocab_size,
                model_type=model_type, model_prefix=sp_model_prefix,
                normalization_rule_name=normalization_rule_name,
                pad_id=0, unk_id=1, bos_id=2, eos_id=3
            )
            # load tokenizer from file
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + '.model')
        self.alphabet = [self.sp_model.id_to_piece(id) for id in range(self.sp_model.get_piece_size())]
        self.vocab = list(self.alphabet) + [self.EMPTY_TOK]
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_model.pad_id(), self.sp_model.unk_id(), \
            self.sp_model.bos_id(), self.sp_model.eos_id()


def truncate_path(d, beam_len):
    return dict(sorted(list(d.items()), key=lambda x: -x[1])[:beam_len])


def similar(str1, str2):
    if str1 == '':
        return 0
    cnt = 0
    for i in range(len(str2)):
        if str2[i] == str1[-i - 1]:
            cnt += 1
    return cnt
