# modify from clovaai

import torch

from .registry import CONVERTERS
from .base_convert import BaseConverter


@CONVERTERS.register_module
class CTCConverter(BaseConverter):
    def __init__(self, character):
        list_token = ['[blank]']
        list_character = list(character)
        super(CTCConverter, self).__init__(character=list_token + list_character)

    def encode(self, text):
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return torch.IntTensor(text), torch.IntTensor(length), torch.IntTensor(text)

    def decode(self, text_index):
        texts = []
        batch_size = text_index.shape[0]
        length = text_index.shape[1]
        for i in range(batch_size):
            t = text_index[i]
            char_list = []
            for idx in range(length):
                if t[idx] != 0 and (not (idx > 0 and t[idx - 1] == t[idx])):
                    char_list.append(self.character[t[idx]])
            text = ''.join(char_list)
            texts.append(text)
        return texts

    def train_encode(self, text):
        return self.encode(text)

    def test_encode(self, text):
        return self.encode(text)
