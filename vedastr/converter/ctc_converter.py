# modify from clovaai

import torch

from .registry import CONVERTERS
from .base_convert import BaseConverter


@CONVERTERS.register_module
class CTCConverter(BaseConverter):
    def __init__(self, character, batch_max_length):
        super(CTCConverter, self).__init__(character=character,
                                           batch_max_length=batch_max_length)
        self.character = ['[blank]'] + self.character

    def encoder(self, text):
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length), torch.IntTensor(text))

    def decoder(self, text_index, length):
        text_index = text_index.view(-1)
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for idx in range(l):
                if t[idx] != 0 and (not (idx > 0 and t[idx - 1] == t[idx])):
                    char_list.append(self.character[t[idx]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts

    def train_encoder(self, text):
        return self.encoder(text)

    def test_encoder(self, text):
        return self.encoder(text)

    def train_decoder(self, text_index, length):
        return self.decoder(text_index, length)

    def test_decoder(self, text_index, length):
        return self.decoder(text_index, length)

