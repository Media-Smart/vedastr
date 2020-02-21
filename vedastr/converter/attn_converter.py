# modify from clovaai

import torch

from .registry import CONVERTERS
from .base_convert import BaseConverter


@CONVERTERS.register_module
class AttnConverter(BaseConverter):
    def __init__(self, character, batch_max_length):
        list_token = ['[GO]', '[s]']
        list_character = list(character)
        super(AttnConverter, self).__init__(character=list_token + list_character,
                                            batch_max_length=batch_max_length)
        self.dict = {v: k - 1 for v, k in self.dict.items()}
        self.batch_max_length += 1

    def train_encoder(self, text):
        length = [len(s) + 1 for s in text]
        batch_text = torch.LongTensor(len(text), self.batch_max_length + 1).fill_(0)
        for idx, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[idx][1:1 + len(text)] = torch.LongTensor(text)
        batch_text_input = batch_text[:, :-1]
        batch_text_target = batch_text[:, 1:]

        return (batch_text_input, torch.IntTensor(length), batch_text_target)

    def test_encoder(self, text):
        batch_text = torch.LongTensor(len(text), 1).fill_(0)
        length = [1 for i in range(len(text))]

        return (batch_text, torch.IntTensor(length), batch_text)

    def decoder(self, text_index):
        texts = []
        batch_size = text_index.shape[0]
        for index in range(batch_size):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            text = text[:text.find('[s]')] if text.find('[s]') != -1 else text
            texts.append(text)

        return texts

    def train_decoder(self, text_index, *args):

        return self.decoder(text_index)

    def test_decoder(self, text_index, *args):

        return self.decoder(text_index)
