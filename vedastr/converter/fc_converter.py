import torch

from .registry import CONVERTERS
from .base_convert import BaseConverter

@CONVERTERS.register_module
class FCConverter(BaseConverter):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [s] for end-of-sentence token.
        list_token = ['[s]']
        ignore_token = ['[ignore]']
        list_character = list(character)
        self.character = list_token + list_character + ignore_token
        self.ignore_index = len(self.character) - 1
        self.num_class = len(self.character) - 1

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(self.ignore_index)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        batch_text_input = batch_text
        batch_text_target = batch_text
        return (batch_text_input, torch.IntTensor(length), batch_text_target)

    def train_encoder(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def test_encoder(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            text = text[:text.find('[s]')]
            texts.append(text)
        return texts

    def test_decoder(self, *args, **kwargs):
        return self.decode(*args, **kwargs)

    def train_decoder(self, *args, **kwargs):
        return self.decode(*args, **kwargs)
