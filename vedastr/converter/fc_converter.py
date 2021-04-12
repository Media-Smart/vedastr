import torch

from .base_convert import BaseConverter
from .registry import CONVERTERS


@CONVERTERS.register_module
class FCConverter(BaseConverter):

    def __init__(self, character, batch_max_length=25):

        list_token = ['[s]']
        ignore_token = ['[ignore]']
        list_character = list(character)
        self.batch_max_length = batch_max_length + 1
        super(FCConverter, self).__init__(character=list_token + list_character + ignore_token)  # noqa 501
        self.ignore_index = self.dict[ignore_token[0]]

    def encode(self, text):
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.ignore_index)  # noqa 501
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        batch_text_input = batch_text
        batch_text_target = batch_text

        return batch_text_input, torch.IntTensor(length), batch_text_target

    def train_encode(self, text):
        return self.encode(text)

    def test_encode(self, text):
        return self.encode(text)

    def decode(self, text_index):
        texts = []
        batch_size = text_index.shape[0]
        for index in range(batch_size):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            text = text[:text.find('[s]')]
            texts.append(text)

        return texts
