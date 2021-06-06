import abc

import torch

from .registry import CONVERTERS


@CONVERTERS.register_module
class BaseConverter(object):
    def __init__(self, character):
        self.character = list(character)
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i
        self.ignore_index = None

    @abc.abstractmethod
    def train_encode(self, *args, **kwargs):
        '''encode text in train phase'''

    def test_encode(self, text):
        if isinstance(text, (list, tuple)):
            num = len(text)
        elif isinstance(text, int):
            num = text
        else:
            raise TypeError(
                f'Type of text should in (list, tuple, int) '
                f'but got {type(text)}'
            )
        ignore_index = self.ignore_index
        if ignore_index is None:
            ignore_index = 0
        batch_text = torch.LongTensor(num, 1).fill_(ignore_index)
        length = [1 for i in range(num)]

        return batch_text, torch.IntTensor(length), batch_text

    @abc.abstractmethod
    def decode(self, *args, **kwargs):
        '''decode label to text in train and test phase'''
