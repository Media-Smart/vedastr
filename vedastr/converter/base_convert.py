from .registry import CONVERTERS
import abc


@CONVERTERS.register_module
class BaseConverter(object):

    def __init__(self, character):
        self.character = list(character)
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    @abc.abstractmethod
    def train_encode(self, *args, **kwargs):
        '''encode text in train phase'''

    @abc.abstractmethod
    def test_encode(self, *args, **kwargs):
        '''encode text in test phase'''

    @abc.abstractmethod
    def decode(self, *args, **kwargs):
        '''decode label to text in train and test phase'''
