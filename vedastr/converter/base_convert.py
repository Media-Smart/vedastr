from .registry import CONVERTERS


@CONVERTERS.register_module
class BaseConverter(object):

    def __init__(self, character):
        self.character = list(character)
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def train_encode(self, *args):
        raise NotImplementedError

    def test_encode(self, *args):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError
