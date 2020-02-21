from .registry import CONVERTERS


@CONVERTERS.register_module
class BaseConverter(object):

    def __init__(self, character, batch_max_length=25):
        self.batch_max_length = batch_max_length
        self.character = list(character)
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i + 1

    def train_encoder(self, *args):
        raise NotImplementedError

    def test_encoder(self, *args):
        raise NotImplementedError

    def train_decoder(self, *args, **kwargs):
        raise NotImplementedError

    def test_decoder(self, *args, **kwargs):
        raise NotImplementedError



