import torch
import numpy as np
from PIL import Image

from .registry import TRANSFORMS

PIL_MODE = {
    'bilinear': Image.BILINEAR,
    'nearest': Image.NEAREST,
    'cubic': Image.CUBIC,
}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


@TRANSFORMS.register_module
class Normalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        assert isinstance(image, torch.Tensor), 'ToTensor should be called before Normalize'

        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=image.device).view(-1, 1, 1)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=image.device).view(-1, 1, 1)

        image.sub_(mean).div_(std)

        return image, label


@TRANSFORMS.register_module
class ToTensor(object):
    def __call__(self, image, label):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))

        nchannel = len(image.mode)
        img = img.view(image.size[1], image.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        return img.float().div(255), label


@TRANSFORMS.register_module
class Resize(object):
    def __init__(self, size, keep_ratio=False, keep_long=False, mode='cubic'):
        self.size = size
        self.keep_ratio = keep_ratio
        self.keep_long = keep_long
        self.mode = mode

    def __call__(self, image, label):
        w, h = image.size
        if self.keep_ratio:
            if self.keep_long:
                long_edge, short_edge = max(self.size), min(self.size)
                scale_factor = min(long_edge / max(h, w), short_edge / min(h, w))
            else:
                scale_factor = min(self.size[0] / h, self.size[1] / w)
            target_size = (int(w * scale_factor), int(h * scale_factor))
            assert 0 not in target_size, "resize shape cannot be 0"
        else:
            target_size = self.size[::-1]
        new_image = image.resize(target_size, resample=PIL_MODE[self.mode])

        return new_image, label


@TRANSFORMS.register_module
class ColorToGray(object):
    def __call__(self, image, label):
        image = image.convert('L')

        return image, label


@TRANSFORMS.register_module
class Sensitive(object):
    def __init__(self, sensitive):
        self.sensitive = sensitive

    def __call__(self, image, label):
        if not self.sensitive:
            label = label.lower()

        return image, label


@TRANSFORMS.register_module
class PadIfNeeded(object):
    def __init__(self, size, pad_value=0):
        self.height = size[0]
        self.width = size[1]
        self.pad_value = pad_value

    def __call__(self, image, label):
        w, h = image.size
        assert h <= self.height and w <= self.width
        assert image.mode in ['RGB', 'L']
        value = self.pad_value
        if image.mode == 'RGB' and isinstance(value, (int, float)):
            value = [self.pad_value] * 3
        value = tuple(value)
        new_image = Image.new(mode=image.mode, size=(self.width, self.height), color=value)
        new_image.paste(image, (0, 0, w, h))

        return new_image, label
