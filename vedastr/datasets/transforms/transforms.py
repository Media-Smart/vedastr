import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from .registry import TRANSFORMS

CV2_MODE = {
    'bilinear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'cubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
}

CV2_BORDER_MODE = {
    'constant': cv2.BORDER_CONSTANT,
    'reflect': cv2.BORDER_REFLECT,
    'reflect101': cv2.BORDER_REFLECT101,
    'replicate': cv2.BORDER_REPLICATE,
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
    def __init__(self, mean=(123.675, 116.280, 103.530), std=(58.395, 57.120, 57.375)):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = image.astype(np.float32)
        mean = np.reshape(np.array(self.mean, dtype=image.dtype), [1, 1, 3])
        std = np.reshape(np.array(self.std, dtype=image.dtype), [1, 1, 3])
        denominator = np.reciprocal(std, dtype=image.dtype)

        new_image = (image - mean) * denominator

        return new_image, label


@TRANSFORMS.register_module
class TensorNormalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        assert isinstance(image, torch.Tensor)

        mean = torch.from_numpy(np.reshape(np.array(self.mean, dtype=np.float32),
                                           (image.shape[0], 1, 1)))
        std = torch.from_numpy(np.reshape(np.array(self.std, dtype=np.float32),
                                          (image.shape[0], 1, 1)))
        new_image = image.sub_(mean).div_(std)

        return new_image, label


@TRANSFORMS.register_module
class ToTensor(object):
    def __call__(self, image, label):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.expand_dims(image, -1)
            image = torch.from_numpy(image).permute(2, 0, 1)
        elif isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)

        return image, label


@TRANSFORMS.register_module
class Resize(object):
    def __init__(self, canva_w, canva_h, img_size, keep_ratio=False, keep_long=False):
        self.canva_w = canva_w
        self.canva_h = canva_h
        self.img_size = img_size
        self.keep_ratio = keep_ratio
        self.keep_long = keep_long

    def __call__(self, image, label):
        if isinstance(image, np.ndarray) and self.keep_ratio:
            img_h, img_w, c = image.shape
            if self.keep_long:
                max_long_edge = max(self.img_size)
                max_short_edge = min(self.img_size)
                scale_factor = min(max_long_edge / max(img_h, img_w),
                                   max_short_edge / min(img_h, img_w))
            else:
                scale_factor = min(self.img_size[0]/img_h, self.img_size[1]/img_w)
            canvas = np.zeros((self.canva_h, self.canva_w, c)).astype(np.float32)

            new_image = cv2.resize(image, (int(img_w * scale_factor), int(img_h * scale_factor)))
            if new_image.ndim == 2:
                canvas[:int(img_h * scale_factor), :int(img_w * scale_factor), 0] = new_image
            else:
                canvas[:int(img_h * scale_factor), :int(img_w * scale_factor), :] = new_image

        elif not self.keep_ratio:
            if isinstance(image, np.ndarray):
                canvas = cv2.resize(image, (self.img_size[1], self.img_size[0]), interpolation=CV2_MODE['cubic'])
            elif isinstance(image, Image.Image):
                canvas = image.resize((self.img_size[1], self.img_size[0]), Image.BICUBIC)

        return canvas, label


@TRANSFORMS.register_module
class ColorToGray(object):
    def __call__(self, image, label):
        if isinstance(image, np.ndarray):
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if image.ndim == 2:
                image = np.expand_dims(image, -1)
        elif isinstance(image, Image.Image):
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
