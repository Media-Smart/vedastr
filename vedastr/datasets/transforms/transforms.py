import random
import numbers

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter, ImageDraw, ImageOps

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
        image = image.resize(target_size, resample=PIL_MODE[self.mode])

        return image, label


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
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image, label):
        w, h = image.size
        nh, nw = self.size
        assert h <= nh and w <= nw

        image = ImageOps.expand(image, (0, 0, nw-w, nh-h), self.fill)

        return image, label


@TRANSFORMS.register_module
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image, label


@TRANSFORMS.register_module
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        return image, label


@TRANSFORMS.register_module
class RandomRotate90(object):
    def __init__(self, degrees=(90,180,270), p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            degree = random.choice(self.degrees)
            if degree == 90:
                rotate = Image.ROTATE_90
            elif degree == 180:
                rotate = Image.ROTATE_180
            elif degree == 270:
                rotate = Image.ROTATE_270
            else:
                raise ValueError(f'degree {degree} not valid, valid degrees are 90, 180, 270].')
            image = image.transpose(rotate)

        return image, label


@TRANSFORMS.register_module
class RandomPerspective(object):
    def __init__(self, distortion_scale=0.5, p=0.5, mode='cubic'):
        self.transform = torchvision.transforms.RandomPerspective(distortion_scale=distortion_scale, p=p, interpolation=PIL_MODE[mode])

    def __call__(self, image, label):
        image = self.transform(image)

        return image, label


@TRANSFORMS.register_module
class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        self.p = p
        self.transform = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, label):
        if random.random() < self.p:
            image = self.transform(image)

        return image, label


@TRANSFORMS.register_module
class AspectRatioJitter(object):
    def __init__(self, jitter=0, p=0.5, mode='cubic'):
        self.jitter = jitter
        self.p = p
        self.mode = mode

    @property
    def scale_factor(self):
        a = random.uniform(1 - self.jitter, 1 + self.jitter)
        b = random.uniform(1 - self.jitter, 1 + self.jitter)

        return a / b

    def __call__(self, image, label):
        if random.random() < self.p:
            w, h = image.size
            area = w * h
            n_ar = w / h * self.scale_factor
            nw = int((area * n_ar) ** 0.5)
            nh = int((area / n_ar) ** 0.5)
            image = image.resize((nw, nh), resample=PIL_MODE[self.mode])

        return image, label


@TRANSFORMS.register_module
class MotionBlur(object):
    def __init__(self, blur_limit=5, p=0.5):
        self.blur_limit = blur_limit
        self.p = p

        assert blur_limit < 6

    def __call__(self, image, label):
        if random.random() < self.p:
            ksize = int(random.choice(np.arange(3, self.blur_limit+1, 2)))
            kernel = Image.new('L', (ksize, ksize), 0)
            draw_kernel = ImageDraw.Draw(kernel)
            xs, xe = random.randint(0, ksize-1), random.randint(0, ksize-1)
            if xs == xe:
                ys, ye = random.sample(range(ksize), 2)
            else:
                ys, ye = random.randint(0, ksize-1), random.randint(0, ksize-1)
            draw_kernel.line((xs, ys, xe, ye), 1)
            kernel = np.asarray(kernel).flatten().tolist()
            kernel = ImageFilter.Kernel((ksize, ksize), kernel)
            image = image.filter(kernel)

        return image, label


@TRANSFORMS.register_module
class GaussianNoise(object):
    def __init__(self, var_limit=(10.0, 50.0), mean=0, p=0.5):
        self.var_limit = var_limit
        self.mean = mean
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            var = random.uniform(self.var_limit[0], self.var_limit[1])
            sigma = var ** 0.5
            random_state = np.random.RandomState(random.randint(0, 2**32 - 1))

            image = np.asarray(image, dtype=np.float)
            gauss_image = random_state.normal(self.mean, sigma, image.shape)
            image = image + gauss_image
            image = Image.fromarray(np.uint8(image))

        return image, label


@TRANSFORMS.register_module
class BaseRotation(object):
    def __init__(self, expand=False, center=None, fill=0, mode='cubic', p=0.5):
        self.resample = PIL_MODE[mode]
        self.expand = expand
        self.center = center
        self.fill = fill
        self.p = p

    def get_angle(self):
        raise NotImplemented

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.get_angle()
            image = image.rotate(angle, self.resample, self.expand, self.center, fillcolor=self.fill)

        return image, label


@TRANSFORMS.register_module
class RandomUniformRotation(BaseRotation):
    def __init__(self, degrees, **kwargs):
        super(RandomUniformRotation, self).__init__(**kwargs)

        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    def get_angle(self):
        return random.uniform(self.degrees)


@TRANSFORMS.register_module
class RandomNormalRotation(BaseRotation):
    def __init__(self, mean=0, std=34, **kwargs):
        super(RandomNormalRotation, self).__init__(**kwargs)

        self.mean = mean
        self.std = std

    def get_angle(self):
        return random.gauss(self.mean, self.std)


@TRANSFORMS.register_module
class RandomScale(object):
    def __init__(self, scales, step=0.0, mode='cubic', p=0.5):
        self.scales = scales
        self.step = step
        self.mode = mode
        self.p = p

    def scale_factor(self):
        if isinstance(self.scales, (int, float)):
            return self.scales

        assert len(self.scales) == 2
        min_scale, max_scale = self.scales

        if self.step == 0:
            return random.uniform(min_scale, max_scale)

        num_steps = int((max_scale - min_scale) / self.step + 1)
        scale_factors = np.linspace(min_scale, max_scale, num_steps).tolist()
        scale_factor = random.choice(scale_factors)

        return scale_factor

    def __call__(self, image, label):
        if random.random() < self.p:
            w, h = image.size
            scale_factor = self.scale_factor()
            nw, nh = int(w * scale_factor), int(h * scale_factor)
            image = image.resize((nw, nh), resample=PIL_MODE[self.mode])

        return image, label


@TRANSFORMS.register_module
class KeepHorizontal(object):
    def __init__(self, clockwise=False):
        self.clockwise = clockwise

    def __call__(self, image, label):
        w, h = image.size
        if h > w:
            if self.clockwise:
                image = image.transpose(Image.ROTATE_270)
            else:
                image = image.transpose(Image.ROTATE_90)

        return image, label
