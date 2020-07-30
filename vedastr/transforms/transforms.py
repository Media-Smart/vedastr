import pdb
import random

import cv2
import torch
import numpy as np
import albumentations as albu
from albumentations import DualTransform
import albumentations.augmentations.functional as F

from .registry import TRANSFORMS

CV2_INTER = {
    'bilinear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC,
    'nearest': cv2.INTER_NEAREST
}

CV2_BORDER = {
    'constant': cv2.BORDER_CONSTANT,
    'reflect': cv2.BORDER_REFLECT,
    'replicate': cv2.BORDER_REPLICATE
}


@TRANSFORMS.register_module
class FactorScale(DualTransform):
    def __init__(self, scale=1.0, interpolation='bilinear',
                 always_apply=False, p=1.0):
        super(FactorScale, self).__init__(always_apply, p)
        self.scale = scale
        self.interpolation = CV2_INTER[interpolation]

    def apply(self, image, scale=1.0, **params):
        return F.scale(image, scale, interpolation=self.interpolation)

    def get_params(self):
        return {'scale': self.scale}


@TRANSFORMS.register_module
class LongestMaxSize(FactorScale):
    def __init__(self, h_max, w_max, interpolation='bilinear',
                 always_apply=False, p=1.0):
        self.h_max = h_max
        self.w_max = w_max
        super(LongestMaxSize, self).__init__(interpolation=interpolation,
                                             always_apply=always_apply,
                                             p=p)

    def update_params(self, params, **kwargs):
        params = super(LongestMaxSize, self).update_params(params, **kwargs)
        rows = params['rows']
        cols = params['cols']

        scale_h = self.h_max / rows
        scale_w = self.w_max / cols
        scale = min(scale_h, scale_w)

        params.update({'scale': scale})
        return params


@TRANSFORMS.register_module
class RandomScale(FactorScale):
    def __init__(self, scale_limit=(0.5, 2), interpolation='bilinear',
                 always_apply=False, p=1.0):
        super(RandomScale, self).__init__(interpolation=interpolation,
                                          always_apply=always_apply,
                                          p=p)
        self.scale_limit = albu.to_tuple(scale_limit)

    def get_params(self):
        return {
            'scale': random.uniform(self.scale_limit[0], self.scale_limit[1])}


@TRANSFORMS.register_module
class Resize(albu.Resize):
    def __init__(self, size, interpolation='bilinear', always_apply=False, p=1):
        super(Resize, self).__init__(size[0], size[1],
                                     CV2_INTER[interpolation], always_apply, p)


@TRANSFORMS.register_module
class PadIfNeeded(albu.PadIfNeeded):
    def __init__(self, min_height, min_width, border_mode='constant',
                 value=None):
        super(PadIfNeeded, self).__init__(min_height=min_height,
                                          min_width=min_width,
                                          border_mode=CV2_BORDER[border_mode],
                                          value=value, )

    def update_params(self, params, **kwargs):
        params = super(PadIfNeeded, self).update_params(params, **kwargs)
        rows = params['rows']
        cols = params['cols']

        if rows < self.min_height:
            h_pad_bottom = self.min_height - rows
        else:
            h_pad_bottom = 0

        if cols < self.min_width:
            w_pad_right = self.min_width - cols
        else:
            w_pad_right = 0

        params.update({'pad_top': 0,
                       'pad_bottom': h_pad_bottom,
                       'pad_left': 0,
                       'pad_right': w_pad_right})
        return params


@TRANSFORMS.register_module
class ToTensor(DualTransform):
    def __init__(self):
        super(ToTensor, self).__init__(always_apply=True)

    def apply(self, image, **params):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[:, :, None]
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)
        else:
            raise TypeError('img shoud be np.ndarray. Got {}'
                            .format(type(image)))
        return image


@TRANSFORMS.register_module
class Sensitive(DualTransform):

    def __init__(self, sensitive):
        self.sensitive = sensitive
        super(Sensitive, self).__init__(always_apply=True)

    def __call__(self, force_apply=False, **kwargs):
        if not self.sensitive:
            label = kwargs.get('label').lower()
            kwargs.update(label=label)

        return kwargs

    def get_transform_init_args_names(self):
        return ()


@TRANSFORMS.register_module
class ToGray(albu.ToGray):
    def __init__(self, channel=1, **kwargs):
        self.channel = channel
        super(ToGray, self).__init__(**kwargs)

    def __call__(self, force_apply=False, **kwargs):
        new_img = self.apply(kwargs['image'])
        # new_res = super.__call__(**kwargs)
        if self.channel == 1:
            kwargs['image'] = new_img[:, :, :1]

        return kwargs


@TRANSFORMS.register_module
class Rotate(albu.Rotate):

    def __init__(self, limit=90,
                 interpolation='bilinear',
                 border_mode='constant',
                 value=0,
                 always_apply=False,
                 p=0.5, ):
        super(Rotate, self).__init__(limit, CV2_INTER[interpolation], CV2_BORDER[border_mode],
                                     value=value, always_apply=always_apply, p=p)


@TRANSFORMS.register_module
class ExpandRotate(Rotate):

    def __init__(self, **kwargs):
        super(ExpandRotate, self).__init__(**kwargs)

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return self.rotate(img, angle, interpolation, self.border_mode, self.value)

    def rotate(self, img, angle, interpolation, border_mode, value):
        pi_angle = np.deg2rad(angle)
        height, width = img.shape[:2]
        new_h = int(height * np.fabs(np.cos(pi_angle)) + width * np.fabs(np.sin(pi_angle)))
        new_w = int(height * np.fabs(np.sin(pi_angle)) + width * np.fabs(np.cos(pi_angle)))

        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        matrix[0, 2] += (new_w - width) / 2
        matrix[1, 2] += (new_h - height) / 2
        img = cv2.warpAffine(img, matrix, (new_w, new_h), interpolation, border_mode, value)

        return img
