import random
import re

import albumentations as albu
import albumentations.augmentations.functional as F
import cv2
import numpy as np
import torch
from albumentations import DualTransform

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

    def __init__(self,
                 scale=1.0,
                 interpolation='bilinear',
                 always_apply=False,
                 p=1.0):
        super(FactorScale, self).__init__(always_apply, p)
        self.scale = scale
        self.interpolation = CV2_INTER[interpolation]

    def _scale(self, img, scale, interpolation):
        height, width = img.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        if new_width == 0 or new_width == 0:
            return img
        return cv2.resize(
            img, (new_width, new_height), interpolation=interpolation)

    def apply(self, image, scale=1.0, **params):
        return self._scale(image, scale, interpolation=self.interpolation)

    def get_params(self):
        return {'scale': self.scale}


@TRANSFORMS.register_module
class LongestMaxSize(FactorScale):

    def __init__(self,
                 h_max,
                 w_max,
                 interpolation='bilinear',
                 always_apply=False,
                 p=1.0):
        self.h_max = h_max
        self.w_max = w_max
        super(LongestMaxSize, self).__init__(
            interpolation=interpolation, always_apply=always_apply, p=p)

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

    def __init__(self,
                 scale_limit=(0.5, 2),
                 interpolation='bilinear',
                 always_apply=False,
                 p=1.0):
        super(RandomScale, self).__init__(
            interpolation=interpolation, always_apply=always_apply, p=p)
        self.scale_limit = albu.to_tuple(scale_limit)

    def get_params(self):
        return {
            'scale': random.uniform(self.scale_limit[0], self.scale_limit[1])
        }


@TRANSFORMS.register_module
class Resize(albu.Resize):

    def __init__(self,
                 size,
                 interpolation='bilinear',
                 always_apply=False,
                 p=1):
        super(Resize, self).__init__(size[0], size[1],
                                     CV2_INTER[interpolation], always_apply, p)


@TRANSFORMS.register_module
class PadIfNeeded(albu.PadIfNeeded):

    def __init__(
            self,
            min_height,
            min_width,
            border_mode='constant',
            value=None,
            position='topleft',
            adaptive=False,
            pad_height_divisor=None,
            pad_width_divisor=None,
    ):
        super(PadIfNeeded, self).__init__(
            min_height=min_height,
            min_width=min_width,
            border_mode=CV2_BORDER[border_mode],
            value=value,
            pad_height_divisor=pad_height_divisor,
            pad_width_divisor=pad_width_divisor,
        )
        assert position in ['topleft', 'center', 'random']
        self.position = position
        self.adp = adaptive

    def apply(self,
              img,
              pad_top=0,
              pad_bottom=0,
              pad_left=0,
              pad_right=0,
              **params):
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.value)

    def get_params_dependent_on_targets(self, params):
        if not self.adp or not self.border_mode == cv2.BORDER_CONSTANT:
            return params
        img = params['image']
        flatten = img.reshape(-1, )
        pad_v = flatten[flatten > (flatten.mean() - 10)].mean()
        self.value = int(pad_v)
        return params

    @property
    def targets_as_params(self):
        return ['image']

    def update_params(self, params, **kwargs):

        params = super(PadIfNeeded, self).update_params(params, **kwargs)
        rows = params['rows']
        cols = params['cols']
        if self.min_height is not None:
            if rows < self.min_height:
                h_pad_bottom = self.min_height - rows
            else:
                h_pad_bottom = 0
        else:
            pad_remained = rows % self.pad_height_divisor
            h_pad_bottom = self.pad_height_divisor - pad_remained if pad_remained > 0 else 0  # noqa 501
        if self.min_width is not None:
            if cols < self.min_width:
                w_pad_right = self.min_width - cols
            else:
                w_pad_right = 0
        else:
            pad_remainder = cols % self.pad_width_divisor
            w_pad_right = self.pad_width_divisor - pad_remainder if pad_remainder > 0 else 0  # noqa 501

        if self.position == 'center':
            half_w_pad_right = w_pad_right // 2
            half_w_pad_left = w_pad_right // 2
            half_h_pad_bottom = h_pad_bottom // 2
            half_h_pad_top = h_pad_bottom // 2
        elif self.position == 'random':
            half_w_pad_right = np.random.choice(np.arange(w_pad_right))
            half_w_pad_left = w_pad_right - half_w_pad_right
            half_h_pad_bottom = np.random.choice(np.arange(h_pad_bottom))
            half_h_pad_top = h_pad_bottom - half_h_pad_bottom
        else:
            half_w_pad_right = w_pad_right
            half_w_pad_left = 0
            half_h_pad_bottom = h_pad_bottom
            half_h_pad_top = 0

        params.update({
            'pad_top': half_h_pad_top,
            'pad_bottom': half_h_pad_bottom,
            'pad_left': half_w_pad_left,
            'pad_right': half_w_pad_right
        })
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
            raise TypeError('img shoud be np.ndarray. Got {}'.format(
                type(image)))
        return image


@TRANSFORMS.register_module
class Sensitive(DualTransform):
    """
    Args:
        sensitive (bool): If false, all upper-case will transfer
                         to lower-case, else do nothing.
    """

    def __init__(self, sensitive):
        self.sensitive = sensitive
        super(Sensitive, self).__init__(always_apply=True)

    def __call__(self, force_apply=False, **kwargs):
        label = kwargs.get('label')
        if not self.sensitive:
            label = label.lower()
        kwargs.update(label=label)

        return kwargs

    def get_transform_init_args_names(self):
        return ()


@TRANSFORMS.register_module
class Filter(DualTransform):
    """
    Args:
        need_character (str) :  For each character in label, replace
                               it with '' if it isn't in need_character.
        p (float) : Probability of filtering.
    """

    def __init__(self, need_character: str, p=1.0):
        super(Filter, self).__init__(p=p)
        self.chars = "".join(sorted(need_character, key=lambda x: ord(x)))

    def __call__(self, force_apply=False, **kwargs):
        label = kwargs.get('label')
        out_of_char = f'[^{self.chars}]'
        label = re.sub(out_of_char, '', label)
        kwargs.update(label=label)

        return kwargs


@TRANSFORMS.register_module
class ToGray(albu.ToGray):

    def __init__(self, channel=1, **kwargs):
        self.channel = channel
        super(ToGray, self).__init__(**kwargs)

    def __call__(self, force_apply=False, **kwargs):
        new_img = self.apply(kwargs['image'])
        if self.channel == 1:
            kwargs['image'] = new_img[:, :, :1]

        return kwargs


@TRANSFORMS.register_module
class Rotate(albu.Rotate):

    def __init__(
            self,
            limit=90,
            interpolation='bilinear',
            border_mode='constant',
            value=0,
            always_apply=False,
            p=0.5,
    ):
        super(Rotate, self).__init__(
            limit,
            CV2_INTER[interpolation],
            CV2_BORDER[border_mode],
            value=value,
            always_apply=always_apply,
            p=p)


@TRANSFORMS.register_module
class AdaptiveRandomCrop(albu.DualTransform):

    def __init__(self,
                 ratio: tuple = None,
                 length: tuple = None,
                 always_apply=False,
                 p=1):
        """
        RandomCrop based on the current image shape.

        Args:
            ratio: (crop_height_ratio, crop_width_ratio), random sample from it. # noqa 501
            length: (crop_height_length, crop_width_length), random sample from it. # noqa 501
        """
        super(AdaptiveRandomCrop, self).__init__(
            always_apply=always_apply, p=p)
        assert (ratio is not None) ^ (
                length
                is not None), 'Only one of args ratio and length can be set.'
        self.ratio = ratio
        self.length = length

    def apply(self, img, **params):

        y1, x1, y2, x2 = params['start_h'], params['start_w'], params[
            'end_h'], params['end_w']
        return F.crop(img, x1, y1, x2, y2)

    @property
    def targets_as_params(self):
        return ['image']

    def get_params_dependent_on_targets(self, params):
        height, width = params['image'].shape[:2]
        if self.ratio is not None:
            shifit_h = int(random.random() * self.ratio[0] * height)
            shifit_w = int(random.random() * self.ratio[1] * width)
        else:
            shifit_h = int(random.random() * self.length[0])
            shifit_w = int(random.random() * self.length[1])
        start_h = int(random.random() * shifit_h)
        start_w = int(random.random() * shifit_w)
        end_h = height - start_h
        end_w = width - start_w
        return {
            'start_h': start_h,
            'start_w': start_w,
            'end_h': end_h,
            'end_w': end_w
        }


@TRANSFORMS.register_module
class ExpandRotate(Rotate):
    """Rotate the image with no cutting.
    """

    def __init__(self, **kwargs):
        super(ExpandRotate, self).__init__(**kwargs)

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return self.rotate(img, angle, self.interpolation, self.border_mode,
                           self.value)

    def rotate(self, img, angle, interpolation, border_mode, value):
        pi_angle = np.deg2rad(angle)
        height, width = img.shape[:2]
        new_h = int(height * np.fabs(np.cos(pi_angle)) +
                    width * np.fabs(np.sin(pi_angle)))
        new_w = int(height * np.fabs(np.sin(pi_angle)) +
                    width * np.fabs(np.cos(pi_angle)))

        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        matrix[0, 2] += (new_w - width) / 2
        matrix[1, 2] += (new_h - height) / 2
        img = cv2.warpAffine(
            img,
            matrix, (new_w, new_h),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=value)

        return img


@TRANSFORMS.register_module
class TIA(DualTransform):
    """Text image augmentation.
    Please refer to https://github.com/Canjie-Luo/Text-Image-Augmentation and
    https://github.com/RubanSeven/Text-Image-Augmentation-python

    modes (list): Three mode are supported, e.g., perspective, stretch and distort. # noqa 501
    segment (int): The number of segments.
    thresh (list):


    """

    def __init__(self,
                 modes: (list, tuple),
                 segment: int,
                 thresh: (list, tuple) = None,
                 **kwargs):
        super(TIA, self).__init__(**kwargs)
        if not isinstance(modes, (list, tuple)):
            modes = [modes]
        for mode in modes:
            assert mode in ['perspective', 'stretch', 'distort'], \
                'Support modes are [perspective, stretch, distort]' \
                f' but got {modes}'
        self.modes = modes
        self.segment = segment
        if thresh is not None:
            if not isinstance(thresh, list):
                thresh = [thresh]
            assert len(thresh) == len(modes)
        self.thresh = {md: th for md, th in zip(modes, thresh)}

    def apply(self, img, **params):
        mode = random.choice(self.modes)
        if mode == 'perspective':
            src_pts, dst_pts, img_w, img_h = self._perspective(
                img, self.thresh[mode])
        elif mode == 'stretch':
            src_pts, dst_pts, img_w, img_h = self._stretch(
                img, self.thresh[mode])
        else:
            src_pts, dst_pts, img_w, img_h = self._distort(
                img, self.thresh[mode])
        pt_count = len(dst_pts)
        # TODO
        trans_ratio = 1.
        grid_size = 100
        rdx = np.zeros((img_h, img_w))
        rdy = np.zeros((img_h, img_w))
        rdx, rdy = self.calc_delta(pt_count, src_pts, dst_pts, img_w, img_h,
                                   grid_size, rdx, rdy)
        img = self.gen_img(img, img_h, img_w, grid_size, rdx, rdy, trans_ratio)
        return img

    def calc_delta(self, pt_count, src_pts, dst_pts, dst_w, dst_h, grid_size,
                   rdx, rdy):
        w = np.zeros(pt_count, dtype=np.float32)

        if pt_count < 2:
            return

        i = 0
        while 1:
            if dst_w <= i < dst_w + grid_size - 1:
                i = dst_w - 1
            elif i >= dst_w:
                break
            j = 0
            while 1:
                if dst_h <= j < dst_h + grid_size - 1:
                    j = dst_h - 1
                elif j >= dst_h:
                    break

                sw = 0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(pt_count):
                    if i == dst_pts[k][0] and j == dst_pts[k][1]:
                        break

                    w[k] = 1. / ((i - dst_pts[k][0]) * (i - dst_pts[k][0]) +
                                 (j - dst_pts[k][1]) * (j - dst_pts[k][1]))

                    sw += w[k]
                    swp = swp + w[k] * np.array(dst_pts[k])
                    swq = swq + w[k] * np.array(src_pts[k])

                if k == pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    miu_s = 0
                    for k in range(pt_count):
                        if i == dst_pts[k][0] and j == dst_pts[k][1]:
                            continue
                        pt_i = dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)

                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(pt_count):
                        if i == dst_pts[k][0] and j == dst_pts[k][1]:
                            continue

                        pt_i = dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])

                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] = np.sum(pt_i * cur_pt) * src_pts[k][0] - \
                                    np.sum(pt_j * cur_pt) * src_pts[k][1]
                        tmp_pt[1] = -np.sum(pt_i * cur_pt_j) * src_pts[k][0] + \
                                    np.sum(pt_j * cur_pt_j) * src_pts[k][1]
                        tmp_pt *= (w[k] / miu_s)
                        new_pt += tmp_pt

                    new_pt += qstar
                else:
                    new_pt = src_pts[k]

                rdx[j, i] = new_pt[0] - i
                rdy[j, i] = new_pt[1] - j

                j += grid_size
            i += grid_size
        return rdx, rdy

    def gen_img(self, img, dst_h, dst_w, grid_size, rdx, rdy, trans_ratio):
        src_h, src_w = img.shape[:2]
        dst = np.zeros_like(img, dtype=np.float32)

        for i in np.arange(0, dst_h, grid_size):
            for j in np.arange(0, dst_w, grid_size):
                ni = i + grid_size
                nj = j + grid_size
                w = h = grid_size
                if ni >= dst_h:
                    ni = dst_h - 1
                    h = ni - i + 1
                if nj >= dst_w:
                    nj = dst_w - 1
                    w = nj - j + 1

                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self._bilinear_interp(di / h, dj / w, rdx[i, j],
                                                rdx[i, nj], rdx[ni, j],
                                                rdx[ni, nj])
                delta_y = self._bilinear_interp(di / h, dj / w, rdy[i, j],
                                                rdy[i, nj], rdy[ni, j],
                                                rdy[ni, nj])
                nx = j + dj + delta_x * trans_ratio
                ny = i + di + delta_y * trans_ratio
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)

                if len(img.shape) == 3:
                    c = img.shape[-1]
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, c))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, c))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i:i + h,
                j:j + w] = self._bilinear_interp(x,
                                                 y,
                                                 img[nyi, nxi],
                                                 img[nyi, nxi1],
                                                 img[nyi1, nxi],
                                                 img[nyi1, nxi1])
        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.uint8)

        return dst

    @staticmethod
    def _bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + \
               (v21 * (1 - y) + v22 * y) * x

    @staticmethod
    def _src_points(img):
        img_h, img_w = img.shape[:2]
        src_pts = list()
        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])
        return src_pts, img_h, img_w

    def _perspective(self, img, thresh):
        src_pts, img_h, img_w = self._src_points(img)
        if thresh is None:
            thresh = img_h // 2
        else:
            thresh = thresh[0]
        dst_pts = list()
        dst_pts.append([0, np.random.randint(thresh)])
        dst_pts.append([img_w, np.random.randint(thresh)])
        dst_pts.append([img_w, img_h - np.random.randint(thresh)])
        dst_pts.append([0, img_h - np.random.randint(thresh)])
        return src_pts, dst_pts, img_w, img_h

    def _stretch(self, img, thresh):
        src_pts, img_h, img_w = self._src_points(img)
        cut = img_w // self.segment
        if thresh is None:
            thresh = cut * 4 // 5
        else:
            thresh = thresh[0]
        dst_pts = list()
        dst_pts.append([0, 0])
        dst_pts.append([img_w, 0])
        dst_pts.append([img_w, img_h])
        dst_pts.append([0, img_h])
        half_thresh = thresh * 0.5
        for cut_idx in np.arange(1, self.segment, 1):
            move = np.random.randint(thresh) - half_thresh
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + move, 0])
            dst_pts.append([cut * cut_idx + move, img_h])
        return src_pts, dst_pts, img_w, img_h

    def _distort(self, img, thresh):
        src_pts, img_h, img_w = self._src_points(img)
        cut = img_w // self.segment
        if thresh is None:
            thresh = cut // 3
        else:
            thresh = thresh[0]
        dst_pts = list()

        dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
        dst_pts.append(
            [img_w - np.random.randint(thresh),
             np.random.randint(thresh)])
        dst_pts.append([
            img_w - np.random.randint(thresh),
            img_h - np.random.randint(thresh)
        ])
        dst_pts.append(
            [np.random.randint(thresh), img_h - np.random.randint(thresh)])
        half_thresh = thresh * 0.5
        for cut_idx in np.arange(1, self.segment, 1):
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([
                cut * cut_idx + np.random.randint(thresh) - half_thresh,
                np.random.randint(thresh) - half_thresh
            ])
            dst_pts.append([
                cut * cut_idx + np.random.randint(thresh) - half_thresh,
                img_h + np.random.randint(thresh) - half_thresh
            ])
        return src_pts, dst_pts, img_w, img_h
