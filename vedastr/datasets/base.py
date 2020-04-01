# modify from clovaai

import os
import re
import logging

from torch.utils.data import Dataset
from PIL import Image


class BaseDataset(Dataset):
    def __init__(self, root, gt_txt=None, transform=None, character='abcdefghijklmnopqrstuvwxyz0123456789',
                 batch_max_length=25, data_filter_off=False, unknown=False):
        assert type(root) == str
        if gt_txt is not None:
            assert os.path.isfile(gt_txt)
            self.gt_txt = gt_txt
        self.root = root
        self.character = character
        self.batch_max_length = batch_max_length
        self.data_filter_off = data_filter_off
        self.unknown = unknown

        if transform:
            self.transforms = transform
        self.samples = 0
        self.img_names = []
        self.gt_texts = []
        self.get_name_list()

        self.logger = logging.getLogger()
        self.logger.info(f'current dataset length is {self.samples} in {self.root}')

    def get_name_list(self):
        raise NotImplementedError

    def filter(self, label):
        if self.data_filter_off:
            return False
        else:
            if len(label) > self.batch_max_length:
                return True
            out_of_char = f'[^{self.character}]'
            if re.search(out_of_char, label.lower()) and not self.unknown:
                return True
            return False

    def __getitem__(self, index):
        img = Image.open(self.img_names[index])
        label = self.gt_texts[index]

        if self.transforms:
            img, label = self.transforms(img, label)
        if not self.unknown:
            out_of_char = f'[^{self.character}]'
            label = re.sub(out_of_char, '', label)
        return img, label

    def __len__(self):
        return self.samples
