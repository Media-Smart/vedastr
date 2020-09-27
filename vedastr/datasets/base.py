# modify from clovaai

import logging
import os
import re

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, root, gt_txt=None, transform=None, character='abcdefghijklmnopqrstuvwxyz0123456789',
                 batch_max_length=100000, data_filter=True):
        assert type(root) == str
        if gt_txt is not None:
            assert os.path.isfile(gt_txt)
            self.gt_txt = gt_txt
        self.root = os.path.abspath(root)
        self.character = character
        self.batch_max_length = batch_max_length
        self.data_filter = data_filter

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
        if not self.data_filter:
            return False
        """We will filter those samples whose length is larger than defined max_length by default."""
        out_of_char = f'[^{self.character}]'
        label = re.sub(out_of_char, '', label.lower())  # replace those character not in self.character with ''
        if len(label) > self.batch_max_length:  # filter whose label larger than batch_max_length
            return True

        return False

    def __getitem__(self, index):
        img = Image.open(self.img_names[index])
        label = self.gt_texts[index]

        if self.transforms:
            aug = self.transforms(image=img, label=label)
            img, label = aug['image'], aug['label']

        return img, label

    def __len__(self):
        return self.samples
