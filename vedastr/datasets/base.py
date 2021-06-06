# modify from clovaai

import logging
import os
import re

import cv2
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
        Args:
            root (str): The dir path of image files.
            transform: Transformation for images, which will be passed
                     automatically if you set transform cfg correctly in
                     configure file.
            character (str): The character will be used. We will filter the
                            sample based on the charatcer.
            batch_max_length (int): The max allowed length of the text
                                   after filter.
            data_filter (bool): If true, we will filter sample based on the
                               character. Otherwise not filter.

    """

    def __init__(self,
                 root: str,
                 transform=None,
                 character: str = 'abcdefghijklmnopqrstuvwxyz0123456789',
                 batch_max_length: int = 100000,
                 data_filter: bool = True):

        assert type(
            root
        ) == str, f'The type of root should be str but got {type(root)}'

        self.root = os.path.abspath(root)
        self.character = character
        self.batch_max_length = batch_max_length
        self.data_filter = data_filter

        if transform is not None:
            self.transforms = transform
        self.samples = 0
        self.img_names = []
        self.gt_texts = []
        self.get_name_list()

        self.logger = logging.getLogger()
        self.logger.info(
            f'current dataset length is {self.samples} in {self.root}')

    def get_name_list(self):
        raise NotImplementedError

    def filter(self, label, retrun_len=False):
        if not self.data_filter:
            if not retrun_len:
                return False
            return False, len(label)
        """We will filter those samples whose length is larger
         than defined max_length by default."""
        character = "".join(sorted(self.character, key=lambda x: ord(x)))
        out_of_char = f'[^{character}]'
        # replace those character not in self.character with ''
        label = re.sub(out_of_char, '', label.lower())
        # filter whose label larger than batch_max_length
        if len(label) > self.batch_max_length:
            if not retrun_len:
                return True
            return True, len(label)
        if not retrun_len:
            return False
        return False, len(label)

    def __getitem__(self, index):
        # default img channel is rgb
        img = cv2.imread(self.img_names[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.gt_texts[index]

        if self.transforms:
            aug = self.transforms(image=img, label=label)
            img, label = aug['image'], aug['label']

        return img, label

    def __len__(self):
        return self.samples
