# modify from clovaai

import re
import sys
import six
import lmdb
import string

import numpy as np
from PIL import Image

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class LmdbDataset(BaseDataset):

    def __init__(self, root, transform=None, character='abcdefghijklmnopqrstuvwxyz0123456789',
                 batch_max_length=25, data_filter_off=False, cv_mode=False):
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        super(LmdbDataset, self).__init__(root=root, transform=transform, character=character,
                                          batch_max_length=batch_max_length, data_filter_off=data_filter_off,
                                          cv_mode=cv_mode)

    def get_name_list(self):
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))

            if self.data_filter_off:
                self.filtered_index_list = [index + 1 for index in range(nSamples)]
                self.samples = nSamples
            else:
                self.filtered_index_list = []
                for index in range(nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if self.filter(label):
                        continue
                    else:
                        self.filtered_index_list.append(index)

                self.samples = len(self.filtered_index_list)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')  # for color image

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                img, label = self.__getitem__(random.choice(range(len(self))))
                return (img, label)
            if self.cv_mode:
                img = np.array(img)
            if self.transforms:
                img, label = self.transforms(img, label)
            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)
