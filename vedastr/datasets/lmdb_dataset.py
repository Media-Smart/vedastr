# modify from clovaai

import re
import six
import random

import lmdb
import numpy as np
from PIL import Image

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class LmdbDataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super(LmdbDataset, self).__init__(*args, **kwargs)

    def get_name_list(self):
        self.env = lmdb.open(self.root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
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
                return img, label

            if self.transforms:
                try:
                    img, label = self.transforms(img, label)
                except:
                    return self.__getitem__(index + 1)
            if not self.unknown:
                out_of_char = f'[^{self.character}]'
                label = re.sub(out_of_char, '', label)

        return img, label
