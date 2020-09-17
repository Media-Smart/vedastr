# modify from clovaai

import random
import re

import lmdb
import six
import cv2
import numpy as np
from PIL import Image

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class LmdbDataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        self.index_list = []
        super(LmdbDataset, self).__init__(*args, **kwargs)

    def get_name_list(self):
        self.env = lmdb.open(self.root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            n_samples = int(txn.get('num-samples'.encode()))
            for index in range(n_samples):
                idx = index + 1  # lmdb starts with 1
                label_key = 'label-%09d'.encode() % idx
                label = txn.get(label_key).decode('utf-8')
                if self.filter(label):  # if length of label larger than max_len, drop this sample
                    continue
                else:
                    self.index_list.append(idx)
            self.samples = len(self.index_list)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.index_list[index]

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
                # img = cv2.imdecode(np.fromstring(value, np.uint8), 3)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except IOError:
                print(f'Corrupted image for {index}')
                img, label = self.__getitem__(random.choice(range(len(self))))
                return img, label

            if self.transforms:
                try:
                    aug = self.transforms(image=img, label=label)
                    img, label = aug['image'], aug['label']
                except:
                    return self.__getitem__(random.choice(range(len(self))))

        return img, label
