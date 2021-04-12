# modify from clovaai

import lmdb
import logging
import numpy as np
import six
from PIL import Image

from .base import BaseDataset
from .registry import DATASETS

logger = logging.getLogger()


@DATASETS.register_module
class LmdbDataset(BaseDataset):
    """ Read the data of lmdb format.
    Please refer to https://github.com/Media-Smart/vedastr/issues/27#issuecomment-691793593  # noqa 501
    if you have problems with creating lmdb format file.

    """

    def __init__(self,
                 root: str,
                 transform=None,
                 character: str = 'abcdefghijklmnopqrstuvwxyz0123456789',
                 batch_max_length: int = 100000,
                 data_filter: bool = True):
        self.index_list = []
        super(LmdbDataset, self).__init__(
            root=root,
            transform=transform,
            character=character,
            batch_max_length=batch_max_length,
            data_filter=data_filter)

    def get_name_list(self):
        self.env = lmdb.open(
            self.root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        with self.env.begin(write=False) as txn:
            n_samples = int(txn.get('num-samples'.encode()))
            for index in range(n_samples):
                idx = index + 1  # lmdb starts with 1
                label_key = 'label-%09d'.encode() % idx
                label = txn.get(label_key).decode('utf-8')
                if self.filter(
                        label
                ):  # if length of label larger than max_len, drop this sample
                    continue
                else:
                    self.index_list.append(idx)
            self.samples = len(self.index_list)

    def read_data(self, index):
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
            img = Image.open(buf).convert('RGB')  # for color image
            img = np.array(img)

            return img, label

    def __getitem__(self, index):

        img, label = self.read_data(index)
        if self.transforms:
            aug = self.transforms(image=img, label=label)
            img, label = aug['image'], aug['label']

        return img, label
