import os
import re
import six
import random
import string

import cv2
import lmdb
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .registery import DATASETS

extension_names = ['.jpg', '.png', '.bmp', '.jpeg']


class BaseDataset(Dataset):
    def __init__(self, root, gt_txt=None, transforms=None, character='abcdefghijklmnopqrstuvwxyz0123456789',
                 sensitive=False, batch_max_length=25, data_filter_off=False):
        assert type(root) == str
        if gt_txt is not None:
            assert os.path.isfile(gt_txt)
            self.gt_txt = gt_txt
        self.root = root
        self.character = character
        self.batch_max_length = batch_max_length
        self.data_filter_off = data_filter_off
        if sensitive:
            self.character = string.printable[:-6]

        if transforms:
            self.transforms = transforms
        self.samples = 0
        self.img_names = []
        self.gt_texts = []
        self.get_name_list()

    def get_name_list(self):
        raise NotImplementedError

    def filter(self, label):
        if self.data_filter_off:
            return False
        else:
            if len(label) > self.batch_max_length:
                return True
            out_of_char = f'[^{self.character}]'
            if re.search(out_of_char, label.lower()):
                return True
            return False

    def __getitem__(self, index):
        img = cv2.imread(self.img_names[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.gt_texts[index]

        if self.transforms:
            img, label = self.transforms(img, label)
        return (img, label)

    def __len__(self):
        return self.samples


@DATASETS.register_module
class BaseDatasetTxt(BaseDataset):

    def __init__(self, img_path, gt_txt, transforms=None, character='abcdefghijklmnopqrstuvwxyz0123456789',
                 sensitive=False, batch_max_length=25, data_filter_off=False):
        super(BaseDatasetTxt, self).__init__(img_path, gt_txt, transforms)

    def get_name_list(self):
        with open(self.gt_txt, 'r') as gt:
            for line in gt.readlines():
                img_name, label = line.strip().split('\t')
                if self.filter(label):
                    continue
                else:
                    self.img_names.append(os.path.join(self.img_path, img_name))
                    self.gt_texts.append(label)

        self.samples = len(self.gt_texts)


@DATASETS.register_module
class BaseDatasetFolder(BaseDataset):

    def __init__(self, root, transforms=None, character='abcdefghijklmnopqrstuvwxyz0123456789', sensitive=False,
                 batch_max_length=25, data_filter_off=False):
        super(BaseDatasetFolder, self).__init__(root, transforms=transforms, character=character,
                                                sensitive=sensitive, batch_max_length=batch_max_length,
                                                data_filter_off=data_filter_off)

    @staticmethod
    def parse_filename(text):
        return text.split('_')[-1]

    def get_name_list(self):
        for item in os.listdir(self.img_path):
            file_name, file_extension = os.path.splitext(item)
            if file_extension in extension_names:
                label = self.parse_filename(file_name)
                if self.filter(label):
                    continue
                else:
                    self.img_names.append(os.path.join(self.img_path, item))
                    self.gt_texts.append(label)
        self.samples = len(self.gt_texts)


@DATASETS.register_module
class LmdbDataset(BaseDataset):

    def __init__(self, root, transforms=None, character='abcdefghijklmnopqrstuvwxyz0123456789', sensitive=False,
                 batch_max_length=25, data_filter_off=False):
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        super(LmdbDataset, self).__init__(root=root, transforms=transforms, character=character, sensitive=sensitive,
                                          batch_max_length=batch_max_length, data_filter_off=data_filter_off)

    def get_name_list(self):
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.samples = nSamples

            if self.data_filter_off:
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                self.filtered_index_list = []
                for index in range(self.nSamples):
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

            img = np.array(img)
            if self.transforms:
                img, label = self.transforms(img, label)
            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


if __name__ == '__main__':
    prefix = r'D:\Project\STR\data\MJ'
    # dataset1 = BaseDatasetTxt(os.path.join(prefix, 'timage'), os.path.join(prefix, 'CUTE.txt'))
    dataset2 = BaseDatasetFolder(os.path.join(prefix))
    _data_loader = torch.utils.data.DataLoader(
        dataset2, batch_size=32,
        shuffle=True,
        num_workers=4
    )
    print('done')
