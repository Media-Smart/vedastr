import cv2
import lmdb
import random

from .lmdb_dataset import LmdbDataset
from .registry import DATASETS


@DATASETS.register_module
class PasteDataset(LmdbDataset):
    """ Concat two images and the combined label should satisfy the constrains.

    Args:
            p: The probability of pasting operation.

    Warnings:: We will create a new transform operation to
              replace this dataset.
    """

    def __init__(self, p: float = 0.1, *args, **kwargs):
        self.len_sample = dict()
        self.len_lists = list()
        self.p = p
        super(PasteDataset, self).__init__(*args, **kwargs)

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
                flag, length = self.filter(label, retrun_len=True)
                if flag:
                    continue
                else:
                    self.index_list.append(idx)
                    self.len_lists.append(length)
                    if length not in self.len_sample:
                        self.len_sample[length] = [len(self.index_list) - 1]
                    else:
                        self.len_sample[length].append(
                            len(self.index_list) - 1)

            self.samples = len(self.index_list)

    def __getitem__(self, index):
        img, label = self.read_data(index)
        th, tw = img.shape[:2]
        c_len = self.len_lists[index]
        max_need_len = self.batch_max_length - c_len
        if max_need_len > 2 and random.random() < self.p:
            p_len = random.randint(1, max_need_len - 1)
            p_idx = random.choice(self.len_sample[p_len])
            p_img, p_label = self.read_data(p_idx)
            p_img = cv2.resize(p_img, (tw, th))
            img = cv2.hconcat([img, p_img])
            label = label + p_label

        if self.transforms:
            aug = self.transforms(image=img, label=label)
            img, label = aug['image'], aug['label']

        return img, label
