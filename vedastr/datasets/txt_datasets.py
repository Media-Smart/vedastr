import os

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class TxtDataset(BaseDataset):

    def __init__(self, root, gt_txt, transform=None, character='abcdefghijklmnopqrstuvwxyz0123456789',
                 batch_max_length=25, data_filter_off=False, unknown=False):
        super(TxtDataset, self).__init__(root, gt_txt, transform, character=character,
                                         batch_max_length=batch_max_length,
                                         data_filter_off=data_filter_off,
                                         unknown=unknown)

    def get_name_list(self):
        with open(self.gt_txt, 'r') as gt:
            for line in gt.readlines():
                img_name, label = line.strip().split('\t')
                if self.filter(label):
                    continue
                else:
                    self.img_names.append(os.path.join(self.root, img_name))
                    self.gt_texts.append(label)

        self.samples = len(self.gt_texts)
