import os

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class TxtDataset(BaseDataset):
    """ Read images based on the txt file.
    The format of lines in txt should be same as follows:
        image_path  label

    The image path and label should be split with '\t'.

    """

    def __init__(
        self,
        root: str,
        gt_txt: str,
        transform=None,
        character: str = 'abcdefghijklmnopqrstuvwxyz0123456789',
        batch_max_length: int = 25,
        data_filter: bool = True,
    ):
        super(TxtDataset, self).__init__(
            root=root,
            transform=transform,
            character=character,
            batch_max_length=batch_max_length,
            data_filter=data_filter,
        )
        if gt_txt is not None:
            assert os.path.isfile(gt_txt)
            self.gt_txt = gt_txt

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
