import os

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class FolderDataset(BaseDataset):
    extension_names = ['.jpg', '.png', '.bmp', '.jpeg']

    def __init__(self, *args, **kwargs):
        super(FolderDataset, self).__init__(*args, **kwargs)

    @staticmethod
    def parse_filename(text):
        return text.split('_')[-1]

    def get_name_list(self):
        for item in os.listdir(self.root):
            file_name, file_extension = os.path.splitext(item)
            if file_extension in self.extension_names:
                label = self.parse_filename(file_name)
                if self.filter(label):
                    continue
                else:
                    self.img_names.append(os.path.join(self.root, item))
                    self.gt_texts.append(label)
        self.samples = len(self.gt_texts)
