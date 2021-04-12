import os

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class FolderDataset(BaseDataset):
    """ Read images in a folder. The format of image filename should be
    same as follows:
    'name_gt.extension', where name represents arbitrary string,
    gt represents the ground-truth of the image, and extension
    represents the postfix (png, jpg, etc.).

    """

    def __init__(self,
                 root: str,
                 transform=None,
                 character: str = 'abcdefghijklmnopqrstuvwxyz0123456789',
                 batch_max_length: int = 100000,
                 data_filter: bool = True,
                 extension_names: tuple = ('.jpg', '.png', '.bmp', '.jpeg')):
        super(FolderDataset, self).__init__(
            root=root,
            transform=transform,
            character=character,
            batch_max_length=batch_max_length,
            data_filter=data_filter)
        self.extension_names = extension_names

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
