# modify from clovaai

import torch
from torch.utils.data import DataLoader, Subset
from torch._utils import _accumulate


class BaseDataloader(object):
    def __init__(self, dataset, batch_size, each_usage, num_workers=4, shuffle=False):
        if isinstance(each_usage, (float, int)):
            each_usage = [each_usage]
        assert isinstance(each_usage, list)
        assert len(dataset) == len(each_usage)
        self.dataloader_iter_list = []
        self.data_loader_list = []
        self.dataset = self.divide_datasets(dataset, each_usage)

    @staticmethod
    def divide_datasets(dataset_list, eachusage):
        temp_datasets = []
        for idx, dl in enumerate(dataset_list):
            usage = eachusage[idx]
            total_num = len(dl)
            number_dataset = int(total_num * float(usage))
            dataset_split = [number_dataset, total_num - number_dataset]
            indices = range(total_num)
            dl, _ = [Subset(dl, indices[offset - length:offset])
                     for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            temp_datasets.append(dl)
        return temp_datasets

    @property
    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts
