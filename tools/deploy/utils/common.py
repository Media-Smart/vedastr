import os

import cv2

from volksdep.metrics import Metric as BaseMetric
from volksdep.datasets import Dataset
from volksdep.calibrators import EntropyCalibrator, EntropyCalibrator2, \
    MinMaxCalibrator


CALIBRATORS = {
    'entropy': EntropyCalibrator,
    'entropy_2': EntropyCalibrator2,
    'minmax': MinMaxCalibrator,
}


class CalibDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        super(CalibDataset, self).__init__()

        self.root = images_dir
        self.samples = os.listdir(images_dir)
        self.transform = transform

    def __getitem__(self, idx):
        image_file = os.path.join(self.root, self.samples[idx])
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']

        return image

    def __len__(self):
        return len(self.samples)


class Metric(BaseMetric):
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, preds, targets):
        self.metric.reset()
        self.metric.add(preds, targets)
        res = self.metric.result()

        return ', '.join(['{}: {:.4f}'.format(k, v) for k, v in res.items()])

    def __str__(self):
        return self.metric.__class__.__name__.lower()
