import os

import torch.nn.functional as F
from PIL import Image
import numpy as np
from volksdep.calibrators import EntropyCalibrator, EntropyCalibrator2, \
    MinMaxCalibrator
from volksdep.datasets import Dataset
from volksdep.metrics import Metric as BaseMetric

CALIBRATORS = {
    'entropy': EntropyCalibrator,
    'entropy_2': EntropyCalibrator2,
    'minmax': MinMaxCalibrator,
}


class CalibDataset(Dataset):
    def __init__(self, images_dir, converter, transform=None):
        super(CalibDataset, self).__init__()

        self.root = images_dir
        self.samples = os.listdir(images_dir)
        self.converter = converter
        self.transform = transform

    def __getitem__(self, idx):
        image_file = os.path.join(self.root, self.samples[idx])
        image = Image.open(image_file)
        if self.transform:
            image, _ = self.transform(image=image, label='')
        label = self.converter.test_encode(1)

        return image, label

    def __len__(self):
        return len(self.samples)


class MetricDataset(Dataset):
    def __init__(self, dataset, converter):
        super(MetricDataset, self).__init__()
        self.dataset = dataset
        self.converter = converter

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label_input, _, _ = self.converter.test_encode(1)
        _, _, label_target = self.converter.train_encode([label])

        return (image, label_input[0]), label_target[0]

    def __len__(self):
        return len(self.dataset)


class Metric(BaseMetric):
    def __init__(self, metric, converter):
        self.metric = metric
        self.converter = converter

    def decode(self, preds):
        indexes = np.argmax(preds, 2)
        pred_str = self.converter.decode(indexes)

        return pred_str

    def __call__(self, preds, targets):
        self.metric.reset()
        pred_str = self.decode(preds)
        target_str = self.converter.decode(targets)
        self.metric.measure(pred_str, None, target_str)
        res = self.metric.result

        return ', '.join(['{}: {:.4f}'.format(k, v) for k, v in res.items()])

    def __str__(self):
        return self.metric.__class__.__name__.lower()
