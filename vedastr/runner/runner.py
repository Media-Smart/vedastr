import torch
import logging
import os.path as osp
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable

from vedastr.utils.checkpoint import load_checkpoint, save_checkpoint

from .registry import RUNNERS

np.set_printoptions(precision=4)

logger = logging.getLogger()


@RUNNERS.register_module
class Runner(object):
    """ Runner

    """

    def __init__(self,
                 loader,
                 model,
                 converter,
                 criterion,
                 metric,
                 optim,
                 lr_scheduler,
                 iterations,
                 workdir,
                 trainval_ratio=1,
                 snapshot_interval=1,
                 gpu=True,
                 test_cfg=None,
                 test_mode=False,
                 need_text=False,
                 grad_clip=0):
        self.loader = loader
        self.model = model
        self.converter = converter
        self.criterion = criterion
        self.metric = metric
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.iterations = iterations
        self.workdir = workdir
        self.trainval_ratio = trainval_ratio
        self.snapshot_interval = snapshot_interval
        self.gpu = gpu
        self.test_cfg = test_cfg
        self.test_mode = test_mode
        self.need_text = need_text
        self.grad_clip = grad_clip
        self.best_norm = 0
        self.best_acc = 0
        self.c_iter = 0

    def __call__(self):
        if self.test_mode:
            self.test_epoch()
        else:
            self.metric.reset()
            logger.info('Start train...')
            for iteration in range(self.iterations):
                img, label = self.loader['train'].get_batch
                self.train_batch(img, label)
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                    self.c_iter = self.iter
                else:
                    self.c_iter = iteration

                if self.trainval_ratio > 0 \
                        and (iteration + 1) % self.trainval_ratio == 0 \
                        and self.loader.get('val'):
                    self.validate_epoch()
                    self.metric.reset()
                if iteration % self.snapshot_interval == 0:
                    self.save_model(out_dir=self.workdir,
                                    filename=f'iter{iteration}.pth',
                                    iteration=iteration,
                                    )

    def validate_epoch(self):
        logger.info('Iteration %d, Start validating' % self.c_iter)
        self.metric.reset()
        for img, label in self.loader['val']:
            self.validate_batch(img, label)
        if self.metric.avg['acc']['true'] >= self.best_acc:
            self.best_acc = self.metric.avg['acc']['true']
            self.save_model(out_dir=self.workdir,
                            filename='best_acc.pth',
                            iteration=self.c_iter)
        if self.metric.avg['edit'] >= self.best_norm:
            self.best_norm = self.metric.avg['edit']
            self.save_model(out_dir=self.workdir,
                            filename='best_norm.pth',
                            iteration=self.c_iter)
        logger.info('Validate, best_acc %.4f, best_edit %s' % (self.best_acc, self.best_norm))
        logger.info('Validate, acc %.4f, edit %s' % (self.metric.avg['acc']['true'],
                                                     self.metric.avg['edit']))
        logger.info(f'\n{self.metric.predict_example_log}')

    def test_epoch(self):
        logger.info('Start testing')
        logger.info('test info: %s' % self.test_cfg)
        self.metric.reset()
        for img, label in self.loader['test']:
            self.test_batch(img, label)

        logger.info('Test, acc %.4f, edit %s' % (self.metric.avg['acc']['true'],
                                                 self.metric.avg['edit']))

    def train_batch(self, img, label):
        self.model.train()

        self.optim.zero_grad()

        label_input, label_len, label_target = self.converter.train_encode(label)
        if self.gpu:
            img = img.cuda()
            label_input = label_input.cuda()
            label_target = label_target
            label_len = label_len
        if self.need_text:
            pred = self.model(img, label_input)
        else:
            pred = self.model(img)
        loss = self.criterion(pred, label_target, label_len, img.shape[0])

        loss.backward()
        if self.grad_clip != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optim.step()

        self.metric.measure(pred, label, pred.shape[0])

        if self.c_iter != 0 and self.c_iter % 10 == 0:
            logger.info(
                'Train, Iter %d, LR %s, Loss %.4f, acc %.4f, edit_distance %s' %
                (self.c_iter, self.lr, loss.item(), self.metric.avg['acc']['true'],
                 self.metric.avg['edit']))

            logger.info(f'\n{self.metric.predict_example_log}')

    def validate_batch(self, img, label):
        self.model.eval()
        with torch.no_grad():
            label_input, label_length, label_target = self.converter.test_encode(label)
            if self.gpu:
                img = img.cuda()
                label_input = label_input.cuda()
            if self.need_text:
                pred = self.model(img, label_input)
            else:
                pred = self.model(img)

            self.metric.measure(pred, label, pred.shape[0])

    def test_batch(self, img, label):
        self.model.eval()
        with torch.no_grad():
            label_input, label_length, label_target = self.converter.test_encode(label)
            if self.gpu:
                img = img.cuda()
                label_input = label_input.cuda()

            if self.need_text:
                pred = self.model(img, label_input)
            else:
                pred = self.model(img)
            self.metric.measure(pred, label, pred.shape[0])

    def save_model(self,
                   out_dir,
                   filename,
                   iteration,
                   save_optimizer=True,
                   meta=None):
        if meta is None:
            meta = dict(iter=iteration, lr=self.lr)
        else:
            meta.update(iter=iteration, lr=self.lr)

        filepath = osp.join(out_dir, filename)
        optimizer = self.optim if save_optimizer else None
        logger.info('Save checkpoint %s', filename)
        save_checkpoint(self.model,
                        filepath,
                        optimizer=optimizer,
                        meta=meta)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        logger.info('Resume from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               logger)

    @property
    def iter(self):
        """int: Current iteration."""
        return self.lr_scheduler.last_iter

    @iter.setter
    def iter(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_iter = val

    @property
    def lr(self):
        lr = [x['lr'] for x in self.optim.param_groups]
        return np.array(lr)

    @lr.setter
    def lr(self, val):
        for idx, param in enumerate(self.optim.param_groups):
            if isinstance(val, Iterable):
                param['lr'] = val[idx]
            else:
                param['lr'] = val

    def resume(self,
               checkpoint,
               resume_optimizer=False,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optim.load_state_dict(checkpoint['optimizer'])
