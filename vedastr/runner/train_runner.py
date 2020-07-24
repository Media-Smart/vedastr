import os.path as osp
from collections import OrderedDict

import torch

from .deploy_runner import DeployRunner
from ..criteria import build_criterion
from ..lr_schedulers import build_lr_scheduler
from ..optimizers import build_optimizer
from ..utils import save_checkpoint


class TrainRunner(DeployRunner):
    def __init__(self, train_cfg, deploy_cfg, common_cfg=None):
        super(TrainRunner, self).__init__(deploy_cfg, common_cfg)

        self.train_dataloader = self._build_dataloader(
            train_cfg['data']['train'])
        if 'val' in train_cfg['data']:
            self.val_dataloader = self._build_dataloader(
                train_cfg['data']['val'])
        else:
            self.val_dataloader = None

        self.optimizer = self._build_optimizer(train_cfg['optimizer'])
        self.criterion = self._build_criterion(train_cfg['criterion'])
        self.lr_scheduler = self._build_lr_scheduler(train_cfg['lr_scheduler'])
        self.max_iterations = train_cfg['max_iterations']
        self.log_interval = train_cfg.get('log_interval', 10)
        self.trainval_ratio = train_cfg.get('trainval_ratio', -1)
        self.snapshot_interval = train_cfg.get('snapshot_interval', -1)
        self.grad_clip = train_cfg.get('grad_clip', 5)
        self.save_best = train_cfg.get('save_best', True)
        self.best_acc = -1
        self.best_norm = -1
        self.c_iter = 0

        assert self.workdir is not None
        assert self.log_interval > 0

        self.best = OrderedDict()

        if train_cfg.get('resume'):
            self.resume(**train_cfg['resume'])

    def _build_optimizer(self, cfg):
        return build_optimizer(cfg, dict(params=self.model.parameters()))

    def _build_criterion(self, cfg):
        return build_criterion(cfg)

    def _build_lr_scheduler(self, cfg):
        return build_lr_scheduler(cfg, dict(optimizer=self.optimizer))

    def _validate_epoch(self):
        self.logger.info('Iteration %d, Start validating' % self.c_iter)
        self.metric.reset()
        for img, label in self.val_dataloader:
            self._validate_batch(img, label)
        if self.metric.avg['acc']['true'] >= self.best_acc and self.save_best:
            self.best_acc = self.metric.avg['acc']['true']
            self.save_model(out_dir=self.workdir, filename='best_acc.pth', iteration=self.c_iter)
        if self.metric.avg['edit'] >= self.best_norm and self.save_best:
            self.best_norm = self.metric.avg['edit']
            self.save_model(out_dir=self.workdir, filename='best_norm.pth', iteration=self.c_iter)
        self.logger.info('Validate, best_acc %.4f, best_edit %s' % (self.best_acc, self.best_norm))
        self.logger.info('Validate, acc %.4f, edit %s' % (self.metric.avg['acc']['true'], self.metric.avg['edit']))
        self.logger.info(f'\n{self.metric.predict_example_log}')

    def _train_batch(self, img, label):
        self.model.train()

        self.optimizer.zero_grad()

        label_input, label_len, label_target = self.converter.train_encode(label)
        if self.use_gpu:
            img = img.cuda()
            label_input = label_input.cuda()
            label_target = label_target
            label_len = label_len
        if self.need_text:
            pred = self.model((img, label_input))
        else:
            pred = self.model((img,))
        loss = self.criterion(pred, label_target, label_len, img.shape[0])

        loss.backward()
        if self.grad_clip != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            pred, prob = self.postprocess(pred)
            self.metric.measure(pred, prob, label)

        if self.c_iter != 0 and self.c_iter % self.log_interval == 0:
            self.logger.info(
                'Train, Iter %d, LR %s, Loss %.4f, acc %.4f, edit_distance %s' %
                (self.c_iter, self.lr, loss.item(), self.metric.avg['acc']['true'],
                 self.metric.avg['edit']))

            self.logger.info(f'\n{self.metric.predict_example_log}')

    def _validate_batch(self, img, label):
        self.model.eval()
        with torch.no_grad():
            label_input, label_length, label_target = self.converter.test_encode(label)
            if self.use_gpu:
                img = img.cuda()
                label_input = label_input.cuda()
            if self.need_text:
                pred = self.model((img, label_input))
            else:
                pred = self.model((img,))

            pred, prob = self.postprocess(pred)
            self.metric.measure(pred, prob, label)

    def __call__(self):
        self.metric.reset()
        self.logger.info('Start train...')
        for iteration in range(self.max_iterations):
            img, label = self.train_dataloader.get_batch
            self._train_batch(img, label)
            if self.lr_scheduler:
                self.lr_scheduler.step()
                self.c_iter = self.iter
            else:
                self.c_iter = iteration

            if self.trainval_ratio > 0 \
                    and (iteration + 1) % self.trainval_ratio == 0 \
                    and self.val_dataloader:
                self._validate_epoch()
                self.metric.reset()
            if (iteration + 1) % self.snapshot_interval == 0:
                self.save_model(out_dir=self.workdir, filename=f'iter{iteration + 1}.pth', iteration=iteration)

    @property
    def epoch(self):
        return self.lr_scheduler.last_epoch

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
        return [x['lr'] for x in self.optimizer.param_groups]

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
        optimizer = self.optimizer if save_optimizer else None
        self.logger.info('Save checkpoint %s', filename)
        save_checkpoint(self.model,
                        filepath,
                        optimizer=optimizer,
                        meta=meta)

    def resume(self, checkpoint, resume_optimizer=False,
               resume_lr_scheduler=False, resume_meta=False,
               map_location='default'):
        checkpoint = self.load_checkpoint(checkpoint,
                                          map_location=map_location)

        if resume_optimizer and 'optimizer' in checkpoint:
            self.logger.info('Resume optimizer')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if resume_lr_scheduler and 'lr_scheduler' in checkpoint:
            self.logger.info('Resume lr scheduler')
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if resume_meta and 'meta' in checkpoint:
            self.logger.info('Resume meta data')
            self.best = checkpoint['meta']['best']
