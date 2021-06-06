import os.path as osp

import torch
import torch.utils.data as tud

from .inference_runner import InferenceRunner
from ..criteria import build_criterion
from ..lr_schedulers import build_lr_scheduler
from ..optimizers import build_optimizer
from ..utils import save_checkpoint, gather_tensor


class TrainRunner(InferenceRunner):

    def __init__(self, train_cfg, inference_cfg, common_cfg=None):
        super(TrainRunner, self).__init__(inference_cfg, common_cfg)

        self.train_dataloader = self._build_dataloader(
            train_cfg['data']['train'])
        assert isinstance(self.train_dataloader, tud.DataLoader), \
            "Only support single dataloader in training phase. " \
            "Check the type of dataset please. " \
            "If you wanna combine different dataset, " \
            "consider using ConcatDataset in your config file please."

        if 'val' in train_cfg['data']:
            self.val_dataloader = self._build_dataloader(
                train_cfg['data']['val'])
            extra_data = len(self.val_dataloader.dataset) % self.world_size
            self.val_exclude_num = self.world_size - extra_data if extra_data != 0 else 0  # noqa 501
        else:
            self.val_dataloader = None

        self.train_cfg = train_cfg

        self.max_iterations = train_cfg.get('max_iterations', False)
        self.max_epochs = train_cfg.get('max_epochs', False)
        assert self.max_epochs ^ self.max_iterations, \
            'max_epochs and max_iterations are mutual exclusion'
        if not self.max_iterations:
            self.max_iterations = len(self.train_dataloader) * self.max_epochs
        if not self.max_epochs:
            self.max_epochs = self.max_iterations // len(self.train_dataloader)

        self.optimizer = self._build_optimizer(train_cfg['optimizer'])
        self.criterion = self._build_criterion(train_cfg['criterion'])
        self.lr_scheduler = self._build_lr_scheduler(train_cfg['lr_scheduler'])

        self.log_interval = train_cfg.get('log_interval', 10)
        self.trainval_ratio = train_cfg.get('trainval_ratio', -1)
        self.snapshot_interval = train_cfg.get('snapshot_interval', -1)
        self.grad_clip = train_cfg.get('grad_clip', 5)
        self.save_best = train_cfg.get('save_best', True)
        self.best_acc = -1
        self.best_norm = -1

        assert self.workdir is not None
        assert self.log_interval > 0

        if train_cfg.get('resume'):
            self.resume(**train_cfg['resume'])

    def _build_optimizer(self, cfg):
        return build_optimizer(cfg, dict(params=self.model.parameters()))

    def _build_criterion(self, cfg):
        if self.converter.ignore_index is not None:
            if cfg['type'] in ['CrossEntropyLoss', 'LabelSmoothingCrossEntropy']:
                self.logger.info(f'Set ignore index as {self.converter.ignore_index}')
                cfg.update(ignore_index=self.converter.ignore_index)

        return build_criterion(cfg)

    def _build_lr_scheduler(self, cfg, last_iter=-1):

        return build_lr_scheduler(
            cfg,
            dict(
                optimizer=self.optimizer,
                niter_per_epoch=len(self.train_dataloader),
                max_epochs=self.max_epochs,
                last_iter=last_iter,
            ))

    def _validate_epoch(self):
        self.logger.info('Iteration %s, Start validating' % (self.iter + 1))
        self.metric.reset()
        for vidx, (img, label) in enumerate(self.val_dataloader):
            exclude_num = self.val_exclude_num if vidx == len(
                self.val_dataloader) else 0
            self._validate_batch(img, label, exclude_num)
        if self.metric.avg['acc']['true'] >= self.best_acc and \
                self.save_best and self.rank == 0:
            self.best_acc = self.metric.avg['acc']['true']
            self.save_model(out_dir=self.workdir, filename='best_acc.pth')
        if self.metric.avg['edit'] >= self.best_norm and \
                self.save_best and self.rank == 0:
            self.best_norm = self.metric.avg['edit']
            self.save_model(out_dir=self.workdir, filename='best_norm.pth')
        self.logger.info('Validate, best_acc %.4f, best_edit %s' %
                         (self.best_acc, self.best_norm))
        self.logger.info(
            'Validate, acc %.4f, edit %s' %
            (self.metric.avg['acc']['true'], self.metric.avg['edit']))
        self.logger.info(f'\n{self.metric.predict_example_log}')

    def _train_batch(self, img, label):
        self.model.train()

        self.optimizer.zero_grad()

        label_input, label_len, label_target = self.converter.train_encode(label)  # noqa 501
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
        all_loss = gather_tensor(loss.detach())
        gather_loss = torch.mean(all_loss)
        loss.backward()
        if self.grad_clip != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            pred, prob = self.postprocess(pred)
            self.metric.measure(pred, prob, label)

        if self.iter != 0 and self.iter % self.log_interval == 0:
            self.logger.info(
                'Train, Epoch %d, Iter %d, LR %s, Loss %.4f, '
                'acc %.4f, edit_distance %s'
                % (self.epoch, self.iter, self.lr, gather_loss.item(),
                   self.metric.avg['acc']['true'], self.metric.avg['edit']))

            self.logger.info(f'\n{self.metric.predict_example_log}')

    def _validate_batch(self, img, label, exclude_num):
        self.model.eval()
        with torch.no_grad():
            label_input, label_length, label_target = self.converter.test_encode(label)  # noqa 501
            if self.use_gpu:
                img = img.cuda()
                label_input = label_input.cuda()
            if self.need_text:
                pred = self.model((img, label_input))
            else:
                pred = self.model((img,))

            pred, prob = self.postprocess(pred, self.postprocess_cfg)
            self.metric.measure(pred, prob, label, exclude_num)

    def __call__(self):
        self.metric.reset()
        self.logger.info('Start train...')
        iter_based = self.lr_scheduler._iter_based
        warmup_iters = self.lr_scheduler.warmup_iters
        flag = True
        count = 0
        while flag:
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(count)
            if hasattr(self.train_dataloader.worker_init_fn, 'set_epoch'):
                self.train_dataloader.worker_init_fn.set_epoch(count)
            for img, label in self.train_dataloader:
                self._train_batch(img, label)
                self.lr_scheduler.iter_nums()  # update steps
                if iter_based:
                    self.lr_scheduler.step()
                elif warmup_iters > 0 and warmup_iters >= self.iter:
                    self.lr_scheduler.step()
                if self.trainval_ratio > 0 \
                        and (self.iter + 1) % self.trainval_ratio == 0 \
                        and self.val_dataloader:
                    self._validate_epoch()
                    self.metric.reset()
                if (self.iter + 1) % self.snapshot_interval == 0 and \
                        self.rank == 0:
                    self.save_model(
                        out_dir=self.workdir,
                        filename=f'iter{self.iter + 1}.pth')
                if self.iter >= self.max_iterations:
                    flag = False
                    break
            if not iter_based:
                self.lr_scheduler.step()
            count += 1
        self._validate_epoch()
        self.logger.info(
            'Ending of training, save the model of the last iterations.')
        if self.rank == 0:
            self.save_model(out_dir=self.workdir, filename='final.pth')

    @property
    def epoch(self):
        return self.lr_scheduler.last_epoch

    @epoch.setter
    def epoch(self, val):
        self.lr_scheduler.last_epoch = val

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
                   save_optimizer=True,
                   save_lr_scheduler=True,
                   meta=None):
        current_meta = dict(
            epoch=self.epoch,
            iter=self.iter,
            lr=self.lr,
            best_acc=self.best_acc,
            best_norm=self.best_norm)
        if meta is None:
            meta = current_meta
        else:
            meta.update(current_meta)

        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        lr_scheduler = self.lr_scheduler if save_lr_scheduler else None
        self.logger.info('Save checkpoint %s', filename)
        save_checkpoint(
            self.model,
            filepath,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            meta=meta)

    def resume(self,
               checkpoint,
               resume_optimizer=False,
               resume_lr_scheduler=False,
               resume_meta=False,
               strict=True,
               map_location='default'):
        checkpoint = self.load_checkpoint(
            checkpoint, map_location=map_location, strict=strict)

        if resume_optimizer and 'optimizer' in checkpoint:
            self.logger.info('Resume optimizer')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if resume_lr_scheduler and 'lr_scheduler' in checkpoint:
            self.logger.info('Resume lr scheduler')
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if resume_meta and 'meta' in checkpoint:
            self.logger.info('Resume meta data')
            meta = checkpoint['meta']
            self.best_acc = meta.get('best_acc', -1)
            self.best_norm = meta.get('best_norm', -1)
            self.iter = meta.get('iter', 0)
            self.epoch = meta.get('epoch', 0)
