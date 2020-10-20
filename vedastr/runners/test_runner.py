import torch

from .inference_runner import InferenceRunner


class TestRunner(InferenceRunner):
    def __init__(self, test_cfg, deploy_cfg, common_cfg=None):
        super(TestRunner, self).__init__(deploy_cfg, common_cfg)

        self.test_dataloader = self._build_dataloader(test_cfg['data'])
        if not isinstance(self.test_dataloader, dict):
            self.test_dataloader = dict(all=self.test_dataloader)
        self.postprocess_cfg = test_cfg.get('postprocess_cfg', None)

    def test_batch(self, img, label):
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

            pred, prob = self.postprocess(pred, self.postprocess_cfg)
            self.metric.measure(pred, prob, label)
            self.backup_metric.measure(pred, prob, label)

    def __call__(self):
        self.logger.info('Start testing')
        self.logger.info('test info: %s' % self.postprocess_cfg)
        self.metric.reset()
        for name, dataloader in self.test_dataloader.items():
            self.backup_metric.reset()
            for img, label in dataloader:
                self.test_batch(img, label)
            self.logger.info('Test, current dataset root %s, acc %.4f, edit distance %.4f' % (
                name, self.backup_metric.avg['acc']['true'], self.metric.avg['edit']
            ))
        self.logger.info('Test, average acc %.4f, edit distance %s' % (self.metric.avg['acc']['true'],
                                                                       self.metric.avg['edit']))
