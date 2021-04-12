import cv2
import numpy as np
import torch

from .inference_runner import InferenceRunner


class TestRunner(InferenceRunner):

    def __init__(self, test_cfg, inference_cfg, common_cfg=None):
        super(TestRunner, self).__init__(inference_cfg, common_cfg)

        self.test_dataloader = self._build_dataloader(test_cfg['data'])
        if not isinstance(self.test_dataloader, dict):
            self.test_dataloader = dict(all=self.test_dataloader)
        self.test_exclude_num = dict()
        for k, v in self.test_dataloader.items():
            extra_data = len(v.dataset) % self.world_size
            self.test_exclude_num[
                k] = self.world_size - extra_data if extra_data != 0 else 0
        self.postprocess_cfg = test_cfg.get('postprocess_cfg', None)

    def test_batch(self, img, label, save_path=None, exclude_num=0):
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

            if save_path is not None:
                for idx, (p, l) in enumerate(zip(pred, label)):
                    if p == l:
                        print(p, '\t', l)
                        cimg = img[idx][0, :, :].cpu().numpy()
                        cimg = (cimg * 0.5) + 0.5
                        cv2.imwrite(save_path + f'/%s_{p}_{l}.png' % idx,
                                    (cimg * 255).astype(np.uint8))
            self.metric.measure(pred, prob, label, exclude_num)
            self.backup_metric.measure(pred, prob, label, exclude_num)

    def __call__(self):
        self.logger.info('Start testing')
        self.logger.info('test info: %s' % self.postprocess_cfg)
        self.metric.reset()
        accs = []
        for name, dataloader in self.test_dataloader.items():
            # pdb.set_trace()
            # print(self.test_exclude_num)
            test_exclude_num = self.test_exclude_num[name]
            save_name = name.split('/')[-1]
            save_path = './goodcase/%s' % save_name
            save_path = None
            # os.makedirs(save_path)
            self.backup_metric.reset()
            for tidx, (img, label) in enumerate(dataloader):
                exclude_num = test_exclude_num if (tidx +
                                                   1) == len(dataloader) else 0
                # if exclude_num !=0 :
                #     print ('current_exclude num:', exclude_num)
                self.test_batch(img, label, save_path, exclude_num)
            accs.append(self.backup_metric.avg['acc']['true'])
            self.logger.info(
                'Test, current dataset root %s, acc %.4f, edit distance %.4f' %
                (name, self.backup_metric.avg['acc']['true'],
                 self.backup_metric.avg['edit']))
        self.logger.info(
            'Test, average acc %.4f, edit distance %s' %
            (self.metric.avg['acc']['true'], self.metric.avg['edit']))
        acc_str = ' '.join(list(map(lambda x: str(x)[:6], accs)))
        self.logger.info('For copy and record, %s' % acc_str)
