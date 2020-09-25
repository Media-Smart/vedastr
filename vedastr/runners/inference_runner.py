import re

import torch
import torch.nn.functional as F

from .base import Common
from ..converter import build_converter
from ..models import build_model
from ..utils import load_checkpoint


class InferenceRunner(Common):
    def __init__(self, deploy_cfg, common_cfg=None):
        deploy_cfg = deploy_cfg.copy()
        common_cfg = {} if common_cfg is None else common_cfg.copy()

        common_cfg['gpu_id'] = deploy_cfg.pop('gpu_id')
        super(InferenceRunner, self).__init__(common_cfg)

        # build test transform
        self.transform = self._build_transform(deploy_cfg['transform'])
        # build converter
        self.converter = self._build_converter(deploy_cfg['converter'])
        # build model
        self.model = self._build_model(deploy_cfg['model'])
        self.postprocess_cfg = deploy_cfg.get('postprocess', None)
        self.model.eval()

    def _build_model(self, cfg):
        self.logger.info('Build model')

        model = build_model(cfg)
        self.need_text = model.need_text
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.cuda()

        return model

    def _build_converter(self, cfg):
        return build_converter(cfg)

    def load_checkpoint(self, filename, map_location='default', strict=True):
        self.logger.info('Load checkpoint from {}'.format(filename))

        if map_location == 'default':
            if self.use_gpu:
                device_id = torch.cuda.current_device()
                map_location = lambda storage, loc: storage.cuda(device_id)
            else:
                map_location = 'cpu'

        return load_checkpoint(self.model, filename, map_location, strict)

    def postprocess(self, preds, cfg=None, label=None):
        if cfg is not None:
            sensitive = cfg.get('sensitive', True)
            character = cfg.get('character', '')
        else:
            sensitive = True
            character = ''

        probs = F.softmax(preds, dim=2)
        max_probs, indexes = probs.max(dim=2)
        preds_str = []
        preds_prob = []
        labels = []
        for i, pstr in enumerate(self.converter.decode(indexes)):
            str_len = len(pstr)
            if str_len == 0:
                prob = 0
            else:
                prob = max_probs[i, :str_len].cumprod(dim=0)[-1]
            preds_prob.append(prob)

            if not sensitive:
                pstr = pstr.lower()
                if label is not None:
                    tmp = label[i].lower()
            if character:
                pstr = re.sub('[^{}]'.format(character), '', pstr)
                if label is not None:
                    tmp = re.sub('[^{}]'.format(character), '', tmp)
                    labels.append(tmp)
            preds_str.append(pstr)
        if label is not None:
             return preds_str, preds_prob, labels

        return preds_str, preds_prob

    def __call__(self, image):
        with torch.no_grad():
            dummy_text = ''
            aug = self.transform(image=image, label=dummy_text)
            image, text = aug['image'], aug['label']
            image = image.unsqueeze(0)
            label_input, label_length, label_target = self.converter.test_encode([text])
            if self.use_gpu:
                image = image.cuda()
                label_input = label_input.cuda()

            if self.need_text:
                pred = self.model((image, label_input))
            else:
                pred = self.model((image,))

            pred, prob = self.postprocess(pred, self.postprocess_cfg)

        return pred, prob
