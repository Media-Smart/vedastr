# modify from clovaai

import torch.nn.functional as F
from nltk.metrics.distance import edit_distance


class STRMeters(object):

    def __init__(self, converter):
        self.reset()
        self.converter = converter
        self.predict_example_log = None
        self.sample = []

    def decode(self, pred):
        preds_prob = F.softmax(pred, dim=2)
        preds_max_prob, pred_index = preds_prob.max(dim=2)
        pred_str = self.converter.decode(pred_index)

        return pred_str, preds_max_prob

    def measure(self, pred, gt, batch_size):
        pred_str, preds_prob = self.decode(pred)
        true_num = 0
        norm_ED = 0
        sample_list = []
        confidence_list = []
        for pstr, gstr, pred_prob in zip(pred_str, gt, preds_prob):
            try:
                confidence_score = pred_prob[:len(pstr)].cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            sample_list.append([pstr, gstr, pred_prob])
            if pstr == gstr:
                true_num += 1

            if len(pstr) == 0 or len(gstr) == 0:
                norm_ED += 0
            elif len(gstr) > len(pstr):
                norm_ED += 1 - edit_distance(pstr, gstr) / len(gstr)
            else:
                norm_ED += 1 - edit_distance(pstr, gstr) / len(pstr)
            confidence_list.append(confidence_score)
        self.show_example(pred_str, gt, confidence_list)
        self.sample = sample_list
        self.all['acc']['true'] += true_num
        self.all['acc']['false'] += (batch_size - true_num)
        self.all['edit'] += norm_ED
        self.count += batch_size
        for key, value in self.all['acc'].items():
            self.avg['acc'][key] = self.all['acc'][key] / self.count
        self.avg['edit'] = self.all['edit'] / self.count

    def reset(self):
        self.all = dict(
            acc=dict(
                true=0,
                false=0
            ),
            edit=0
        )
        self.avg = dict(
            acc=dict(
                true=0,
                false=0
            ),
            edit=0
        )
        self.count = 0

    def show_example(self, preds, labels, confidence_score):
        count = 0
        self.predict_example_log = None
        dashed_line = '-' * 80
        head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
        self.predict_example_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
        for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):

            self.predict_example_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
            count += 1
            if count > 4:
                break
        self.predict_example_log += f'{dashed_line}'
        return self.predict_example_log
