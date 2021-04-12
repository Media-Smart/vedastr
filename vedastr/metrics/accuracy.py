# modify from clovaai
import random
import torch
from nltk.metrics.distance import edit_distance

from ..utils import gather_tensor, get_dist_info
from .registry import METRICS


@METRICS.register_module
class Accuracy(object):

    def __init__(self):
        self.reset()
        self.predict_example_log = None

    @property
    def result(self):
        res = {
            'acc': self.avg['acc']['true'],
            'edit_distance': self.avg['edit'],
        }
        return res

    def measure(self, preds, preds_prob, gts, exclude_num=0):
        batch_size = len(gts)
        true_nums = []
        norm_EDs = []
        r, w = get_dist_info()
        for pstr, gstr in zip(preds, gts):
            if pstr == gstr:
                true_nums.append(1.)
            else:
                true_nums.append(0.)
            if len(pstr) == 0 or len(gstr) == 0:
                norm_EDs.append(0)
            elif len(gstr) > len(pstr):
                norm_EDs.append(1 - edit_distance(pstr, gstr) / len(gstr))
            else:
                norm_EDs.append(1 - edit_distance(pstr, gstr) / len(pstr))
        # gather batch_size, true_num, norm_ED from different workers
        batch_sizes = gather_tensor(torch.tensor(batch_size)[None].cuda())
        true_nums = gather_tensor(
            torch.tensor(true_nums)[None].cuda()).flatten()
        norm_EDs = gather_tensor(torch.tensor(norm_EDs)[None].cuda()).flatten()

        # remove exclude data
        if exclude_num != 0:
            batch_size = torch.sum(batch_sizes).cpu().numpy() - exclude_num
            true_nums = list(true_nums.split(true_nums.shape[0] // w))
            for i in range(1, exclude_num + 1):
                true_nums[-i] = true_nums[-i][:-1]
            true_nums = torch.cat(true_nums)
            # true_nums = true_nums.flatten()[:-exclude_num]
            norm_EDs = list(norm_EDs.split(true_nums.shape[0] // w))
            for i in range(1, exclude_num + 1):
                norm_EDs[-i] = norm_EDs[-i][:-1]
            norm_EDs = torch.cat(norm_EDs)
            # norm_EDs = norm_EDs.flatten()[:-exclude_num]
        else:
            batch_size = torch.sum(batch_sizes).cpu().numpy()

        true_num = torch.sum(true_nums).cpu().numpy()
        norm_ED = torch.sum(norm_EDs).cpu().numpy()

        if preds_prob is not None:
            self.show_example(preds, preds_prob, gts)
        self.all['acc']['true'] += true_num
        self.all['acc']['false'] += (batch_size - true_num)
        self.all['edit'] += norm_ED
        self.count += batch_size
        for key, value in self.all['acc'].items():
            self.avg['acc'][key] = self.all['acc'][key] / self.count
        self.avg['edit'] = self.all['edit'] / self.count

    def reset(self):
        self.all = dict(acc=dict(true=0, false=0), edit=0)
        self.avg = dict(acc=dict(true=0, false=0), edit=0)
        self.count = 0

    def show_example(self, preds, preds_prob, gts):
        count = 0
        self.predict_example_log = None
        dashed_line = '-' * 80
        head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'  # noqa 501
        self.predict_example_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
        show_inds = list(range(len(gts)))
        random.shuffle(show_inds)
        show_inds = show_inds[:5]
        show_gts = [gts[i] for i in show_inds]
        show_preds = [preds[i] for i in show_inds]
        show_prob = [preds_prob[i] for i in show_inds]
        for gt, pred, prob in zip(show_gts, show_preds, show_prob):
            self.predict_example_log += f'{gt:25s} | {pred:25s} | {prob:0.4f}\t{str(pred == gt)}\n'  # noqa 501
            count += 1
            if count > 4:
                break
        self.predict_example_log += f'{dashed_line}'

        return self.predict_example_log
