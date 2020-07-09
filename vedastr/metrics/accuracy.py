# modify from clovaai

from nltk.metrics.distance import edit_distance

from .registry import METRICS


@METRICS.register_module
class Accuracy(object):

    def __init__(self):
        self.reset()
        self.predict_example_log = None

    def measure(self, preds, preds_prob, gts):
        batch_size = len(gts)
        true_num = 0
        norm_ED = 0
        for pstr, gstr in zip(preds, gts):
            if pstr == gstr:
                true_num += 1

            if len(pstr) == 0 or len(gstr) == 0:
                norm_ED += 0
            elif len(gstr) > len(pstr):
                norm_ED += 1 - edit_distance(pstr, gstr) / len(gstr)
            else:
                norm_ED += 1 - edit_distance(pstr, gstr) / len(pstr)
        self.show_example(preds, preds_prob, gts)
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

    def show_example(self, preds, preds_prob, gts):
        count = 0
        self.predict_example_log = None
        dashed_line = '-' * 80
        head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
        self.predict_example_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
        for gt, pred, prob in zip(gts[:5], preds[:5], preds_prob[:5]):
            self.predict_example_log += f'{gt:25s} | {pred:25s} | {prob:0.4f}\t{str(pred == gt)}\n'
            count += 1
            if count > 4:
                break
        self.predict_example_log += f'{dashed_line}'

        return self.predict_example_log
