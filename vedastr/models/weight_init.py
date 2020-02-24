# modify from mmcv and mmdetection

import torch.nn as nn


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 is_rnn=False,
                 mode='fan_in',
                 nonlinearity='leaky_relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        if is_rnn:
            for name, param in module.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, bias)
                elif 'weight' in name:
                    nn.init.kaiming_uniform_(param,
                                             a=a,
                                             mode=mode,
                                             nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_uniform_(module.weight,
                                     a=a,
                                     mode=mode,
                                     nonlinearity=nonlinearity)

    else:
        if is_rnn:
            for name, param in module.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, bias)
                elif 'weight' in name:
                    nn.init.kaiming_normal_(param,
                                            a=a,
                                            mode=mode,
                                            nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight,
                                    a=a,
                                    mode=mode,
                                    nonlinearity=nonlinearity)

    if not is_rnn and hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    # Acknowledgment to FAIR's internal code
    kaiming_init(module,
                 a=1,
                 mode='fan_in',
                 nonlinearity='leaky_relu',
                 distribution='uniform')


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            kaiming_init(m)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            constant_init(m, 1)
        elif isinstance(m, nn.Linear):
            kaiming_init(m)
        elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
            kaiming_init(m, is_rnn=True)
