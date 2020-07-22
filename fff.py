import torch
from configs import rosetta as rs
from vedastr.models import build_model
from vedastr.converter import build_converter
import numpy as np


def to(inp, device_or_dtype):
    if not isinstance(inp, (tuple, list)):
        if type(inp).__module__ == torch.__name__:
            if device_or_dtype == 'torch':
                pass
            elif device_or_dtype == 'numpy':
                inp = inp.detach().cpu().numpy()
            else:
                inp = inp.to(device_or_dtype)
        elif type(inp).__module__ == np.__name__:
            if not isinstance(inp, np.ndarray):
                inp = np.array(inp)

            if device_or_dtype == 'torch':
                inp = torch.from_numpy(inp)
            elif device_or_dtype == 'numpy':
                pass
            else:
                inp = inp.astype(device_or_dtype)
        elif isinstance(inp, (int, float)):
            if device_or_dtype == 'torch':
                inp = torch.tensor(inp)
            elif device_or_dtype == 'numpy':
                inp = np.array(inp)
        else:
            raise TypeError('Unsupported type {}, expect int, float, np.ndarray or torch.Tensor'.format(type(inp)))

        return inp
    else:
        out = []
        for x in inp:
            out.append(to(x, device_or_dtype))

        return out


def main():
    model = build_model(rs.deploy['model'])
    converter = build_converter(rs.common['converter'])
    input_args = [torch.Tensor(1, 1, 32, 100).cuda(), torch.Tensor(1, ).cuda()]
    model(input_args)
    print('done')


if __name__ == '__main__':
    main()
