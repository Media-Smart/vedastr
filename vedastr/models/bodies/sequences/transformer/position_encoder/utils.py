import torch


def generate_encoder(in_channels, max_len):
    pos = torch.arange(max_len).float().unsqueeze(1)

    i = torch.arange(in_channels).float().unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)

    position_encoder = pos * angle_rates
    position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
    position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])

    return position_encoder
