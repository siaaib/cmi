from typing import Optional

import torch
import torch.nn as nn


class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1, bidir=True):
        super(ResidualBiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(hidden_size * dir_factor, hidden_size * dir_factor)
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor)
        self.fc2 = nn.Linear(hidden_size * dir_factor, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h)
        # res.shape = (batch_size, sequence_size, 2*hidden_size)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # skip connection
        res = res + x

        return res, new_h


class ResidGRUFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        num_layers,
        bidirectional,
        out_size: Optional[int] = None,
    ):
        super().__init__()
        self.fc = nn.Linear(in_channels, hidden_size)
        self.height = hidden_size * (2 if bidirectional else 1)
        self.num_layers = num_layers
        self.res_bigrus = nn.ModuleList(
            [
                ResidualBiGRU(hidden_size, n_layers=1, bidir=bidirectional)
                for _ in range(self.num_layers)
            ]
        )
        self.out_chans = 1
        self.out_size = out_size
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

    def forward(self, x: torch.Tensor, h=None) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            torch.Tensor: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps)
        if h is None:
            # (re)initialize the hidden state
            h = [None for _ in range(self.num_layers)]
        if self.out_size is not None:
            x = x.unsqueeze(1)  # x: (batch_size, 1, in_channels, time_steps)
            x = self.pool(x)  # x: (batch_size, 1, in_channels, output_size)
            x = x.squeeze(1)  # x: (batch_size, in_channels, output_size)
        x = x.transpose(1, 2)  # x: (batch_size, output_size, in_channels)
        x = self.fc(x)  # x: (batch_size, output_size, hidden_size)
        for i, res_bigru in enumerate(self.res_bigrus):
            x, _ = res_bigru(x, h[i])  # x: (batch_size, output_size, hidden_size * num_directions)
        x = x.transpose(1, 2)  # x: (batch_size, hidden_size * num_directions, output_size)
        x = x.unsqueeze(1)  # x: (batch_size, out_chans, hidden_size * num_directions, time_steps)
        return x
