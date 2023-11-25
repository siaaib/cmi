# ref: https://github.com/bamps53/kaggle-dfl-3rd-place-solution/blob/master/models/cnn_3d.py
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        (
            b,
            c,
            _,
        ) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        norm=nn.BatchNorm1d,
        se=False,
        res=False,
    ):
        super().__init__()
        self.res = res
        if not mid_channels:
            mid_channels = out_channels
        if se:
            non_linearity = SEModule(out_channels)
        else:
            non_linearity = nn.ReLU(inplace=True)
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm(out_channels),
            non_linearity,
        )

    def forward(self, x):
        if self.res:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, in_channels, out_channels, scale_factor, norm=nn.BatchNorm1d, se=False, res=False
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool1d(scale_factor),
            DoubleConv(in_channels, out_channels, norm=norm, se=se, res=res),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels, out_channels, bilinear=True, scale_factor=2, norm=nn.BatchNorm1d
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels // 2, kernel_size=scale_factor, stride=scale_factor
            )
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def create_layer_norm(channel, length):
    return nn.LayerNorm([channel, length])


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        nhead: int,
        n_classes: int,
    ):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_layers
        )
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = self.conv(x)  # (batch_size, n_channels, n_timesteps)
        x = x.transpose(1, 2)  # (batch_size, n_timesteps, n_channels)
        x = self.transformer_encoder(x)
        x = self.linear(x)  # (batch_size, n_timesteps, n_classes)
        return x


class UNet1DDecoderTransformer(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        duration: int,
        bilinear: bool = True,
        se: bool = False,
        res: bool = False,
        scale_factor: int = 2,
        dropout: float = 0.2,
        num_layers: int = 1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.duration = duration
        self.bilinear = bilinear
        self.se = se
        self.res = res
        self.scale_factor = scale_factor
        self.num_layers = num_layers

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(
            self.n_channels, 64, norm=partial(create_layer_norm, length=self.duration)
        )
        self.down1 = Down(
            64, 128, scale_factor, norm=partial(create_layer_norm, length=self.duration // 2)
        )
        self.down2 = Down(
            128, 256, scale_factor, norm=partial(create_layer_norm, length=self.duration // 4)
        )
        self.down3 = Down(
            256, 512, scale_factor, norm=partial(create_layer_norm, length=self.duration // 8)
        )
        self.down4 = Down(
            512,
            1024 // factor,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 16),
        )

        self.up1 = Up(
            1024,
            512 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 8),
        )
        self.up2 = Up(
            512,
            256 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 4),
        )
        self.up3 = Up(
            256,
            128 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 2),
        )
        self.up4 = Up(
            128, 64, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration)
        )
        self.transformer = TransformerDecoder(
            n_channels,
            128,
            2,
            0.2,
            16,
            self.n_classes,
        )
        self.gru = nn.GRU(64, 64, 2, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(128, 64, 1, bidirectional=True, batch_first=True)

        self.cls = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, self.n_classes, kernel_size=1, padding=0),
            nn.Dropout(dropout),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Optional[torch.Tensor]]:
        """Forward

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """

        # 1D U-Net
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_transformer = self.transformer(x)
        x_lstm = x.permute([0, 2, 1])  # b, t, c
        x_lstm, _ = self.gru(x_lstm)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm.permute([0, 2, 1])  # b, c, t
        # classifier
        x_lstm = self.cls(x_lstm).transpose(1, 2)  # (batch_size, n_classes, n_timesteps)
        return 0.25 * x_transformer + 0.75 * x_lstm
