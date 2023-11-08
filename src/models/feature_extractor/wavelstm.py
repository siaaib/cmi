from typing import Callable, Optional

import torch
import torch.nn as nn

from src.models.feature_extractor.cnn_wave import WaveNetSpectrogram
from src.models.feature_extractor.lstm import LSTMFeatureExtractor


class WavenetSpectrogramLSTMFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int | tuple = 128,
        kernel_size: int = 3,
        wave_layers: tuple = (10, 6, 2),
        downsample: int = 2,
        sigmoid: bool = False,
        output_size: Optional[int] = None,
        reinit: bool = True,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.wave_feature_extractor = WaveNetSpectrogram(
            in_channels=in_channels,
            base_filters=base_filters,
            wave_layers=wave_layers,
            kernel_size=kernel_size,
            downsample=downsample,
            sigmoid=sigmoid,
            output_size=output_size,
            reinit=reinit,
        )
        self.lstm_feature_extractor = LSTMFeatureExtractor(
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            out_size=output_size,
        )
        self.height = self.wave_feature_extractor.height + self.lstm_feature_extractor.height
        self.out_chans = self.wave_feature_extractor.out_chans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            torch.Tensor : (batch_size, out_chans, height, time_steps)
        """

        wave_img = self.wave_feature_extractor(x)  # (batch_size, cnn_chans, height, time_steps)
        lstm_img = self.lstm_feature_extractor(x)
        # (batch_size, in_channels, height, time_steps)
        print(wave_img.shape, lstm_img.shape)
        wave_img = wave_img.view(
            wave_img.shape[0], 1, wave_img.shape[1] * wave_img.shape[2], wave_img.shape[3]
        )
        img = torch.cat([wave_img, lstm_img], dim=1)  # (batch_size, out_chans, height, time_steps)

        return img
