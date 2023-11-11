from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup


class Spec2DCNNMulti(nn.Module):
    def __init__(
        self,
        feature_extractor_1: nn.Module,
        feature_extractor_2: nn.Module,
        decoder_1: nn.Module,
        decoder_2: nn.Module,
        encoder_name: str,
        in_channels_1: int,
        in_channels_2: int,
        encoder_weights: Optional[str] = None,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor_1 = feature_extractor_1
        self.feature_extractor_2 = feature_extractor_2
        self.encoder_1 = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels_1,
            classes=1,
        )
        self.encoder_2 = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels_2,
            classes=1,
        )
        self.decoder_1 = decoder_1
        self.decoder_2 = decoder_2
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_timesteps, n_classes)
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_timesteps, n_classes)
        """
        x_1 = self.feature_extractor_1(x)  # (batch_size, n_channels, height, n_timesteps)
        x_2 = self.feature_extractor_2(x)  # (batch_size, n_channels, height, n_timesteps)
        if do_mixup and labels is not None:
            x_1, labels = self.mixup(x_1, labels)
            x_2, labels = self.mixup(x_2, labels)
        if do_cutmix and labels is not None:
            x_1, labels = self.cutmix(x_1, labels)
            x_2, labels = self.cutmix(x_2, labels)

        x_1 = self.encoder_1(x_1).squeeze(1)  # (batch_size, height, n_timesteps)
        x_2 = self.encoder_2(x_2).squeeze(1)  # (batch_size, height, n_timesteps)
        logits_1 = self.decoder_1(x_1)  # (batch_size, n_classes, n_timesteps)
        logits_2 = self.decoder_2(x_2)  # (batch_size, n_classes, n_timesteps)
        logits = (logits_1 + logits_2) / 2
        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output["loss"] = loss

        return output
