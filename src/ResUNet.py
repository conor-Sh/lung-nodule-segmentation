from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.projection is not None:
            identity = self.projection(identity)

        out += identity
        return F.relu(out)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.res_block(x)
        down = self.max_pool(skip)
        return skip, down


class UpBlock(nn.Module):
    def __init__(self, decoder_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up_convolution = nn.ConvTranspose2d(
            decoder_channels, out_channels, kernel_size=2, stride=2
        )
        self.res_block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_convolution(x)
        if x.size() != skip.size():
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.res_block(x)


class ResUNet(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.down1 = Encoder(1, 64)
        self.down2 = Encoder(64, 128)
        self.down3 = Encoder(128, 256)
        self.down4 = Encoder(256, 512)

        self.bottleneck = ResidualBlock(512, 1024)

        self.up1 = UpBlock(1024, 512, 512)
        self.up2 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up4 = UpBlock(128, 64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.bottleneck(x)
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        return self.final_conv(x)
