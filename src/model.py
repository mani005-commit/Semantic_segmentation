# Model Architecture
# lightweight UNet model for Pascal VOC segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DepthwiseSeparableConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.conv(x)
        p = self.pool(x)

        return x, p


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.conv = DepthwiseSeparableConv(in_channels, out_channels)

    def forward(self, x, skip):

        x = self.up(x)

        # Fix spatial mismatch caused by pooling/upsampling
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(
                x,
                size=skip.size()[2:],
                mode="bilinear",
                align_corners=True
            )

        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)

        return x


class UNet(nn.Module):

    def __init__(self, num_classes=21):
        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock(3, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)

        # Bottleneck
        self.bottleneck = DepthwiseSeparableConv(128, 256)

        # Decoder
        self.dec3 = DecoderBlock(256 + 128, 128)
        self.dec2 = DecoderBlock(128 + 64, 64)
        self.dec1 = DecoderBlock(64 + 32, 32)

        # Final segmentation layer
        self.final = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        # Final output
        out = self.final(d1)

        return out