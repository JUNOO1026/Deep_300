import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.d_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU()
                                    )

    def forward(self, x):
        return self.d_conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.d_conv = DoubleConv(in_channels, out_channels)
        self.d_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.d_conv(x)
        down = self.d_sample(skip)
        return down, skip


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.d_conv = DoubleConv(in_channels, out_channels)

    def forward(self, u_input, skip_input):
        up = self.up_conv(u_input)

        if up.shape != skip_input.shape:
            up = F.pad(up, (0, 1, 0, 0))

        x = torch.cat([up, skip_input], dim=1)

        return self.d_conv(x)


class UNet(nn.Module):
    def __init__(self, out_classes=2):
        super().__init__()
        # Contracting Path
        self.down_conv1 = DownConv(in_channels=3, out_channels=64)
        self.down_conv2 = DownConv(in_channels=64, out_channels=128)
        self.down_conv3 = DownConv(in_channels=128, out_channels=256)
        self.down_conv4 = DownConv(in_channels=256, out_channels=512)
        # Double path
        self.double_conv = DoubleConv(in_channels=512, out_channels=1024)
        # Expanding Path
        self.up_conv1 = UpConv(in_channels=1024, out_channels=512)
        self.up_conv2 = UpConv(in_channels=512, out_channels=256)
        self.up_conv3 = UpConv(in_channels=256, out_channels=128)
        self.up_conv4 = UpConv(in_channels=128, out_channels=64)
        # Final Path
        self.final_path = DoubleConv(in_channels=64, out_channels=out_classes)

    def forward(self, x):
        x, skip1 = self.down_conv1(x)
        x, skip2 = self.down_conv2(x)
        x, skip3 = self.down_conv3(x)
        x, skip4 = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv1(x, skip4)
        x = self.up_conv2(x, skip3)
        x = self.up_conv3(x, skip2)
        x = self.up_conv4(x, skip1)
        x = self.final_path(x)
        return x

from torchinfo import summary

model = UNet()
