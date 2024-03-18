import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU()
                                  )

    def forward(self, x):
        return self.conv(x)

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_padding):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=output_padding)

    def forward(self, x):
        return self.convt(x)

class UNet(nn.Module):
    def __init__(self, init_weights=True):
        super().__init__()

        self.conv1 = ConvBlock(in_channels=3, out_channels=32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = ConvBlock(in_channels=32, out_channels=64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = ConvBlock(in_channels=128, out_channels=256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = ConvBlock(in_channels=256, out_channels=512)
        self.conv6 = ConvBlock(in_channels=512, out_channels=256)
        self.convt1 = ConvTransposeBlock(in_channels=256, out_channels=256, output_padding=(1, 1))
        self.conv7 = ConvBlock(in_channels=256, out_channels=128)
        self.convt2 = ConvTransposeBlock(in_channels=128, out_channels=128, output_padding=(1, 1))
        self.conv8 = ConvBlock(in_channels=128, out_channels=64)
        self.convt3 = ConvTransposeBlock(in_channels=64, out_channels=64, output_padding=(1, 1))
        self.conv9 = ConvBlock(in_channels=64, out_channels=32)
        self.convt4 = ConvTransposeBlock(in_channels=32, out_channels=32, output_padding=(1, 1))
        self.fc = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)


        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:  #  bias가 있다면
                        nn.init.constant_(m.bias, 0)


    def forward(self, x):
        contract1 = self.conv1(x)
        poo1 = self.maxpool1(contract1)
        contract2 = self.conv2(poo1)
        pool2 = self.maxpool2(contract2)
        contract3 = self.conv3(pool2)
        pool3 = self.maxpool3(contract3)
        contract4 = self.conv4(pool3)
        pool4 = self.maxpool4(contract4)
        contract5 = self.conv5(pool4)
        expansive1 = self.conv6(contract5)
        expansive2 = self.convt1(expansive1)
        expansive2_p = F.pad(expansive2, (0, 1, 0, 0))
        expansive2 = torch.cat((expansive2_p, contract4))
        expansive3 = self.conv7(expansive2)
        expansive4 = self.convt2(expansive3)
        expansive4 = torch.cat((expansive4, contract3))
        expansive5 = self.conv8(expansive4)
        expansive6 = self.convt3(expansive5)
        expansive6 = torch.cat((expansive6, contract2))
        expansive7 = self.conv9(expansive6)
        expansive8 = self.convt4(expansive7)
        expansive8 = torch.cat((expansive8, contract1))
        fc = self.fc(expansive8)
        return fc


model = UNet()
summary(model, input_size=(4, 3, 800, 600), device='cuda')



