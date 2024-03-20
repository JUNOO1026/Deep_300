import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

cfgs = {"basic": [4, 4]}


class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.Cconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()
                                   )

    def forward(self, x):
        return self.Cconv(x)


class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Econv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                                        bias=False)

    def forward(self, x):
        return self.Econv(x)

class UNet(nn.Module):
    def __init__(self, out_classes=2, init_weights=True):
        super().__init__()
        self.C_conv1 = ContractingBlock(in_channels=3, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.C_conv2 = ContractingBlock(in_channels=64, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.C_conv3 = ContractingBlock(in_channels=128, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.C_conv4 = ContractingBlock(in_channels=256, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.C_conv5 = ContractingBlock(in_channels=512, out_channels=1024)
        self.E_conv1 = ExpandingBlock(in_channels=1024, out_channels=512)
        self.C_conv6 = ContractingBlock(in_channels=2*512, out_channels=512)
        self.E_conv2 = ExpandingBlock(in_channels=512, out_channels=256)
        self.C_conv7 = ContractingBlock(in_channels=2*256, out_channels=256)
        self.E_conv3 = ExpandingBlock(in_channels=256, out_channels=128)
        self.C_conv8 = ContractingBlock(in_channels=2*128, out_channels=128)
        self.E_conv4 = ExpandingBlock(in_channels=128, out_channels=64)
        self.C_conv9 = ContractingBlock(in_channels=2*64, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=out_classes, kernel_size=3, stride=1, padding=1)

        if init_weights:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        contract1 = self.C_conv1(x)
        contract1_2 = self.pool1(contract1)
        contract2 = self.C_conv2(contract1_2)
        contract2_2 = self.pool2(contract2)
        contract3 = self.C_conv3(contract2_2)
        contract3_2 = self.pool3(contract3)
        contract4 = self.C_conv4(contract3_2)
        contract4_2 = self.pool4(contract4)
        contract5 = self.C_conv5(contract4_2)
        expand1 = self.E_conv1(contract5)
        expand1_add = F.pad(expand1, (0, 1, 0, 0))
        expand_fn1 = torch.cat((expand1_add, contract4), dim=1)
        repeat1 = self.C_conv6(expand_fn1)
        expand2 = self.E_conv2(repeat1)
        expand_fn2 = torch.cat((expand2, contract3), dim=1)
        repeat2 = self.C_conv7(expand_fn2)
        expand3 = self.E_conv3(repeat2)
        expand_fn3 = torch.cat((expand3, contract2), dim=1)
        repeat3 = self.C_conv8(expand_fn3)
        expand4 = self.E_conv4(repeat3)
        expand_fn4 = torch.cat((expand4, contract1), dim=1)
        repeat4 = self.C_conv9(expand_fn4)
        output = self.output(repeat4)

        return output

# model = UNet()
# summary(model, (2, 3, 800, 600), device='cuda')
