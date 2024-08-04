from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .drop import DropPath
from .attbase import attbase

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class newAtt(nn.Module):
    dim_head = 16
    def __init__(self, input_dim, channel_wise=False):
        super(newAtt, self).__init__()
        if channel_wise:
            self.dim_head = 1
        self.mrla = attbase(input_dim=input_dim, dim_head=self.dim_head)
    def forward(self, xt):
        attfeature = self.mrla(xt)
        return attfeature

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class AttResUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64,
                 drop_path=0.2,
                 channel_wise_mrla=False):
        super(AttResUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)

        self.in_conv = DoubleConv(in_channels, base_c)
        self.mrla1 = newAtt(input_dim=base_c, channel_wise=channel_wise_mrla)
        self.bn_mrla1 = norm_layer(base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.mrla2 = newAtt(input_dim=base_c *2, channel_wise=channel_wise_mrla)
        self.bn_mrla2 = norm_layer(base_c *2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.mrla3 = newAtt(input_dim=base_c *4, channel_wise=channel_wise_mrla)
        self.bn_mrla3 = norm_layer(base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.mrla4 = newAtt(input_dim=base_c *8, channel_wise=channel_wise_mrla)
        self.bn_mrla4 = norm_layer(base_c * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        atten1 = self.mrla1(x1)
        atten1 = self.bn_mrla1(atten1)
        atten1 = self.relu(atten1)
        x1 = x1 + self.drop_path(atten1)
        x2 = self.down1(x1)
        atten2 = self.mrla2(x2)
        atten2 = self.bn_mrla2(atten2)
        atten2 = self.relu(atten2)
        x2 = x2 + self.drop_path(atten2)
        x3 = self.down2(x2)
        atten3 = self.mrla3(x3)
        atten3 = self.bn_mrla3(atten3)
        atten3 = self.relu(atten3)
        x3 = x3 + self.drop_path(atten3)
        x4 = self.down3(x3)
        atten4 = self.mrla4(x4)
        atten4 = self.bn_mrla4(atten4)
        atten4 = self.relu(atten4)
        x4 = x4 + self.drop_path(atten4)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        logits = self.out_conv(x9)
        return {"out": logits}
