"""
Architektur B, Stufe 1 und 2.

UNetResNet18          U-Net mit ResNet-18-Encoder (ImageNet); trennt Hintergrund, Hauptschild,
                      andere Schilder und Mast (4 Klassen, src/data/segmentation.py)
ZusatzKlassifikator   ResNet-18 für ein einzelnes, ausgeschnittenes Zusatzschild (eine Klasse je Schild)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def _weights(pretrained):
    return ResNet18_Weights.IMAGENET1K_V1 if pretrained else None


class _Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class UNetResNet18(nn.Module):
    def __init__(self, n_klassen: int = 4, pretrained: bool = True):
        super().__init__()
        b = resnet18(weights=_weights(pretrained))
        self.stem = nn.Sequential(b.conv1, b.bn1, b.relu)          # 1/2, 64
        self.pool = b.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = b.layer1, b.layer2, b.layer3, b.layer4  # 1/4 .. 1/32
        self.up4 = _Up(512, 256, 256)
        self.up3 = _Up(256, 128, 128)
        self.up2 = _Up(128, 64, 64)
        self.up1 = _Up(64, 64, 64)
        self.head = nn.Conv2d(64, n_klassen, 1)

    def forward(self, x):
        s0 = self.stem(x)
        e1 = self.enc1(self.pool(s0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d = self.up1(self.up2(self.up3(self.up4(e4, e3), e2), e1), s0)
        return F.interpolate(self.head(d), size=x.shape[-2:], mode="bilinear", align_corners=False)


class ZusatzKlassifikator(nn.Module):
    def __init__(self, n_klassen: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        b = resnet18(weights=_weights(pretrained))
        b.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, n_klassen))
        self.net = b

    def forward(self, x):
        return self.net(x)
