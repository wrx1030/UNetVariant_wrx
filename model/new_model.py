import torch
import torch.nn.functional as F

from .model_parts import *

class newUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(newUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down1 = Down(n_channels, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.endconv = DoubleConv(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc1 = OutConv(32, n_classes, True)
        self.outc2 = OutConv(32, 1)
        # self.fulcon = Fulcon(512*32*32, 64)
        # self.upconv1 = Up_Conv(64, 256, 32)
        # self.ECAblock = ECABlock(512)

    def forward(self, x):
        x1conv, x1 = self.down1(x)
        x2conv, x2 = self.down2(x1)
        x3conv, x3 = self.down3(x2)
        x4conv, x4 = self.down4(x3)
        x5 = self.endconv(x4)
        # x_eca = self.ECAblock(x5)

        # fulconnect
        # y = self.fulcon(x5)
        # y = y.view(-1, 64, 1, 1)
        # y = self.upconv1(y)

        y = self.up1(x5, x4conv)
        y2 = self.up1(x5, x4conv)
        y = y + y2

        y = self.up2(y, x3conv)
        y2 = self.up2(y2, x3conv)
        y = y + y2

        y = self.up3(y, x2conv)
        y2 = self.up3(y2, x2conv)
        y = y + y2

        y = self.up4(y, x1conv)
        y2 = self.up4(y2, x1conv)
        y = y + y2

        logits = self.outc1(y)
        logits2 = self.outc2(y2)

        return logits, logits2


