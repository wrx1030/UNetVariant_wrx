import torch
import torch.nn.functional as F

from .model_parts import *

class threeBnewUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(threeBnewUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down1 = Down(n_channels, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.endconv = DoubleConv(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up_tb(256, 128)
        self.up3 = Up_tb(128, 64)
        self.up4 = Up_tb(64, 32)
        self.outc1 = OutConv(32, n_classes, True)

        self.fulcon = Fulcon(512*32*32, 64)
        self.upconv1 = Up_Conv(64, 256, 32)

        self.trans_conv1 = Transpose_Conv(256, 256)
        self.trans_conv2 = Transpose_Conv(256, 128)
        self.trans_conv3 = Transpose_Conv(128, 64)
        self.trans_conv4 = Transpose_Conv(64, 32)
        self.outc2 = OutConv(32, 1)

        self.ECAblock1 = ECABlock(256)
        self.ECAblock2 = ECABlock(128)
        self.ECAblock3 = ECABlock(64)

    def forward(self, x):
        xconv, x1 = self.down1(x)
        x1conv, x2 = self.down2(x1)
        x2conv, x3 = self.down3(x2)
        x3conv, x4 = self.down4(x3)
        x5 = self.endconv(x4)
        # x_eca = self.ECAblock(x5)

        # fulconnect
        y = self.fulcon(x5)
        y = y.view(-1, 64, 1, 1)
        z = self.upconv1(y)
        y = self.upconv1(y)


        y = self.trans_conv1(y, x3conv)
        z = self.trans_conv1(z, x3conv)
        x = self.up1(x5, x3conv)
        y = self.trans_conv2(y, x2conv)
        z = self.trans_conv2(z, x2conv)
        yz = torch.cat([y, z], dim=1)
        yz = self.ECAblock1(yz)
        x = self.up2(x, yz)
        y = self.trans_conv3(y, x1conv)
        z = self.trans_conv3(z, x1conv)
        yz = torch.cat([y, z], dim=1)
        yz = self.ECAblock2(yz)
        x = self.up3(x, yz)
        y = self.trans_conv4(y, xconv)
        z = self.trans_conv4(z, xconv)
        yz = torch.cat([y, z], dim=1)
        yz = self.ECAblock3(yz)
        x = self.up4(x, yz)
        logits = self.outc1(x)
        logits2 = self.outc2(y)
        logits3 = self.outc2(z)

        return logits, logits2, logits3


