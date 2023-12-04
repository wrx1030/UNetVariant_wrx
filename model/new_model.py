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

        self.fulcon = Fulcon(512 * 32 * 32, 64)
        self.upconv1 = Up_Conv(64, 256, 32)

        self.trans_conv1 = Transpose_Conv(256, 256)
        self.trans_conv2 = Transpose_Conv(256, 128)
        self.trans_conv3 = Transpose_Conv(128, 64)
        self.trans_conv4 = Transpose_Conv(64, 32)
        self.outc2 = OutConv(32, 1)

        self.down1_s2 = DownD_s2(n_channels, 32)
        self.down2_s2 = DownD_s2(32, 64)
        self.down3_s2 = DownT_s2(64, 128)
        self.down4_s2 = DownT_s2(128, 256)
        self.conv_s2 = TripleConv_s2(256, 512)
        self.up1_s2 = Up_s2(512, 256)
        self.up2_s2 = Up_s2(256, 128)
        self.up3_s2 = Up_s2(128, 64)
        self.up4_s2 = Up_s2(64, 32)
        self.outc_s2 = OutConv_s2(32, n_classes, True)

    def forward(self, x):
        ct = x
        xconv, x1 = self.down1(x)
        x1conv, x2 = self.down2(x1)
        x2conv, x3 = self.down3(x2)
        x3conv, x4 = self.down4(x3)
        x5 = self.endconv(x4)
        # x_eca = self.ECAblock(x5)

        # fulconnect
        y = self.fulcon(x5)
        y = y.view(-1, 64, 1, 1)
        y = self.upconv1(y)

        y = self.trans_conv1(y, x3conv)
        x = self.up1(x5, x3conv)
        y = self.trans_conv2(y, x2conv)
        x = self.up2(x, y)
        y = self.trans_conv3(y, x1conv)
        x = self.up3(x, y)
        y = self.trans_conv4(y, xconv)
        x = self.up4(x, y)
        logits = self.outc1(x)
        logits2 = self.outc2(y)

        logits = F.interpolate(logits, scale_factor=(1.2, 1.2), mode='bilinear')  # 上采样，裁切，点乘
        logits = logits[:, :, 51:563, 51:563]
        logits = torch.mul(logits, ct)


        x0conv_s2, x1_s2 = self.down1_s2(logits)
        x1conv_s2, x2_s2 = self.down2_s2(x1_s2)
        x2conv_s2, x3_s2 = self.down3_s2(x2_s2)
        x3conv_s2, x4_s2 = self.down4_s2(x3_s2)
        x5_s2 = self.conv_s2(x4_s2)
        y = self.up1_s2(x5_s2, x3conv_s2)
        y = self.up2_s2(y, x2conv_s2)
        y = self.up3_s2(y, x1conv_s2)
        y = self.up4_s2(y, x0conv_s2)
        y = self.outc_s2(y)

        return y, logits2
