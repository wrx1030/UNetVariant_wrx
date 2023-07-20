
from .model_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down1 = Down(n_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.conv = DoubleConv(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes, True)

    def forward(self, x):
        x0conv, x1 = self.down1(x)
        x1conv, x2 = self.down2(x1)
        x2conv, x3 = self.down3(x2)
        x3conv, x4 = self.down4(x3)
        x5 = self.conv(x4)
        y = self.up1(x5, x3conv)
        y = self.up2(y, x2conv)
        y = self.up3(y, x1conv)
        y = self.up4(y, x0conv)
        y = self.outc(y)

        return y
