import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None, kernel_size=3):
        super(DoubleConv, self).__init__()
        if not mid_channel:
            mid_channel = out_channel

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, out_channel, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    # downsample, doubleconv
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.up_double_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )
    
    def forward(self, x):
        return self.up_double_conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channel, out_channel, concat_channel):
        super(Up, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2) # double channel
        self.double_conv = DoubleConv(out_channel+concat_channel, out_channel)

    def forward(self, x, in_feature):
        x = self.up_conv(x)

        # TODO: Need padding/cropping in concat?
        x = torch.cat([in_feature, x], dim=1)
        return self.double_conv(x)
    
class FinalConv(nn.Module):
    def __init__(self, in_channel, num_classes=5):
        super(FinalConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)