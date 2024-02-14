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
            nn.BatchNorm2d(),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    # downsample, doubleconv
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()