import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU()
        )