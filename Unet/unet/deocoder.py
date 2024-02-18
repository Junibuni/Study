import torch
import torch.nn as nn

from . import network

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, skip_layers, num_classes=5):
        super(UNetDecoder, self).__init__()
        self.skip_layers = skip_layers

        self.module1 = network.Up(in_channels, in_channels/2)
        self.module2 = network.Up(in_channels/2, in_channels/4)
        self.module3 = network.Up(in_channels/4, in_channels/8)
        self.module4 = network.Up(in_channels/8, in_channels/16)
        self.final_module = network.Up(in_channels/16, num_classes)

    def forward(self, x):
        pass

    def get_blocks_for_concat(self):
        pass