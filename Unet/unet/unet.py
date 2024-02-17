# 백본 이랑 유넷을 연결시키는 다연결시켜서 full network
import torch
import torch.nn as nn

from . import network

class UNet(nn.Module):
    def __init__(self, in_channels=3, classes=5):
        pass
