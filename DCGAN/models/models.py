import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        """
        Args:
            nc = number of channels
            nz = generator input dim
            ngf = input channel size
        """
        super().__init__()
        self.layer1 = nn.Sequential(

        )
        self.layer2 = nn.Sequential(

        )
        self.layer3 = nn.Sequential(

        )
        self.layer4 = nn.Sequential(

        )
        self.last = nn.Sequential(

        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)
        return out
