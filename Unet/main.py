import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchinfo import summary


a = torch.rand((1, 3, 244, 244))
print(a.shape)
print(a.dim())