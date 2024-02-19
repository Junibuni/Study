import numpy as np
import torch
import torch.nn as nn
import torchvision

from unet.backbone import get_backbone

from torchinfo import summary
model = get_backbone("efficientnetb0")

#print(summary(efficientnet_b0, input_size=(1, 3, 224, 224)))