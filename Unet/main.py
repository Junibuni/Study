import numpy as np
import torch
import torch.nn as nn
import torchvision

from unet.backbone import get_backbone

vgg19_model = get_backbone("vgg19")
print(vgg19_model)

def compute_output_size(input_size, layer):
    input_tensor = torch.rand(1, *input_size)
    output_tensor = layer(input_tensor)
    return output_tensor.size()[1:]

# Iterate through the layers and print out the expected output size
input_size = (3, 224, 224)  # Input size for VGG
for name, layer in vgg19_model.named_children():
    output_size = compute_output_size(input_size, layer)
    print(f"Layer: {name}, Output Size: {output_size}")
    input_size = output_size