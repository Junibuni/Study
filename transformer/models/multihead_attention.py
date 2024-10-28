import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        pass

    def forward(self, query, key, value, mask=None):
        pass
