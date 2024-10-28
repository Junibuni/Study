import torch.nn as nn
from models.multihead_attention import MultiHeadAttention
from models.feed_forward import FeedForwardNetwork

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_hidden_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        pass

    def forward(self, x, mask=None):
        pass
