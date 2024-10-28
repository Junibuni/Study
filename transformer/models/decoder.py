import torch.nn as nn
from models.multihead_attention import MultiHeadAttention
from models.feed_forward import FeedForwardNetwork

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_hidden_dim, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        pass

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        pass
