import torch.nn as nn
from models.multihead_attention import MultiHeadAttention
from models.feed_forward import FeedForwardNetwork

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_hidden_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForwardNetwork(embedding_dim, ffn_hidden_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_output = self.self_attention(x, x, x, mask)
        # Add and norm
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        # Add and norm
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, num_heads, ffn_hidden_dim, dropout_rate=0.1):
        super(Encoder, self).__init__()
        # Repeat for num_layers
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, ffn_hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
