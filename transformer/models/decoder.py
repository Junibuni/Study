import torch.nn as nn
from models.multihead_attention import MultiHeadAttention
from models.feed_forward import FeedForwardNetwork

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_hidden_dim, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForwardNetwork(embedding_dim, ffn_hidden_dim, dropout_rate)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, num_heads, ffn_hidden_dim, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, ffn_hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
