import torch.nn as nn
from models.encoder import EncoderLayer
from models.decoder import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_encoder_layers, num_decoder_layers, ffn_hidden_dim, dropout_rate=0.1):
        super(Transformer, self).__init__()
        pass

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        pass
