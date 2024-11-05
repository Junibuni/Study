import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_encoder_layers, num_decoder_layers, ffn_hidden_dim, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_encoder_layers, embedding_dim, num_heads, ffn_hidden_dim, dropout_rate)
        self.decoder = Decoder(num_decoder_layers, embedding_dim, num_heads, ffn_hidden_dim, dropout_rate)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return output
