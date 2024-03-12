from functools import reduce

import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, in_c, in_shape, num_sb=4, bb_out=128, znum=16, concat=True):
        super(Encoder, self).__init__()
        # in_shape: list/tuple in BCHW format
        repeat_num = int(np.log2(np.max(in_shape[2:]))) - 2
        assert(repeat_num > 0 and np.sum([i % np.power(2, repeat_num-1) for i in in_shape[2:]]) == 0)

        self.first_conv = nn.Conv2d(in_c, 128, kernel_size=3, stride=1, padding=1, bias=True) # match channel to 128
        self.big_blocks = nn.Sequential()

        self.big_blocks.add_module("big_block_0", BigBlock(128, bb_out, num_small_block=num_sb))
        for i in range(1, repeat_num):
            self.big_blocks.add_module(f"big_block_{i}", BigBlock(bb_out, bb_out, num_small_block=num_sb, type="encoder")) # in(in_c) out(128)

        divisor = 2**repeat_num
        result = int(reduce(lambda x, y: x * (y / divisor), in_shape[2:], 1)) * bb_out
        print("result: ", result)
        self.linear = nn.Linear(result, znum)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.big_blocks(x)

        # flatten(reshape)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        return self.linear(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        pass

    def forward(self, x):
        pass

class IntegrationNet(nn.Module):
    def __init__(self):
        super(IntegrationNet, self).__init__()
        pass

    def forward(self, x):
        pass

class SmallBlock(nn.Module):
    def __init__(self, in_c, out_c=128):
        super(SmallBlock, self).__init__()
        self.up_conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True)
        self.l_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.up_conv(x)
        return self.l_relu(x)
    
class BigBlock(nn.Module):
    """
    if concat, automaticall out_c = in
    """
    def __init__(self, in_c, out_c, num_small_block=4, concat=True, type="encoder"):
        super(BigBlock, self).__init__()
        assert type in ["encoder", "decoder"], f"{type} not defined for Big Block"
        self.concat = concat
        self.small_blocks = nn.Sequential()
        
        self.small_blocks.add_module("small_block_0", SmallBlock(in_c, 128))
        for i in range(1, num_small_block):
            self.small_blocks.add_module(f"small_block_{i}", SmallBlock(128, 128)) # in(in_c) out(128)
        
        # WARNING: input_size / 2^N != int, size mismatch
        # in_c*2 by concat
        match type:
            case "encoder": #conv
                self.pool = nn.Conv2d(128+in_c, out_c, kernel_size=3, stride=2, padding=1, bias=True)
            case "decoder": #upconv
                self.pool = nn.ConvTranspose2d(128+in_c, out_c, kernel_size=4, stride=2, padding=1, bias=True)
    
    def forward(self, x):
        x0 = x
        x = self.small_blocks(x)
        #print(x0.shape, x.shape)
        x = torch.concat([x, x0], dim=1)

        x = self.pool(x)
        return x


        
        
if __name__ == "__main__":
    input_size = (4, 1, 768, 512)
    input = torch.randn(input_size)

    encoder = Encoder(1, input.shape)
    print(encoder)
    output = encoder(input)

    print(output.shape)
    print()
    print(output)