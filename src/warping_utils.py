import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F

class upBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(upBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        #self.norm = nn.BatchNorm2d(out_channels, affine=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
    def forward(self, x):
        
        x = self.conv(x)
        x = F.relu(x)
        x = self.upsample(x)
        
        return x
    
    
class downBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(downBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        #self.norm = nn.BatchNorm2d(out_channels, affine=True)
        self.maxpool = nn.MaxPool2d((2,2))
        
    def forward(self, x):
        
        x = self.conv(x)
        #x = self.norm(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        return x

class sameBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(sameBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
    def forward(self, x):
        
        x = self.conv(x)
        x = F.relu(x)
        
        return x

class Encoder(nn.Module):
    
    def __init__(self, in_channels, num_blocks=4, block_expansion=128, max_channels=1024):
        super(Encoder, self).__init__()
        
        downblocks = []
        for i in range(num_blocks):
            downblocks.append(downBlock(in_channels if i==0 else min(max_channels, block_expansion * (2 ** i)),
                                        min(max_channels, block_expansion * (2 ** (i+1)))))
            
        self.downblocks = nn.ModuleList(downblocks)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            
            nn.init.constant_(module.weight, 0)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        outs = [x]
        for down_block in self.downblocks:
            outs.append(down_block(outs[-1]))
        return outs
    
    
    
class Decoder(nn.Module):
    
    def __init__(self, in_channels, num_blocks=5, block_expansion=64, max_features=1024):
        super(Decoder, self).__init__()
        
        upblocks = []
        for i in range(num_blocks)[::-1]:
            
            out_filters = min(max_features, block_expansion * (2 ** i)) if i>0 else 256
            if i==num_blocks-1:
                in_filters = min(max_features, block_expansion * (2 ** (i + 1)))
                upblocks.append(upBlock(in_filters, out_filters))
            elif i==0:
                in_filters = min(max_features, block_expansion * (2 ** (i + 1))) + in_channels
                upblocks.append(sameBlock(in_filters, out_filters))
            else:
                in_filters = 2*min(max_features, block_expansion * (2 ** (i + 1)))
                upblocks.append(upBlock(in_filters, out_filters))
            
            #[4: 1024X1024; 3:2048X512; 2:1024X256; 1:512X128; 0:148X256]
            #upblocks.append(upBlock(in_filters, out_filters))
            
        self.up_blocks = nn.ModuleList(upblocks)
        self.out_filters = block_expansion + in_channels
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            
            nn.init.constant_(module.weight, 0)
            nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        
        out = x.pop()

        for i, up_block in enumerate(self.up_blocks):
            
            if i==0:
                out = up_block(out)
                
            else:
                skip = x.pop()
                out = torch.cat([out, skip], dim=1)
                out = up_block(out)
                   
        return out
    

class q_unet(nn.Module):
    
    def __init__(self, kp=3):
        super(q_unet, self).__init__()
        
        self.kp = kp
        
        self.encoder_q = Encoder(self.kp)
        self.decoder_q = Decoder(self.kp)
        
        
    def forward(self, x):
        
        query = self.decoder_q(self.encoder_q(x))
        return query
    
    
class k_unet(nn.Module):
    
    def __init__(self, kp=6):
        super(k_unet, self).__init__()
        
        self.kp = kp
        
        self.encoder_k = Encoder(self.kp)
        self.decoder_k = Decoder(self.kp)
        
    def forward(self, x):
        
        key = self.decoder_k(self.encoder_k(x))
        return key
    
    
class cross_attention_module(nn.Module):
    
    def __init__(self, d_embed=256, in_proj_bias=True, out_proj_bias=True):
        super(cross_attention_module, self).__init__()
        
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        
        self.out_proj = nn.Linear(4, 4, bias=out_proj_bias)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            
            nn.init.constant_(module.weight, 0)
            nn.init.constant_(module.bias, 0)
        
    def forward(self, q, k, v):
        
        input_shape = q.shape
        
        batch_size, sequence_length, d_embed = input_shape
        
        q = self.q_proj(q) #B, 4096, 256
        k = self.k_proj(k) #B, 4096, 256
        
        
        weight = q @ k.transpose(-1, -2) #B, seqlen_Q, seqlen_KV
        weight /= math.sqrt(d_embed)
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v #B, 4096,4        
        output = self.out_proj(output)
        
        output = output.transpose(1, 2).contiguous()
        
        return output
        
        
        
        
        
        
        
        
        
#def kp2heatmap():
    
    

        
        
        
        
        
        
        
        
        