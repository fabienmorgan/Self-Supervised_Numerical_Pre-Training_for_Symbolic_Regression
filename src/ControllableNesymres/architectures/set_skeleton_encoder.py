import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class SetSkeletonEncoder(pl.LightningModule):
    def __init__(self,cfg, attn_mask):
        super().__init__()
        self.masked_attention1 = nn.MultiheadAttention(cfg.dim_hidden, cfg.num_heads, attn_mask=attn_mask)
        self.norm1 = nn.LayerNorm(cfg.dim_hidden)
        self.masked_attention2 = nn.MultiheadAttention(cfg.dim_hidden, cfg.num_heads, attn_mask=attn_mask)
        self.activation = getattr(F, cfg.activation)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.dim_hidden, cfg.dim_hidden),
            self.activation,
            nn.Linear(cfg.dim_hidden, cfg.dim_hidden)
        )
        self.norm2 = nn.LayerNorm(cfg.dim_hidden)        

def forward(self, x):
        x_res1 = x
        x = self.masked_attention1(x, x, x, need_weights = False)
        x += x_res1
        x = self.norm1(x)
        x_res2 = x
        x = self.masked_attention2(x, x, x, need_weights = False)
        x = self.ffn(x)
        x += x_res2
        x = self.norm2(x)
        return x