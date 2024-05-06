import torch.nn as nn
import pytorch_lightning as pl

class SetSkeletonEncoderLayer(pl.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self.multihead_attention1 = nn.MultiheadAttention(cfg.dim_hidden, cfg.num_heads)
        self.norm1 = nn.LayerNorm(cfg.dim_hidden)
        self.multihead_attention2 = nn.MultiheadAttention(cfg.dim_hidden, cfg.num_heads)
        self.activation = getattr(nn, cfg.activation)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.dim_hidden, cfg.dim_hidden),
            self.activation(),
            nn.Linear(cfg.dim_hidden, cfg.dim_hidden)
        )
        self.norm2 = nn.LayerNorm(cfg.dim_hidden)        

    def forward(self, x, attention_mask = None):
        x_res1 = x
        x, _ = self.multihead_attention1(x, x, x, attn_mask=attention_mask, need_weights = False)
        x = self.norm1(x + x_res1)
        x_res2 = x
        x, _ = self.multihead_attention2(x, x, x, attn_mask=attention_mask, need_weights = False)
        x = self.ffn(x)
        x = self.norm2(x + x_res2)
        return x
    
class SetSkeletonEncoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([SetSkeletonEncoderLayer(cfg) for _ in range(cfg.skel_enc_layers)])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x