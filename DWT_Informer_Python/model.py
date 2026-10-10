
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_encoding(length: int, dim: int, device, dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=dtype)[:, None]
    rates = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype)
                      * (-math.log(10000) / dim))
    emb = torch.zeros(1, length, dim, device=device, dtype=dtype)
    emb[0, :, 0::2] = torch.sin(pos * rates)
    emb[0, :, 1::2] = torch.cos(pos * rates[:emb[..., 1::2].shape[-1]])
    return emb


class ProbSparseAttention(nn.Module):
    """Informer-inspired sampled-key active-query attention (causal or full).

    Unselected queries use uniform global context (encoder) or prefix context
    (causal decoder); active queries attend over all permissible keys.
    """
    def __init__(self, d_model: int, heads: int, factor: int = 5,
                 dropout: float = 0.05, causal: bool = False):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads, self.d_head, self.factor, self.causal = (
            heads, d_model // heads, factor, causal)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(z):
            return z.view(b, length, self.heads, self.d_head).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        size_sample = min(length, max(1, int(self.factor * math.ceil(math.log(length + 1)))))
        top_n = min(length, max(1, int(self.factor * math.ceil(math.log(length + 1)))))
        sample_idx = torch.randint(length, (length, size_sample), device=x.device)
        sampled_k = k[:, :, sample_idx, :]  # [B,H,L,Ks,D]
        sparse_scores = (q.unsqueeze(-2) * sampled_k).sum(-1) / math.sqrt(self.d_head)
        sparsity = sparse_scores.max(-1).values - sparse_scores.mean(-1)
        selected_idx = sparsity.topk(top_n, dim=-1).indices  # [B,H,top]
        selected_q = q.gather(2, selected_idx[..., None].expand(-1,-1,-1,self.d_head))
        attn = torch.matmul(selected_q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.causal:
            positions = torch.arange(length, device=x.device)[None,None,None,:]
            causal_mask = positions > selected_idx[..., None]
            attn = attn.masked_fill(causal_mask, torch.finfo(attn.dtype).min)
            context = v.cumsum(dim=2) / torch.arange(1, length + 1, device=x.device,
                                                     dtype=v.dtype)[None,None,:,None]
        else:
            context = v.mean(dim=2, keepdim=True).expand_as(v).clone()
        weights = self.dropout(torch.softmax(attn, dim=-1))
        selected_context = torch.matmul(weights, v)
        context = context.scatter(2, selected_idx[...,None].expand(-1,-1,-1,self.d_head),
                                  selected_context)
        return self.out(context.transpose(1,2).contiguous().view(b,length,dim))


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, factor, dropout):
        super().__init__()
        self.attn = ProbSparseAttention(dim, heads, factor, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4*dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        return self.norm2(x + self.drop(self.ffn(x)))


class Distillation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, padding_mode='circular')
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(F.elu(self.conv(x.transpose(1,2)))).transpose(1,2)


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, factor, dropout):
        super().__init__()
        self.self_attn = ProbSparseAttention(dim, heads, factor, dropout, causal=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n1, self.n2, self.n3 = (nn.LayerNorm(dim) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, encoded):
        x = self.n1(x + self.drop(self.self_attn(x)))
        cross, _ = self.cross_attn(x, encoded, encoded, need_weights=False)
        x = self.n2(x + self.drop(cross))
        return self.n3(x + self.drop(self.ffn(x)))


class DWTInformer(nn.Module):
    def __init__(self, stations: int, input_channels: int, horizon: int,
                 d_model: int = 512, heads: int = 8, enc_layers: int = 3,
                 dec_layers: int = 2, factor: int = 5, dropout: float = 0.05,
                 station_dim: int = 32, with_geo: bool = True,
                 with_station_attention: bool = True):
        super().__init__()
        self.stations, self.horizon, self.dim = stations, horizon, d_model
        self.with_geo = with_geo
        self.station_projection = nn.Linear(input_channels, station_dim)
        self.geo_projection = nn.Linear(3, station_dim)
        self.with_station_attention = with_station_attention
        if with_station_attention:
            self.spatial_attention = nn.MultiheadAttention(station_dim, 4, batch_first=True)
            self.spatial_norm = nn.LayerNorm(station_dim)
        self.input_projection = nn.Linear(stations * station_dim, d_model)
        self.future_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.enc = nn.ModuleList([EncoderBlock(d_model, heads, factor, dropout)
                                  for _ in range(enc_layers)])
        self.distill = nn.ModuleList([Distillation(d_model)
                                      for _ in range(max(0, enc_layers-1))])
        self.dec = nn.ModuleList([DecoderBlock(d_model, heads, factor, dropout)
                                  for _ in range(dec_layers)])
        self.readout = nn.Linear(d_model, stations)

    def forward(self, x: torch.Tensor, geo: torch.Tensor):
        """x [B,L,stations,channels], geo [stations,3], output [B,horizon,stations] mm."""
        if x.ndim != 4 or x.shape[2] != self.stations:
            raise ValueError("x must be [B,L,number_of_stations,number_of_features]")
        if geo.shape != (self.stations, 3):
            raise ValueError("geo must be [number_of_stations,3]")
        b, length, _, _ = x.shape
        station_hidden = self.station_projection(x)
        if self.with_geo:
            station_hidden = station_hidden + self.geo_projection(geo)[None,None,:,:]
        if self.with_station_attention:
            z = station_hidden.reshape(b*length, self.stations, -1)
            az, _ = self.spatial_attention(z,z,z,need_weights=False)
            station_hidden = self.spatial_norm(z + az).reshape(b,length,self.stations,-1)
        hidden = self.input_projection(station_hidden.reshape(b,length,-1))
        hidden = hidden + sinusoidal_encoding(length,self.dim,hidden.device,hidden.dtype)
        # Decoder warm-up only uses known encoder history, NEVER target precipitation.
        warm = max(1,length//2)
        start = hidden[:, -warm:, :]
        future = self.future_token.expand(b,self.horizon,-1)
        dec = torch.cat((start, future),dim=1)
        dec = dec + sinusoidal_encoding(dec.shape[1],self.dim,dec.device,dec.dtype)
        encoded = hidden
        for i,layer in enumerate(self.enc):
            encoded = layer(encoded)
            if i < len(self.distill):
                encoded = self.distill[i](encoded)
        for layer in self.dec:
            dec = layer(dec,encoded)
        # Rainfall cannot be negative. The nonnegativity constraint is an
        # implementation option; authors did not specify the output activation.
        return F.softplus(self.readout(dec[:, -self.horizon:, :]))
