"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from ...core import register

__all__ = ['HybridEncoder']


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride,
            padding=(kernel_size-1)//2 if padding is None else padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act  = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in  = ch_in
        self.ch_out = ch_out
        self.conv1  = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2  = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act    = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        y = self.conv(x) if hasattr(self, 'conv') else self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
        k, b = self.get_equivalent_kernel_bias()
        self.conv.weight.data = k
        self.conv.bias.data   = b

    def get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_bn_tensor(self.conv1)
        k1, b1 = self._fuse_bn_tensor(self.conv2)
        return k3 + self._pad_1x1_to_3x3_tensor(k1), b3 + b1

    def _pad_1x1_to_3x3_tensor(self, k):
        return 0 if k is None else F.pad(k, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None: return 0, 0
        k = branch.conv.weight
        mu, var = branch.norm.running_mean, branch.norm.running_var
        g, b, eps = branch.norm.weight, branch.norm.bias, branch.norm.eps
        std = (var + eps).sqrt()
        t = (g / std).reshape(-1, 1, 1, 1)
        return k * t, b - mu * g / std


class CSPRepLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3,
                 expansion=1.0, bias=None, act="silu"):
        super().__init__()
        hid = int(out_channels * expansion)
        self.conv1       = ConvNormLayer(in_channels, hid, 1, 1, bias=bias, act=act)
        self.conv2       = ConvNormLayer(in_channels, hid, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[RepVggBlock(hid, hid, act=act) for _ in range(num_blocks)])
        self.conv3       = ConvNormLayer(hid, out_channels, 1, 1, bias=bias, act=act) if hid != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.dropout    = nn.Dropout(dropout)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(t, pe): return t if pe is None else t + pe

    def forward(self, src, src_mask=None, pos_embed=None):
        res = src
        if self.normalize_before: src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = res + self.dropout1(src)
        if not self.normalize_before: src = self.norm1(src)
        res = src
        if self.normalize_before: src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = res + self.dropout2(src)
        if not self.normalize_before: src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers     = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm       = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        out = src
        for l in self.layers:
            out = l(out, src_mask=src_mask, pos_embed=pos_embed)
        return self.norm(out) if self.norm else out


class TemporalFusion(nn.Module):
    """Active during training only when use_temporal=True."""
    def __init__(self, channels, act='silu'):
        super().__init__()
        self.compress = ConvNormLayer(channels * 2, channels, 1, 1, act=act)
        self.refine   = ConvNormLayer(channels,     channels, 3, 1, act=act)

    def forward(self, current, previous):
        return self.refine(self.compress(torch.cat([current, previous], dim=1))) + current


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 use_temporal=True,
                 version='v2'):
        super().__init__()
        self.in_channels        = in_channels
        self.feat_strides       = feat_strides
        self.hidden_dim         = hidden_dim
        self.use_encoder_idx    = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature     = pe_temperature
        self.eval_spatial_size  = eval_spatial_size
        self.use_temporal       = use_temporal
        self.out_channels       = [hidden_dim] * len(in_channels)
        self.out_strides        = feat_strides

        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            self.input_proj.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(ch, hidden_dim, 1, bias=False)),
                ('norm', nn.BatchNorm2d(hidden_dim))
            ])))

        el = TransformerEncoderLayer(hidden_dim, nhead=nhead,
                                     dim_feedforward=dim_feedforward,
                                     dropout=dropout, activation=enc_act)
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(el), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        if use_temporal:
            self.temporal_fusion  = nn.ModuleList([
                TemporalFusion(hidden_dim, act=act) for _ in range(len(use_encoder_idx))
            ])
            self._prev_feat_cache = [None] * len(use_encoder_idx)

        self.lateral_convs    = nn.ModuleList()
        self.fpn_blocks       = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult),
                            act=act, expansion=expansion))

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks       = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult),
                            act=act, expansion=expansion))

        self._reset_parameters()

    def reset_temporal_cache(self):
        if self.use_temporal:
            self._prev_feat_cache = [None] * len(self.use_encoder_idx)

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pe = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pe)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        gw, gh = torch.meshgrid(torch.arange(int(w), dtype=torch.float32),
                                torch.arange(int(h), dtype=torch.float32), indexing='ij')
        pd  = embed_dim // 4
        om  = 1. / (temperature ** (torch.arange(pd, dtype=torch.float32) / pd))
        ow  = gw.flatten()[..., None] @ om[None]
        oh  = gh.flatten()[..., None] @ om[None]
        return torch.concat([ow.sin(), ow.cos(), oh.sin(), oh.cos()], dim=1)[None]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj = [self.input_proj[i](f) for i, f in enumerate(feats)]

        if self.use_temporal and self.training:
            self._prev_feat_cache = [None] * len(self.use_encoder_idx)

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj[enc_ind].shape[2:]
                src  = proj[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pe = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src.device)
                else:
                    pe = getattr(self, f'pos_embed{enc_ind}', None)
                    # Safety check: precomputed PE may not match if input size
                    # differs from eval_spatial_size (e.g. teacher in eval()
                    # during multi-scale distillation training).
                    if pe is None or pe.shape[1] != h * w:
                        pe = self.build_2d_sincos_position_embedding(
                            w, h, self.hidden_dim, self.pe_temperature)
                    pe = pe.to(src.device)
                mem = self.encoder[i](src, pos_embed=pe)
                cur = mem.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

                if self.use_temporal and self.training:
                    prev = self._prev_feat_cache[i]
                    if prev is None: prev = torch.zeros_like(cur)
                    elif prev.shape != cur.shape:
                        prev = F.interpolate(prev, size=(h, w), mode='bilinear', align_corners=False)
                    fused = self.temporal_fusion[i](cur, prev)
                    self._prev_feat_cache[i] = cur.detach()
                    proj[enc_ind] = fused
                else:
                    proj[enc_ind] = cur

        inner = [proj[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            fh = self.lateral_convs[len(self.in_channels) - 1 - idx](inner[0])
            inner[0] = fh
            up  = F.interpolate(fh, scale_factor=2., mode='nearest')
            out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([up, proj[idx - 1]], dim=1))
            inner.insert(0, out)

        outs = [inner[0]]
        for idx in range(len(self.in_channels) - 1):
            ds  = self.downsample_convs[idx](outs[-1])
            out = self.pan_blocks[idx](torch.concat([ds, inner[idx + 1]], dim=1))
            outs.append(out)

        return outs
