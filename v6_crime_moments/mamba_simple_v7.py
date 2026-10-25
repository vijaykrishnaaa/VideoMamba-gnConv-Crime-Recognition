# Copyright (c) 2023, Tri Dao, Albert Gu.
# Modified V7: HorNet gnConv2D (order=5, multi-scale kernels, GhostModule)
# Paper: https://arxiv.org/abs/2207.14284
#
# V7 = the guide's V6 gnConv2D, UNCHANGED inside, with two adapter bug fixes
# and one option:
#   FIX 1 (frame reshape): V6 did  x.reshape(B*T, C, H, W)  on a (B, C, T*H*W)
#         tensor. That does NOT split frames: it regroups (channel, time) pairs,
#         so each "frame" seen by gnConv held 96 channels x 8 time steps.
#         V7 does reshape(B, C, T, H, W) -> permute -> (B*T, C, H, W), so gnConv
#         now really runs on each frame's 14x14 patch grid with all 768 channels.
#   FIX 2 (cls position): the backward scan (gnconv_b) receives the FLIPPED
#         sequence, where the cls token is LAST, but V6 always stripped index 0.
#         V7 strips index 0 for the forward scan and index -1 for the backward scan.
#   OPTION (gnconv_from_layer): layers below this index are built WITHOUT gnConv.
#         In V1-V3 the gnConv of frozen layers 0-15 was zero-initialised and
#         frozen, so it always output exactly 0. Removing it gives identical
#         predictions with ~181M fewer parameters and much less compute.
#
# This file is loaded only by v4_crime_fixed/train_crime_v9.py. The original
# VideoMamba/mamba/mamba_ssm/modules/mamba_simple.py is left untouched so
# V1-V3 results stay reproducible.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


# ============================================================
# GhostModule — Lightweight Feature Projection (from GhostNet)
# (unchanged from V6)
# ============================================================
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# ============================================================
# Helper: Depthwise Conv2d (unchanged from V6)
# ============================================================
def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2,
                     bias=bias, groups=dim)


# ============================================================
# HorNet gnConv2D — Recursive Gated Convolution
# __init__ and the gating math are the guide's V6 code, unchanged.
# Only the 1D <-> 2D adapter in forward() is fixed.
# ============================================================
class gnConv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=14, s=1.0, bias=True):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.scale = s

        # Spatial grid size: 14x14 for 224x224 input with patch_size=16
        self.h = h
        self.w = w

        # Input projection: GhostModule
        self.proj_in = GhostModule(dim, 2 * dim, 1)

        # Depthwise convolution for spatial context gathering
        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        # Output projection (zero-initialised for safe injection)
        self.proj_out = nn.Conv2d(dim, dim, 1)

        # Multi-scale convolutions for recursive gating: kernels [3, 5, 7, 1, ...]
        kernelsize_my = [3, 5, 7, 1, 3, 5, 7, 1]
        pad_my = [1, 2, 3, 0, 1, 2, 3, 0]

        self.pws = nn.ModuleList([
            nn.Conv2d(self.dims[i], self.dims[i + 1], stride=1,
                      kernel_size=kernelsize_my[i],
                      padding=(pad_my[i], pad_my[i]))
            for i in range(order - 1)
        ])

        # Zero initialisation: the module outputs 0 at the start of training,
        # so the pretrained backbone is not disturbed.
        nn.init.constant_(self.proj_out.weight, 0)
        if self.proj_out.bias is not None:
            nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x, cls_first=True):
        """
        x: (B, C, L) from the Mamba branch, L = T*H*W (+1 if a cls token is present).
        cls_first=True  -> forward scan, cls token (if any) is at index 0.
        cls_first=False -> backward scan (flipped sequence), cls token is at index -1.
        Patches are processed per frame on the H x W grid; the cls position gets 0.
        """
        B, C, L = x.shape
        H, W = self.h, self.w
        spatial_tokens = H * W

        extra = L % spatial_tokens
        if extra not in (0, 1):
            raise ValueError(f"gnConv: sequence length {L} is not T*{spatial_tokens} (+1 cls)")
        has_cls = extra == 1

        if has_cls:
            x_patches = x[:, :, 1:] if cls_first else x[:, :, :-1]
        else:
            x_patches = x
        L_patches = x_patches.shape[2]
        T = L_patches // spatial_tokens

        # FIX 1: (B, C, T*H*W) -> (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W)
        x_2d = (x_patches.reshape(B, C, T, H, W)
                .permute(0, 2, 1, 3, 4)
                .reshape(B * T, C, H, W))

        # === 2D gnConv processing (guide's code, unchanged) ===
        fused_x = self.proj_in(x_2d)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)

        x_2d = pwa * dw_list[0]
        for i in range(self.order - 1):
            x_2d = self.pws[i](x_2d) * dw_list[i + 1]

        x_2d = self.proj_out(x_2d)

        # Inverse of FIX 1: (B*T, C, H, W) -> (B, T, C, H, W) -> (B, C, T, H, W) -> (B, C, T*H*W)
        x_patches_out = (x_2d.reshape(B, T, C, H, W)
                         .permute(0, 2, 1, 3, 4)
                         .reshape(B, C, L_patches))

        if not has_cls:
            return x_patches_out
        # FIX 2: put the (zero) cls contribution back at the position it came from
        x_cls_zero = x_patches_out.new_zeros(B, C, 1)
        if cls_first:
            return torch.cat([x_cls_zero, x_patches_out], dim=2)
        return torch.cat([x_patches_out, x_cls_zero], dim=2)


# ============================================================
# Mamba block with residual gnConv2D injection (V7)
# ============================================================
class Mamba(nn.Module):
    def __init__(
        self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
        dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
        dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True,
        layer_idx=None, device=None, dtype=None, bimamba=True,
        gnconv_from_layer=0,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError("mamba_ssm selective_scan_fn could not be imported "
                              "(is ~/vlil_40/VideoMamba/mamba on sys.path and videomamba_env active?)")
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.bimamba = bimamba
        self.use_fast_path = False
        use_gnconv = layer_idx is None or layer_idx >= gnconv_from_layer

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Original Conv1d (keeps pretrained K400 / UCF-101 weights)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            bias=conv_bias, kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, **factory_kwargs,
        )
        # HorNet gnConv2D (order=5, multi-scale, GhostModule) in parallel with Conv1d
        self.gnconv = (gnConv(dim=self.d_inner, order=5, h=14, w=14, s=1.0, bias=conv_bias)
                       if use_gnconv else None)

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        if self.bimamba:
            A_b = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                         "n -> d n", d=self.d_inner).contiguous()
            self.A_b_log = nn.Parameter(torch.log(A_b))
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner, out_channels=self.d_inner,
                bias=conv_bias, kernel_size=d_conv, groups=self.d_inner,
                padding=d_conv - 1, **factory_kwargs,
            )
            self.gnconv_b = (gnConv(dim=self.d_inner, order=5, h=14, w=14, s=1.0, bias=conv_bias)
                             if use_gnconv else None)

            self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, T=1):
        batch, seqlen, dim = hidden_states.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)

        # FORWARD DIRECTION: Conv1d + gnConv2D (cls token at index 0)
        conv_out = self.conv1d(x)[..., :seqlen]
        if self.gnconv is not None:
            conv_out = conv_out + self.gnconv(x, cls_first=True)[..., :seqlen]
        x_conv = self.act(conv_out)

        x_dbl = self.x_proj(rearrange(x_conv, "b d l -> (b l) d"))
        dt, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B_mat = rearrange(B_mat, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C_mat = rearrange(C_mat, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x_conv, dt, A, B_mat, C_mat, self.D.float(), z=z,
            delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=False,
        )

        if self.bimamba:
            A_b = -torch.exp(self.A_b_log.float())
            xz_flip = xz.flip([-1])
            x_b, z_b = xz_flip.chunk(2, dim=1)

            # BACKWARD DIRECTION: Conv1d + gnConv2D (flipped, cls token at index -1)
            conv_out_b = self.conv1d_b(x_b)[..., :seqlen]
            if self.gnconv_b is not None:
                conv_out_b = conv_out_b + self.gnconv_b(x_b, cls_first=False)[..., :seqlen]
            x_conv_b = self.act(conv_out_b)

            x_dbl_b = self.x_proj_b(rearrange(x_conv_b, "b d l -> (b l) d"))
            dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_b = self.dt_proj_b.weight @ dt_b.t()
            dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
            B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            y_b = selective_scan_fn(
                x_conv_b, dt_b, A_b, B_b, C_b, self.D_b.float(), z=z_b,
                delta_bias=self.dt_proj_b.bias.float(), delta_softplus=True,
            )

            out = F.linear(
                rearrange(y + y_b.flip([-1]), "b d l -> b l d"),
                self.out_proj.weight, self.out_proj.bias
            )
        else:
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        raise NotImplementedError("gnConv2D does not support step-based decoding.")

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype)
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype)
        return conv_state, ssm_state
