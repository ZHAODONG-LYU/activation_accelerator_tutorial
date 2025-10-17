#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成 bf16 测试向量（64x768）与配套的 Y（Add/Mul），以及可选的 PyTorch 参考输出。
- X_test_tensor: torch.bfloat16, shape = (64, 768)
- Y_test_tensor: torch.bfloat16, shape = (64, 768)（仅第46–53行用于 Add/Mul 测试，其余可为0）
- 同时导出 bf16 位模式（uint16）方便 AXI 侧加载
- 可选 (--emit_ref)：生成七类算子（Softmax / LayerNorm / RMSNorm / SiLU / GELU / Add / Mul）的参考输出
"""

import os
import json
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F



# ------------------------------
# 工具函数：bf16 <-> 位模式
# ------------------------------
def nanmax(x: torch.Tensor, dim=None, keepdim=False):
    # 把 NaN 改为 -inf，再做 max；全 NaN 的切片返回 NaN
    if dim is None:
        if torch.isnan(x).all():
            return torch.tensor(float('nan'), dtype=x.dtype, device=x.device)
        xc = x.clone()
        xc[torch.isnan(xc)] = -float('inf')
        return xc.max()
    else:
        xc = x.clone()
        mask_nan = torch.isnan(xc)
        xc[mask_nan] = -float('inf')
        vals, idx = torch.max(xc, dim=dim, keepdim=keepdim)
        all_nan = mask_nan.all(dim=dim, keepdim=keepdim)
        vals = vals.masked_fill(all_nan, float('nan'))
        return vals

def nansum(x: torch.Tensor, dim=None, keepdim=False):
    xc = x.clone()
    xc[torch.isnan(xc)] = 0.0
    return torch.sum(xc, dim=dim, keepdim=keepdim)

def nanmean(x: torch.Tensor, dim=None, keepdim=False):
    mask = ~torch.isnan(x)
    xc = x.clone()
    xc[~mask] = 0.0
    s = torch.sum(xc, dim=dim, keepdim=keepdim)
    cnt = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1)
    m = s / cnt
    if dim is None:
        return m if mask.any() else torch.tensor(float('nan'), dtype=x.dtype, device=x.device)
    else:
        return m.masked_fill(cnt == 0, float('nan'))


def to_bf16_bits(t_bf16: torch.Tensor) -> np.ndarray:
    """
    将 torch.bfloat16 Tensor 的位模式导出为 numpy.uint16（行主序）。
    """
    assert t_bf16.dtype == torch.bfloat16
    # 通过 float32 过渡到 bf16 的高 16bit：PyTorch 内部已管理好
    # 直接 view 成 uint16 即可得到 bf16 的 16bit 模式
    return t_bf16.view(torch.uint16).cpu().contiguous().numpy()

def from_bf16_bits(bits_u16: np.ndarray, shape) -> torch.Tensor:
    """
    将 numpy.uint16 的 bf16 位模式恢复为 torch.bfloat16 Tensor。
    """
    t = torch.from_numpy(bits_u16.astype(np.uint16)).view(shape)
    return t.view(torch.bfloat16)

def f32_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """
    将 float32 Tensor 舍入为 bfloat16（遵循 PyTorch 的 round-to-nearest-even）。
    """
    assert x.dtype == torch.float32
    return x.to(torch.bfloat16)

def bf16_to_f32(x: torch.Tensor) -> torch.Tensor:
    """
    将 bfloat16 Tensor 提升为 float32（不会增加精度，但便于某些运算）。
    """
    assert x.dtype == torch.bfloat16
    return x.to(torch.float32)

# ------------------------------
# 数值常量（bf16）
# ------------------------------
BF16_INFO = torch.finfo(torch.bfloat16)
BF16_MAX = float(BF16_INFO.max)                # 最大正数
BF16_MIN_NORMAL = float(BF16_INFO.tiny)        # 最小正正规数（~1.1754944e-38）
# 近似构造“较大”的次正规：用比最小正规数更小的一些 float32 数，转换到 bf16 后将落入 subnormal 或被 flush
SUBNORMAL_EXAMPLE_POS = 1e-39
SUBNORMAL_EXAMPLE_NEG = -1e-39

# # 用 fp32->bf16 的精确量化结果作为常量（注意：这是正规数，不是次正规）
# _ONE_E_MINUS_30_BF16 = torch.tensor(1e-30, dtype=torch.float32).to(torch.bfloat16)
# _ONE_E_MINUS_30_BF16_NEG = (-torch.tensor(1e-30, dtype=torch.float32)).to(torch.bfloat16)

# # 方便在 fill_row 里以 float32 形式填充（随后全局会再 cast 到 bf16，位型保持一致）
# SUBNORMAL_EXAMPLE_POS = float(_ONE_E_MINUS_30_BF16.item())      # 实际是 “1e-30 的 bf16 量化值” 转回 f32
# SUBNORMAL_EXAMPLE_NEG = float(_ONE_E_MINUS_30_BF16_NEG.item())

# ------------------------------
# 行号分配的说明（写入 manifest）
# ------------------------------
def build_manifest():
    man = {}
    # 0–11: 通用/特殊
    man.update({
        0:  "all +0.0",
        1:  "all +1.0",
        2:  "all -1.0",
        3:  "all +BF16_MAX",
        4:  "all -BF16_MAX",
        5:  "all +BF16_MIN_NORMAL",
        6:  "all -BF16_MIN_NORMAL",
        7:  "all +subnormal (via 1e-40 -> bf16)",
        8:  "all -subnormal (via -1e-40 -> bf16)",
        9:  "mixed normals: [1,-1,0.5,-0.5,2,-2,0.1,-0.1,10,-10] pattern",
        10: "mixed specials: [+Inf,-Inf,NaN,+0.0,-0.0,1,-1] pattern",
        11: "alternating +0.0 / -0.0",
    })
    # 12–19: 随机分布
    man.update({
        12: "uniform[-10,10]",
        13: "uniform[-1000,1000]",
        14: "uniform[-0.1,0.1]",
        15: "uniform[0,1]",
        16: "normal(mean=0, std=5)",
        17: "uniform[-5,5] with injected +Inf / NaN",
        18: "uniform[-5,5] with injected subnormals",
        19: "uniform[0, BF16_MAX/2]",
    })
    # 20–25: Softmax 特定
    man.update({
        20: "softmax: large positives increasing (100 + 0.1*k)",
        21: "softmax: large negatives decreasing (-100 - 0.1*k)",
        22: "softmax: one large (10.0) others 0.0",
        23: "softmax: one +Inf others 0.0",
        24: "softmax: two +Inf others 0.0",
        25: "softmax: one NaN others 0.0",
    })
    # 26–31: LayerNorm
    man.update({
        26: "layernorm: [1,-1,1,-1,...]",
        27: "layernorm: near-constant (5.000, 5.001, ...)",
        28: "layernorm: random with +Inf / NaN injected",
        29: "layernorm: 0, eps, 2*eps, ...",
        30: "layernorm: large values uniform[0, BF16_MAX/10]",
        31: "layernorm: small normals uniform[MIN_N, 10*MIN_N]",
    })
    # 32–37: RMSNorm
    man.update({
        32: "rmsnorm: [1,-1,1,-1,...]",
        33: "rmsnorm: near zero (0.000, 0.001, ...)",
        34: "rmsnorm: random with +Inf / NaN injected",
        35: "rmsnorm: 0, eps, 2*eps, ...",
        36: "rmsnorm: large values uniform[0, BF16_MAX/10]",
        37: "rmsnorm: small normals uniform[MIN_N, 10*MIN_N]",
    })
    # 38–41: SiLU
    man.update({
        38: "SiLU: dense around zero (linspace -5..5)",
        39: "SiLU: large positives (linspace 10..100)",
        40: "SiLU: large negatives (linspace -10..-100)",
        41: "SiLU: random with +Inf/-Inf/NaN injected",
    })
    # 42–45: GELU
    man.update({
        42: "GELU: dense around zero (linspace -5..5)",
        43: "GELU: large positives (linspace 10..100)",
        44: "GELU: large negatives (linspace -10..-100)",
        45: "GELU: random with +Inf/-Inf/NaN injected",
    })
    # 46–53: Elementwise Add/Mul（X 行；Y 由脚本配对生成）
    man.update({
        46: "Ewise X: uniform[0,10]   (pair Y for general add/mul)",
        47: "Ewise X: uniform[-10,0]  (pair Y for general add/mul)",
        48: "Ewise X: large values uniform[BF16_MAX/2, BF16_MAX] (overflow tests)",
        49: "Ewise X: small normals uniform[MIN_N, 10*MIN_N]     (underflow tests)",
        50: "Ewise X: random with +Inf injected",
        51: "Ewise X: random with -Inf injected",
        52: "Ewise X: random with NaN injected",
        53: "Ewise X: random with 0.0 injected",
    })
    # 54–63: 其他扩展/病态
    for i in range(54, 64):
        man[i] = "extra/mix patterns (lognormal / alternating extremes / different seeds)"
    return man

# ------------------------------
# 生成每一行的向量
# ------------------------------
def fill_row(idx: int, D: int, rng: np.random.Generator) -> torch.Tensor:
    """返回 float32 Tensor（稍后统一 cast 到 bf16）"""
    # 常用构件
    lin01 = torch.linspace(0, 1, D, dtype=torch.float32)
    # 用于在大范围内均匀取样
    lin_pos = torch.linspace(10.0, 100.0, D, dtype=torch.float32)
    lin_neg = torch.linspace(-10.0, -100.0, D, dtype=torch.float32)
    # 用于在 0点附近密集取样
    lin_zero = torch.linspace(-5.0, 5.0, D, dtype=torch.float32)

    # eps（LayerNorm 与 RMSNorm 的数值保护项用例）
    EPS_LN = 1e-4
    EPS_RMS = 1e-4

    # 便捷随机函数（用 numpy 生成，再转 torch）
    def unif(a, b, size):
        return torch.from_numpy(rng.uniform(a, b, size).astype(np.float32))
    def normal(mean, std, size):
        return torch.from_numpy(rng.normal(mean, std, size).astype(np.float32))

    # ---- 分类生成 ----
    if idx == 0:
        return torch.zeros(D, dtype=torch.float32)
    if idx == 1:
        return torch.ones(D, dtype=torch.float32)
    if idx == 2:
        return -torch.ones(D, dtype=torch.float32)
    if idx == 3:
        return torch.full((D,), BF16_MAX, dtype=torch.float32)
    if idx == 4:
        return torch.full((D,), -BF16_MAX, dtype=torch.float32)
    if idx == 5:
        return torch.full((D,), BF16_MIN_NORMAL, dtype=torch.float32)
    if idx == 6:
        return torch.full((D,), -BF16_MIN_NORMAL, dtype=torch.float32)
    if idx == 7:
        return torch.full((D,), SUBNORMAL_EXAMPLE_POS, dtype=torch.float32)
    if idx == 8:
        return torch.full((D,), SUBNORMAL_EXAMPLE_NEG, dtype=torch.float32)
    if idx == 9:
        base = torch.tensor([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1, 10.0, -10.0], dtype=torch.float32)
        return base.repeat(D // base.numel() + 1)[:D].clone()
    if idx == 10:
        base = torch.tensor([float('inf'), float('-inf'), float('nan'), 0.0, -0.0, 1.0, -1.0], dtype=torch.float32)
        return base.repeat(D // base.numel() + 1)[:D].clone()
    if idx == 11:
        a = torch.zeros(D, dtype=torch.float32) # 全部 +0.0
        a[1::2] = -0.0 # 奇数位置 -0.0
        return a

    if idx == 12:
        return unif(-10.0, 10.0, D) 
    if idx == 13:
        return unif(-1000.0, 1000.0, D)
    if idx == 14:
        return unif(-0.1, 0.1, D)
    if idx == 15:
        return unif(0.0, 1.0, D)
    if idx == 16:
        return normal(0.0, 5.0, D)
    if idx == 17:
        a = unif(-5.0, 5.0, D)
        # 注入若干 Inf / NaN
        inj = min(8, D)       # 计划注入个数为8个
        pos = rng.choice(D, size=inj, replace=False)   # 不重复位置，随机选择位置注入
        for k, p in enumerate(pos):
            a[p] = float('inf') if k % 2 == 0 else float('nan')
        return a
    if idx == 18:
        a = unif(-5.0, 5.0, D)
        inj = min(8, D)
        pos = rng.choice(D, size=inj, replace=False)
        for p in pos:
            a[p] = SUBNORMAL_EXAMPLE_POS if rng.random() < 0.5 else SUBNORMAL_EXAMPLE_NEG
        return a
    if idx == 19:
        return unif(0.0, BF16_MAX/2.0, D)

    if idx == 20:
        return 100.0 + 0.1 * torch.arange(D, dtype=torch.float32)
    if idx == 21:
        return -100.0 - 0.1 * torch.arange(D, dtype=torch.float32)
    if idx == 22:
        a = torch.zeros(D, dtype=torch.float32)
        a[D//2] = 10.0 # 中间索引为10
        return a
    if idx == 23:
        a = torch.zeros(D, dtype=torch.float32)
        a[D//2] = float('inf')
        return a
    if idx == 24:
        a = torch.zeros(D, dtype=torch.float32)
        a[D//3] = float('inf'); a[2*D//3] = float('inf')
        return a
    if idx == 25:
        a = torch.zeros(D, dtype=torch.float32)
        a[D//2] = float('nan')
        return a

    if idx == 26:
        base = torch.tensor([1.0, -1.0], dtype=torch.float32)
        return base.repeat(D // 2 + 1)[:D].clone()
    if idx == 27:
        # 5.000, 5.001, 5.002, ...
        return 5.0 + 0.001 * torch.arange(D, dtype=torch.float32)
    if idx == 28:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(8, D), replace=False)
        for k, p in enumerate(pos):
            a[p] = float('inf') if k % 3 == 0 else float('nan')
        return a
    if idx == 29:
        return torch.arange(D, dtype=torch.float32) * EPS_LN
    if idx == 30:
        return unif(0.0, BF16_MAX/10.0, D)
    if idx == 31:
        return unif(BF16_MIN_NORMAL, BF16_MIN_NORMAL*10.0, D)

    if idx == 32:
        base = torch.tensor([1.0, -1.0], dtype=torch.float32)
        return base.repeat(D // 2 + 1)[:D].clone()
    if idx == 33:
        return torch.arange(D, dtype=torch.float32) * 0.001
    if idx == 34:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(8, D), replace=False)
        for k, p in enumerate(pos):
            a[p] = float('inf') if k % 3 == 0 else float('nan')
        return a
    if idx == 35:
        return torch.arange(D, dtype=torch.float32) * EPS_RMS
    if idx == 36:
        return unif(0.0, BF16_MAX/10.0, D)
    if idx == 37:
        return unif(BF16_MIN_NORMAL, BF16_MIN_NORMAL*10.0, D)

    if idx == 38:
        return lin_zero
    if idx == 39:
        return lin_pos
    if idx == 40:
        return lin_neg
    if idx == 41:
        a = unif(-5.0, 5.0, D)
        # 注入 +Inf / -Inf / NaN
        pos = rng.choice(D, size=min(12, D), replace=False)
        for i, p in enumerate(pos):
            a[p] = float('inf') if i % 3 == 0 else (float('-inf') if i % 3 == 1 else float('nan'))
        return a

    if idx == 42:
        return lin_zero
    if idx == 43:
        return lin_pos
    if idx == 44:
        return lin_neg
    if idx == 45:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(12, D), replace=False)
        for i, p in enumerate(pos):
            a[p] = float('inf') if i % 3 == 0 else (float('-inf') if i % 3 == 1 else float('nan'))
        return a

    # 46–53: 仅作为 Ewise X
    if idx == 46:
        return unif(0.0, 10.0, D)
    if idx == 47:
        return unif(-10.0, 0.0, D)
    if idx == 48:
        return unif(BF16_MAX/2.0, BF16_MAX, D)
    if idx == 49:
        return unif(BF16_MIN_NORMAL, BF16_MIN_NORMAL*10.0, D)
    if idx == 50:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(8, D), replace=False)
        a[pos] = float('inf')
        return a
    if idx == 51:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(8, D), replace=False)
        a[pos] = float('-inf')
        return a
    if idx == 52:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(8, D), replace=False)
        a[pos] = float('nan')
        return a
    if idx == 53:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(8, D), replace=False)
        a[pos] = 0.0
        return a

    # 54–63：扩展/病态
    if idx in range(54, 64):
        choice = idx % 3
        if choice == 0:
            # 对数正态（偏态分布）
            a = np.random.lognormal(mean=0.0, sigma=1.0, size=D).astype(np.float32)
            return torch.from_numpy(a)
        elif choice == 1:
            # 交替极值
            a = torch.zeros(D, dtype=torch.float32)
            a[::2] = BF16_MAX
            a[1::2] = -BF16_MAX
            return a
        else:
            # 不同种子随机
            local_rng = np.random.default_rng(1000 + idx)
            return torch.from_numpy(local_rng.uniform(-50.0, 50.0, D).astype(np.float32))

    raise ValueError(f"Unhandled row index {idx}")

# ------------------------------
# 生成 Y_test_tensor（仅 46–53 有意义）
# 每个行号都设计一个“配对”的 Y，用于触发溢出/下溢/NaN 传播等
# ------------------------------
def build_Y_for_ewise(X: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    N, D = X.shape
    Y = torch.zeros_like(X, dtype=torch.float32)

    # 46: X in [0,10] → Y in [0,10]
    Y[46] = torch.from_numpy(rng.uniform(0.0, 10.0, D).astype(np.float32))

    # 47: X in [-10,0] → Y in [-10,0]
    Y[47] = torch.from_numpy(rng.uniform(-10.0, 0.0, D).astype(np.float32))

    # 48: X large (MAX/2 .. MAX) → Y large (MAX/2 .. MAX) 触发溢出
    Y[48] = torch.from_numpy(rng.uniform(BF16_MAX/2.0, BF16_MAX, D).astype(np.float32))

    # 49: X small normals → Y small normals 触发下溢/次正规
    Y[49] = torch.from_numpy(rng.uniform(BF16_MIN_NORMAL, BF16_MIN_NORMAL*10.0, D).astype(np.float32))

    # 50: X 含 +Inf → Y 含 -Inf，测试 Inf + (-Inf) 与 0*Inf
    Y[50] = torch.from_numpy(rng.uniform(-5.0, 5.0, D).astype(np.float32))
    pos = rng.choice(D, size=min(8, D), replace=False)
    Y[50, pos] = float('-inf')
    # 同时在少数位置放 0，便于 0*Inf 测试
    pos0 = rng.choice(D, size=min(8, D), replace=False)
    Y[50, pos0] = 0.0

    # 51: X 含 -Inf → Y 含 +Inf
    Y[51] = torch.from_numpy(rng.uniform(-5.0, 5.0, D).astype(np.float32))
    pos = rng.choice(D, size=min(8, D), replace=False)
    Y[51, pos] = float('inf')

    # 52: X 含 NaN → Y 任意；也额外注入少量 NaN，看传播
    Y[52] = torch.from_numpy(rng.uniform(-5.0, 5.0, D).astype(np.float32))
    pos = rng.choice(D, size=min(8, D), replace=False)
    Y[52, pos] = float('nan')

    # 53: X 含 0.0 → Y 含 Inf，测试 0*Inf = NaN
    Y[53] = torch.from_numpy(rng.uniform(-5.0, 5.0, D).astype(np.float32))
    pos = rng.choice(D, size=min(8, D), replace=False)
    Y[53, pos] = float('inf')

    return Y.to(torch.bfloat16)  # 统一 bf16

# ------------------------------
# 参考实现（PyTorch）
# ------------------------------
# def ref_softmax(x_bf16: torch.Tensor, dim=-1) -> torch.Tensor:
#     x = x_bf16.to(torch.float32)
#     # 按常规做数值稳定 softmax
#     x = x - nanmax(x, dim=dim, keepdim=True)
#     y = torch.exp(x)
#     denom = nansum(y, dim=dim, keepdim=True)
#     out = y / denom
#     return out.to(torch.bfloat16)

# def ref_layernorm(x_bf16: torch.Tensor, eps=1e-5, D=None) -> torch.Tensor:
#     x = x_bf16.to(torch.float32)
#     if D is None:
#         D = x.shape[-1]
#     mean = nanmean(x, dim=-1, keepdim=True)
#     var = nanmean((x - mean) ** 2, dim=-1, keepdim=True)
#     out = (x - mean) / torch.sqrt(var + eps)
#     return out.to(torch.bfloat16)

# def ref_rmsnorm(x_bf16: torch.Tensor, eps=1e-5) -> torch.Tensor:
#     x = x_bf16.to(torch.float32)
#     msq = nanmean(x**2, dim=-1, keepdim=True)
#     out = x / torch.sqrt(msq + eps)
#     return out.to(torch.bfloat16)

# def ref_silu(x_bf16: torch.Tensor) -> torch.Tensor:
#     x = x_bf16.to(torch.float32)
#     out = x * torch.sigmoid(x)
#     return out.to(torch.bfloat16)

# def ref_gelu(x_bf16: torch.Tensor, approximate='tanh') -> torch.Tensor:
#     x = x_bf16.to(torch.float32)
#     # 使用 PyTorch 的近似实现（默认 approximate='tanh'）
#     out = F.gelu(x, approximate=approximate)
#     return out.to(torch.bfloat16)

# def ref_add(x_bf16: torch.Tensor, y_bf16: torch.Tensor) -> torch.Tensor:
#     return (x_bf16.to(torch.float32) + y_bf16.to(torch.float32)).to(torch.bfloat16)

# def ref_mul(x_bf16: torch.Tensor, y_bf_16: torch.Tensor) -> torch.Tensor:
#     return (x_bf16.to(torch.float32) * y_bf16.to(torch.float32)).to(torch.bfloat16)

def ref_softmax(x_bf16: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # F.softmax 内部已做数值稳定，不需要手动减 max
    return F.softmax(x_bf16.to(torch.float32), dim=dim).to(torch.bfloat16)

def ref_layernorm(x_bf16: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    # F.layer_norm 需要传入 normalized_shape（最后一维的长度）
    return F.layer_norm(x, normalized_shape=(x.shape[-1],), eps=eps).to(torch.bfloat16)

def ref_rmsnorm(x_bf16: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    # 优先用内置（PyTorch 新版提供 F.rms_norm）；否则回退到等价实现
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, normalized_shape=(x.shape[-1],), eps=eps).to(torch.bfloat16)
    # fallback：x / sqrt(mean(x^2) + eps)
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms).to(torch.bfloat16)

def ref_silu(x_bf16: torch.Tensor) -> torch.Tensor:
    return F.silu(x_bf16.to(torch.float32)).to(torch.bfloat16)

def ref_gelu(x_bf16: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    return F.gelu(x_bf16.to(torch.float32), approximate=approximate).to(torch.bfloat16)

def ref_add(x_bf16: torch.Tensor, y_bf16: torch.Tensor) -> torch.Tensor:
    return torch.add(x_bf16.to(torch.float32), y_bf16.to(torch.float32)).to(torch.bfloat16)

def ref_mul(x_bf16: torch.Tensor, y_bf16: torch.Tensor) -> torch.Tensor:
    return torch.mul(x_bf16.to(torch.float32), y_bf16.to(torch.float32)).to(torch.bfloat16)
# ------------------------------
# 主流程
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True, help="输出目录")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser.add_argument("--emit_ref", action="store_true", help="是否生成七类算子的参考输出（bf16与f32）")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--D", type=int, default=768)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    N, D = args.N, args.D

    # 1) 生成 X（float32 -> bf16）
    X_f32 = torch.stack([fill_row(i, D, rng) for i in range(N)], dim=0)  # (N, D), float32
    print(X_f32[20,:10])
    X_bf16 = f32_to_bf16(X_f32)

    # 2) 生成 Y（仅 46–53 行有效）
    # Y_bf16 = torch.zeros_like(X_bf16)
    Y_bf16 = X_bf16.clone()                                      # 行 0..N-1 先与 X 相同
    Y_pair = build_Y_for_ewise(X_bf16, rng)                      # 只在 46–53 行有意义
    lo, hi = 46, 54
    Y_bf16[lo:hi] = Y_pair[lo:hi]                                # 用专用逻辑覆盖 46–53
    # Y_bf16 = build_Y_for_ewise(X_bf16, rng)

    # 3) 导出位模式（uint16）
    X_bits = to_bf16_bits(X_bf16)
    Y_bits = to_bf16_bits(Y_bf16)
    X_bits.tofile(os.path.join(args.outdir, "X_test_tensor_bf16.bin"))
    Y_bits.tofile(os.path.join(args.outdir, "Y_test_tensor_bf16.bin"))

    # 也保存 .npy / .pt 方便 Python 侧调试
    np.save(os.path.join(args.outdir, "X_test_tensor_bf16_bits.npy"), X_bits)
    np.save(os.path.join(args.outdir, "Y_test_tensor_bf16_bits.npy"), Y_bits)
    torch.save(X_bf16, os.path.join(args.outdir, "X_test_tensor_bf16.pt"))
    torch.save(Y_bf16, os.path.join(args.outdir, "Y_test_tensor_bf16.pt"))

    # 4) 写 manifest（行号→场景说明）
    manifest = build_manifest()
    with open(os.path.join(args.outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved X/Y bf16 vectors & bits to: {args.outdir}")

    # 5) 可选：生成七类算子的参考输出
    if args.emit_ref:
        # ---- Softmax：对整张 X_bf16 计算 ----
        softmax_ref_bf16 = ref_softmax(X_bf16, dim=1)
        torch.save(softmax_ref_bf16,              os.path.join(args.outdir, "ref_softmax_bf16.pt"))
        torch.save(bf16_to_f32(softmax_ref_bf16), os.path.join(args.outdir, "ref_softmax_f32.pt"))

        # ---- LayerNorm / RMSNorm（整块，eps=1e-5）----
        ln_ref_bf16  = ref_layernorm(X_bf16, eps=1e-5)
        rms_ref_bf16 = ref_rmsnorm(X_bf16,   eps=1e-5)
        torch.save(ln_ref_bf16,               os.path.join(args.outdir, "ref_layernorm_bf16.pt"))
        torch.save(bf16_to_f32(ln_ref_bf16),  os.path.join(args.outdir, "ref_layernorm_f32.pt"))
        torch.save(rms_ref_bf16,              os.path.join(args.outdir, "ref_rmsnorm_bf16.pt"))
        torch.save(bf16_to_f32(rms_ref_bf16), os.path.join(args.outdir, "ref_rmsnorm_f32.pt"))

        # ---- SiLU / GELU（整块）----
        silu_ref_bf16 = ref_silu(X_bf16)
        gelu_ref_bf16 = ref_gelu(X_bf16, approximate='tanh')  # 若你的 ref_gelu 需要该参数
        torch.save(silu_ref_bf16,               os.path.join(args.outdir, "ref_silu_bf16.pt"))
        torch.save(bf16_to_f32(silu_ref_bf16),  os.path.join(args.outdir, "ref_silu_f32.pt"))
        torch.save(gelu_ref_bf16,               os.path.join(args.outdir, "ref_gelu_bf16.pt"))
        torch.save(bf16_to_f32(gelu_ref_bf16),  os.path.join(args.outdir, "ref_gelu_f32.pt"))

        # ---- Elementwise Add / Mul（整块逐元素）----
        add_ref_bf16 = ref_add(X_bf16, Y_bf16)
        mul_ref_bf16 = ref_mul(X_bf16, Y_bf16)
        torch.save(add_ref_bf16,                os.path.join(args.outdir, "ref_add_bf16.pt"))
        torch.save(mul_ref_bf16,                os.path.join(args.outdir, "ref_mul_bf16.pt"))
        torch.save(bf16_to_f32(add_ref_bf16),   os.path.join(args.outdir, "ref_add_f32.pt"))
        torch.save(bf16_to_f32(mul_ref_bf16),   os.path.join(args.outdir, "ref_mul_f32.pt"))

        def save_ref_pack(name, t_bf16: torch.Tensor):
            # 1) PyTorch
            torch.save(t_bf16, os.path.join(args.outdir, f"{name}_bf16.pt"))
            torch.save(bf16_to_f32(t_bf16), os.path.join(args.outdir, f"{name}_f32.pt"))
            # 2) 位流/NPY（bf16 的 16bit bitstream）
            bits = to_bf16_bits(t_bf16)
            bits.tofile(os.path.join(args.outdir, f"{name}_bf16.bin"))
            np.save(os.path.join(args.outdir, f"{name}_bf16_bits.npy"), bits)
        # ========== 新增：另存为 golden_out_config_i_* 的名字 ==========
        def save_as_golden(config_idx: int, t_bf16: torch.Tensor, outdir: str, also_pt_f32: bool = False):
            """把同一个 bf16 Tensor 另存为 golden_out_config_<i>_bf16.bin
            如果 also_pt_f32=True，则同时额外保存 .pt/.f32 版本（可选）。"""
            # 保存 .bin（uint16 原始位流）
            bits = to_bf16_bits(t_bf16)
            bits.tofile(os.path.join(outdir, f"golden_out_config_{config_idx}_bf16.bin"))
            if also_pt_f32:
                torch.save(t_bf16, os.path.join(outdir, f"golden_out_config_{config_idx}_bf16.pt"))
                torch.save(bf16_to_f32(t_bf16), os.path.join(outdir, f"golden_out_config_{config_idx}_f32.pt"))
        OP2CFG = {
            "softmax":    0,
            "silu":       1,
            "rmsnorm":    2,
            "layernorm":  3,
            "gelu":       4, 
            "add":        5,
            "mul":        6,      # 同上
        }

        save_ref_pack("ref_softmax", softmax_ref_bf16)
        save_ref_pack("ref_layernorm", ln_ref_bf16)
        save_ref_pack("ref_rmsnorm", rms_ref_bf16)
        save_ref_pack("ref_silu", silu_ref_bf16)
        save_ref_pack("ref_gelu", gelu_ref_bf16)
        save_ref_pack("ref_add", add_ref_bf16)
        save_ref_pack("ref_mul", mul_ref_bf16)

        # 再按“golden_out_config_i”的旧风格额外保存一份（关键改动在这里）
        if "add" in OP2CFG:
            save_as_golden(OP2CFG["add"], add_ref_bf16, args.outdir, also_pt_f32=False)
        if "softmax" in OP2CFG:
            save_as_golden(OP2CFG["softmax"], softmax_ref_bf16, args.outdir, also_pt_f32=False)
        if "layernorm" in OP2CFG:
            save_as_golden(OP2CFG["layernorm"], ln_ref_bf16, args.outdir, also_pt_f32=False)
        if "rmsnorm" in OP2CFG:
            save_as_golden(OP2CFG["rmsnorm"], rms_ref_bf16, args.outdir, also_pt_f32=False)
        if "silu" in OP2CFG:
            save_as_golden(OP2CFG["silu"], silu_ref_bf16, args.outdir, also_pt_f32=False)
        if "gelu" in OP2CFG:
            save_as_golden(OP2CFG["gelu"], gelu_ref_bf16, args.outdir, also_pt_f32=False)
        if "mul" in OP2CFG:
            save_as_golden(OP2CFG["mul"], mul_ref_bf16, args.outdir, also_pt_f32=False)


        print("[OK] Saved PyTorch reference outputs (bf16 & f32).")

    # 6) 简要报告
    print("== Summary ==")
    print(f"X: shape={tuple(X_bf16.shape)}, dtype={X_bf16.dtype}")
    print(f"Y: shape={tuple(Y_bf16.shape)}, dtype={Y_bf16.dtype}")
    print("Artifacts:")
    print(" - X_test_tensor_bf16.bin / Y_test_tensor_bf16.bin  (uint16 bitstreams)")
    print(" - X_test_tensor_bf16.pt   / Y_test_tensor_bf16.pt  (torch tensors)")
    print(" - X_test_tensor_bf16_bits.npy / Y_test_tensor_bf16_bits.npy")
    print(" - manifest.json")
    if args.emit_ref:
        print(" - ref_*_{bf16,f32}.pt  (seven ops)")
    print("[Done]")

if __name__ == "__main__":
    main()
