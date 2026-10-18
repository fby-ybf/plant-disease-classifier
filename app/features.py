"""特征提取：颜色统计 + 病斑几何 + 边缘密度（12 维）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

FEATURE_NAMES = [
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
    "h_mean", "s_mean", "v_mean",
    "green_ratio", "lesion_ratio", "edge_density",
]


def extract(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    diff = mx - mn
    # HSV
    h = np.where(diff < 1e-6, 0.0, np.where(mx == r, ((g - b) / np.maximum(diff, 1e-6)) % 6,
               np.where(mx == g, (b - r) / np.maximum(diff, 1e-6) + 2, (r - g) / np.maximum(diff, 1e-6) + 4))) / 6.0
    s = np.where(mx < 1e-6, 0.0, diff / np.maximum(mx, 1e-6))
    v = mx
    # 掩码：只统计叶片区域（去掉浅色背景）
    leaf = v < 0.88
    green_ratio = float(((g > r + 0.05) & (g > b + 0.05) & leaf).mean())
    lesion_ratio = float((((s < 0.18) | (h > 0.03) & (h < 0.12)) & leaf).mean())
    # 边缘密度：相邻像素差的均值（Sobel 近似）
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return np.array(
        [
            r.mean(), g.mean(), b.mean(), r.std(), g.std(), b.std(),
            h[leaf].mean() if leaf.any() else 0.0,
            s[leaf].mean() if leaf.any() else 0.0,
            v[leaf].mean() if leaf.any() else 0.0,
            green_ratio, lesion_ratio, float(gx + gy),
        ],
        dtype="float32",
    )


def extract_many(paths: list[Path]) -> np.ndarray:
    return np.vstack([extract(p) for p in paths])
