"""合成叶片图像生成器：按病斑规律生成三类叶片图片。

真实项目中把 generate_dataset() 换成「遍历图片目录 + 打标签」即可，
后续特征提取与训练流程不变。
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["healthy", "powdery_mildew", "leaf_rust"]
IMAGE_SIZE = (128, 128)
LEAF_BASE = (46, 125, 60)  # 健康叶片的基准绿色


def _leaf_mask(size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """生成椭圆形叶片掩码，避免背景干扰特征。"""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    rx, ry = w * 0.42, h * 0.40
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def _draw_lesions(draw: ImageDraw.ImageDraw, kind: str, rng: random.Random) -> None:
    w, h = IMAGE_SIZE
    if kind == "healthy":
        return
    if kind == "powdery_mildew":  # 灰白色大斑块
        for _ in range(rng.randint(14, 22)):
            x, y = rng.randint(10, w - 10), rng.randint(10, h - 10)
            r = rng.randint(6, 13)
            shade = rng.randint(205, 235)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(shade, shade, shade - 8))
    elif kind == "leaf_rust":  # 橙褐色密集小点
        for _ in range(rng.randint(60, 90)):
            x, y = rng.randint(8, w - 8), rng.randint(8, h - 8)
            r = rng.randint(1, 3)
            draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=(rng.randint(175, 215), rng.randint(95, 130), rng.randint(30, 55)),
            )


def generate_image(kind: str, rng: random.Random | None = None) -> Image.Image:
    rng = rng or random.Random()
    base = np.zeros((*IMAGE_SIZE[::-1], 3), dtype=np.uint8)
    base[:] = (235, 240, 232)  # 背景
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 9, base.shape)
    leaf = np.clip(np.array(LEAF_BASE) + noise, 0, 255).astype(np.uint8)
    mask = _leaf_mask()
    base[mask] = leaf[mask]
    img = Image.fromarray(base)
    _draw_lesions(ImageDraw.Draw(img), kind, rng)
    return img


def generate_dataset(out_dir: str | Path, per_class: int = 120, seed: int = 42) -> list[tuple[Path, str]]:
    out = Path(out_dir)
    rng = random.Random(seed)
    samples: list[tuple[Path, str]] = []
    for kind in CLASSES:
        (out / kind).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = generate_image(kind, rng)
            path = out / kind / f"{kind}_{i:03d}.png"
            img.save(path)
            samples.append((path, kind))
    return samples
