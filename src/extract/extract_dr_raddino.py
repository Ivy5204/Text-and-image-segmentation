"""DR features from RAD-DINO (microsoft/rad-dino).

RAD-DINO is a DINOv2 ViT-B/14 self-supervised on ~800k chest X-rays.  It is a
deliberately different inductive bias from the current CheXpert-supervised
DenseNet121: self-supervised vs supervised, transformer vs convolution,
800k vs 224k pretraining images.

The native resolution is 518 (37x37 = 1369 tokens), which is far too expensive
on this CPU-only box, so a size probe runs first and the largest affordable
resolution is used for the full pass.  Position embeddings are interpolated
automatically by transformers.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = r"E:\模型\torch_cache\rad-dino"
FEATURE_DIR = r"E:\模型\mm4class\features"
OUT_PATH = os.path.join(FEATURE_DIR, "features_dr_raddino.npz")

MEAN = 0.5307
STD = 0.2583
TARGET_PER_IMAGE = 2.5      # seconds; pick the largest size under this


def load_model():
    from transformers import Dinov2Model

    model = Dinov2Model.from_pretrained(MODEL_DIR, local_files_only=True)
    model.eval()
    return model


def load_image(path, size):
    from PIL import Image

    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    # RAD-DINO expects 3 channels; grayscale replicated.
    return torch.from_numpy(np.stack([arr] * 3, axis=0))


@torch.no_grad()
def pooled_and_tokens(model, batch):
    out = model(pixel_values=batch)
    hidden = out.last_hidden_state          # (B, 1+N, 768), CLS first
    cls = hidden[:, 0]
    patch = hidden[:, 1:]
    gap = patch.mean(dim=1)
    feats = torch.cat([cls, gap], dim=1)    # 1536, same width as the US branch
    return feats, patch


def probe_size(model, paths):
    print("分辨率测速 (每档 3 张):")
    chosen = 224
    for size in (224, 336, 518):
        batch = torch.stack([load_image(paths[0], size)])
        pooled_and_tokens(model, batch)     # warm up
        t0 = time.perf_counter()
        for _ in range(3):
            pooled_and_tokens(model, batch)
        per = (time.perf_counter() - t0) / 3
        ntok = (size // 14) ** 2
        print(f"  {size:>4}px  ({ntok:>4} token)  {per:5.2f} 秒/张"
              f"  483 张约 {per*483/60:5.1f} 分钟")
        if per <= TARGET_PER_IMAGE:
            chosen = size
    print(f"  选用 {chosen}px")
    return chosen


def main():
    torch.set_flush_denormal(True)
    manifest = list(csv.DictReader(
        open(os.path.join(FEATURE_DIR, "manifest.csv"), encoding="utf-8-sig")))
    rows = [r for r in manifest if r["modality"] == "DR"]
    print(f"DR 样本 {len(rows)} 条")

    model = load_model()
    n = sum(p.numel() for p in model.parameters())
    print(f"RAD-DINO 加载完成: {n/1e6:.1f} M 参数")

    size = probe_size(model, [r["path"] for r in rows])

    feats, tokens, ids = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, row in enumerate(rows):
            batch = load_image(row["path"], size).unsqueeze(0)
            f, patch = pooled_and_tokens(model, batch)
            feats.append(f[0].numpy().astype(np.float32))
            side = int(round(patch.shape[1] ** 0.5))
            grid = patch[0].transpose(0, 1).reshape(-1, side, side).unsqueeze(0)
            small = F.adaptive_avg_pool2d(grid, (7, 7))
            tokens.append(
                small.flatten(2).transpose(1, 2)[0].to(torch.float16).numpy()
            )
            ids.append(int(row["patient_id"]))
            done = i + 1
            rate = done / (time.time() - t0)
            eta = (len(rows) - done) / rate if rate else 0
            print(f"  {done}/{len(rows)}  {rate:4.2f} 张/秒  "
                  f"剩余 {eta/60:4.1f} 分钟", end="\r", flush=True)
    print()

    features = np.stack(feats)
    token_arr = np.stack(tokens)
    print(f"池化特征 {features.shape}, token {token_arr.shape}")
    np.savez_compressed(
        OUT_PATH,
        patient_id=np.asarray(ids, dtype=np.int64),
        features=features,
        tokens=token_arr,
        image_size=np.asarray([size]),
    )
    print(f"已写出 {OUT_PATH} ({os.path.getsize(OUT_PATH)/1048576:.1f} MB)")


if __name__ == "__main__":
    main()
