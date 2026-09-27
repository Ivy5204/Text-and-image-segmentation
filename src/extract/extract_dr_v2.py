"""Improved DR feature extraction: higher resolution, multi-layer, TTA.

Three independent upgrades over extract_features.py, saved as separate
variants so each one's contribution can be measured on its own:

    res     224 -> 320.  Original films are 907x1495 .. 1263x1862, so 224 was
            throwing away ~16x the pixel area.  At 320 the final feature map
            is 10x10 instead of 7x7.
    layers  only the last block -> transition3 (512ch) + norm5 (1024ch),
            each pooled with GAP+GMP.
    tta     4 views (clean, hflip, rot +7, rot -7) averaged.

Saved variants (all keyed by 住院号 through the shared manifest order):
    f_last        (N, 2048)  norm5, clean, 320
    f_last_tta    (N, 2048)  norm5, 4-view mean
    f_multi       (N, 3072)  transition3 + norm5, clean, 320
    f_multi_tta   (N, 3072)  transition3 + norm5, 4-view mean
    tokens        (N, 100, 1024) norm5 clean grid at 320 (10x10)
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

XRV_CACHE = r"E:\模型\torch_cache\torchxrayvision"
FEATURE_DIR = r"E:\模型\mm4class\features"
OUT_PATH = os.path.join(FEATURE_DIR, "features_dr_v2.npz")

IMG_SIZE = 320
BATCH = 4


def build_encoder():
    import torchvision
    import torchxrayvision as xrv

    xrv_model = xrv.models.DenseNet(
        weights="densenet121-res224-chex", cache_dir=XRV_CACHE
    )
    state = {k: v for k, v in xrv_model.state_dict().items()
             if k.startswith("features.")}
    model = torchvision.models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                     padding=3, bias=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    model.eval()
    return model


def load_dr(path, size=IMG_SIZE):
    from PIL import Image

    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy((arr * 1024.0).astype(np.float32)).unsqueeze(0)


def make_views(img: torch.Tensor):
    """clean, horizontal flip, rotate +7, rotate -7."""
    views = [img]
    views.append(torch.flip(img, dims=[2]))
    for angle in (7.0, -7.0):
        rad = np.radians(angle)
        theta = torch.tensor(
            [[np.cos(rad), -np.sin(rad), 0.0],
             [np.sin(rad), np.cos(rad), 0.0]], dtype=torch.float32
        ).unsqueeze(0)
        grid = F.affine_grid(theta, (1, 1, IMG_SIZE, IMG_SIZE), align_corners=False)
        views.append(F.grid_sample(img.unsqueeze(0), grid,
                                   align_corners=False)[0])
    return views


def pool(fm: torch.Tensor) -> torch.Tensor:
    act = F.relu(fm)
    return torch.cat([
        F.adaptive_avg_pool2d(act, 1).flatten(1),
        F.adaptive_max_pool2d(act, 1).flatten(1),
    ], dim=1)


def main():
    torch.set_flush_denormal(True)
    manifest = list(csv.DictReader(
        open(os.path.join(FEATURE_DIR, "manifest.csv"), encoding="utf-8-sig")))
    rows = [r for r in manifest if r["modality"] == "DR"]
    print(f"DR 样本 {len(rows)} 条, 输入 {IMG_SIZE}px, "
          f"{len(make_views(torch.zeros(1, IMG_SIZE, IMG_SIZE)))} 个 TTA 视图")

    model = build_encoder()
    children = list(model.features.children())
    captured = {}
    children[9].register_forward_hook(
        lambda m, i, o: captured.__setitem__("t3", o))
    children[11].register_forward_hook(
        lambda m, i, o: captured.__setitem__("n5", o))
    print("已挂 hook: transition3(索引9, 512ch) 和 norm5(索引11, 1024ch)")

    with torch.no_grad():
        model(torch.randn(BATCH, 1, IMG_SIZE, IMG_SIZE))
    print(f"  特征图尺寸: t3 {tuple(captured['t3'].shape[1:])}, "
          f"n5 {tuple(captured['n5'].shape[1:])}")

    n = len(rows)
    f_last = np.zeros((n, 2048), dtype=np.float32)
    f_last_tta = np.zeros((n, 2048), dtype=np.float32)
    f_multi = np.zeros((n, 3072), dtype=np.float32)
    f_multi_tta = np.zeros((n, 3072), dtype=np.float32)
    tokens = np.zeros((n, 100, 1024), dtype=np.float16)
    ids = []

    t0 = time.time()
    with torch.no_grad():
        for i, row in enumerate(rows):
            img = load_dr(row["path"])
            last_views, multi_views = [], []
            for v in make_views(img):
                model(v.unsqueeze(0))
                p_t3 = pool(captured["t3"])
                p_n5 = pool(captured["n5"])
                last_views.append(p_n5)
                multi_views.append(torch.cat([p_t3, p_n5], dim=1))

            f_last[i] = last_views[0][0].numpy()
            f_last_tta[i] = torch.stack(last_views).mean(0)[0].numpy()
            f_multi[i] = multi_views[0][0].numpy()
            f_multi_tta[i] = torch.stack(multi_views).mean(0)[0].numpy()
            tokens[i] = (captured["n5"][0].flatten(1).transpose(0, 1)
                         .to(torch.float16).numpy())
            ids.append(int(row["patient_id"]))

            done = i + 1
            rate = done / (time.time() - t0)
            eta = (n - done) / rate if rate else 0
            print(f"  {done}/{n}  {rate:4.2f} 张/秒  剩余 {eta/60:4.1f} 分钟",
                  end="\r", flush=True)
    print()

    np.savez_compressed(
        OUT_PATH,
        patient_id=np.asarray(ids, dtype=np.int64),
        f_last=f_last,
        f_last_tta=f_last_tta,
        f_multi=f_multi,
        f_multi_tta=f_multi_tta,
        tokens=tokens,
    )
    print(f"已写出 {OUT_PATH} ({os.path.getsize(OUT_PATH)/1048576:.1f} MB)")
    for name in ("f_last", "f_last_tta", "f_multi", "f_multi_tta"):
        print(f"  {name:<12} {eval(name).shape}")
    print(f"  tokens       {tokens.shape}")


if __name__ == "__main__":
    main()
