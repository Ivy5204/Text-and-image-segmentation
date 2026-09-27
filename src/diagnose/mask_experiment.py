"""Is the model reading the image border, and can that alone solve the task?

The Grad-CAM mass analysis shows 41.5% of the DR energy and 41.9% of the
ultrasound energy sitting in the outer 80-100% ring (area share: 36%), and the
three abnormal classes lean on that ring much harder than 阴性 does.  That is
the signature the course warned about ("the model reads the hospital name").

Two masked variants settle it:
    center-only : outer 20% of each side zeroed  -> keeps anatomy, kills text
    border-only : inner 60% zeroed              -> keeps text, kills anatomy

If border-only scores near the full model, the task is being solved without the
lungs.  If it collapses, the border carries genuine anatomical signal (rib
cage, costophrenic angles, diaphragm) rather than a watermark shortcut.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch

RADDINO_DIR = r"E:\模型\torch_cache\rad-dino"
FEATURE_DIR = r"E:\模型\mm4class\features"
OUT_PATH = os.path.join(FEATURE_DIR, "features_dr_masks.npz")

SIZE = 518
MARGIN = 0.20          # outer 20% of each side is blanked
MEAN = 0.5307
STD = 0.2583


def load_model():
    from transformers import Dinov2Model

    m = Dinov2Model.from_pretrained(RADDINO_DIR, local_files_only=True)
    m.eval()
    return m


def load_masked(path, mode):
    from PIL import Image

    img = Image.open(path).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    n = int(round(SIZE * MARGIN))
    if mode == "center":                # blank the border, keep the centre
        arr[:n, :] = MEAN
        arr[-n:, :] = MEAN
        arr[:, :n] = MEAN
        arr[:, -n:] = MEAN
    elif mode == "border":              # blank the centre, keep the border
        arr[n:-n, n:-n] = MEAN
    arr = (arr - MEAN) / STD
    return torch.from_numpy(np.stack([arr] * 3, axis=0))


@torch.no_grad()
def pool(model, batch):
    out = model(pixel_values=batch)
    hidden = out.last_hidden_state
    return torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)


def main():
    torch.set_flush_denormal(True)
    rows = [r for r in csv.DictReader(
        open(os.path.join(FEATURE_DIR, "manifest_dual.csv"), encoding="utf-8-sig"))
        if int(r["has_dr"]) == 1]
    print(f"DR 样本 {len(rows)} 条, 输入 {SIZE}px, 遮蔽比例 {MARGIN:.0%}")

    model = load_model()
    with torch.no_grad():
        pool(model, torch.randn(1, 3, SIZE, SIZE))

    out = {}
    for mode in ("center", "border"):
        feats, ids = [], []
        t0 = time.time()
        with torch.no_grad():
            for i, r in enumerate(rows):
                ten = load_masked(r["dr_path"], mode).unsqueeze(0)
                feats.append(pool(model, ten)[0].numpy().astype(np.float32))
                ids.append(int(r["patient_id"]))
                done = i + 1
                rate = done / (time.time() - t0)
                print(f"  [{mode}] {done}/{len(rows)}  {rate:4.2f} 张/秒"
                      f"  剩余 {(len(rows)-done)/rate/60:4.1f} 分钟",
                      end="\r", flush=True)
        print()
        out[f"{mode}_features"] = np.stack(feats)
        out[f"{mode}_ids"] = np.asarray(ids, dtype=np.int64)

    np.savez_compressed(OUT_PATH, **out)
    print(f"已写出 {OUT_PATH} ({os.path.getsize(OUT_PATH)/1048576:.1f} MB)")


if __name__ == "__main__":
    main()
