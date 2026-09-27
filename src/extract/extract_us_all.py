"""Extract ultrasound features for EVERY sample that has an ultrasound image.

The single-image pipeline only pulled the ultrasound for the 396 samples that
had no DR, discarding it for the other 483.  In fact all 879 samples have an
ultrasound, so the dual-image setup gets 1362 images instead of 879.

Same encoder and layout as extract_features.py: USF-MAE (ViT-B/16), 224px,
GAP+GMP = 1536 dims, plus the 14x14 patch grid pooled to 7x7.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

CKPT = (r"E:\模型\torch_cache\usfmae"
        r"\USF-MAE_full_pretrain_43dataset_100epochs.pt")
FEATURE_DIR = r"E:\模型\mm4class\features"
OUT_PATH = os.path.join(FEATURE_DIR, "features_us_all.npz")

IMG_SIZE = 224
BATCH = 8
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_encoder():
    import timm

    model = timm.create_model("vit_base_patch16_224", pretrained=False,
                              num_classes=0, img_size=IMG_SIZE)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    encoder = {k: v for k, v in state.items()
               if not k.startswith(("decoder", "mask_token"))}
    missing, unexpected = model.load_state_dict(encoder, strict=False)
    print(f"  权重加载: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def load_us(path):
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE),
                                                 Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(((arr - MEAN) / STD).transpose(2, 0, 1))


def main():
    torch.set_flush_denormal(True)
    rows = list(csv.DictReader(
        open(os.path.join(FEATURE_DIR, "manifest_dual.csv"), encoding="utf-8-sig")))
    rows = [r for r in rows if int(r["has_us"]) == 1]
    print(f"有超声图的样本 {len(rows)} 条")

    model = build_encoder()
    with torch.no_grad():
        for _ in range(2):
            model(torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE))

    feats, tokens, ids = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            batch = torch.stack([load_us(r["us_path"]) for r in chunk])
            tok = model.forward_features(batch)      # (B, 197, 768)
            patch = tok[:, 1:, :]
            gap = patch.mean(dim=1)
            gmp = patch.max(dim=1).values
            feats.append(torch.cat([gap, gmp], dim=1).numpy().astype(np.float32))

            # 14x14 -> 7x7 to match the layout the fusion stage expects
            grid = patch.transpose(1, 2).reshape(-1, 768, 14, 14)
            small = F.adaptive_avg_pool2d(grid, (7, 7))
            tokens.append(small.flatten(2).transpose(1, 2)
                          .to(torch.float16).numpy())
            ids.extend(int(r["patient_id"]) for r in chunk)

            done = min(start + BATCH, len(rows))
            rate = done / (time.time() - t0)
            eta = (len(rows) - done) / rate if rate else 0
            print(f"  {done}/{len(rows)}  {rate:4.1f} 张/秒  "
                  f"剩余 {eta/60:4.1f} 分钟", end="\r", flush=True)
    print()

    features = np.concatenate(feats, axis=0)
    token_arr = np.concatenate(tokens, axis=0)
    print(f"池化特征 {features.shape}, token {token_arr.shape}")
    np.savez_compressed(OUT_PATH, patient_id=np.asarray(ids, dtype=np.int64),
                        features=features, tokens=token_arr)
    print(f"已写出 {OUT_PATH} ({os.path.getsize(OUT_PATH)/1048576:.1f} MB)")


if __name__ == "__main__":
    main()
