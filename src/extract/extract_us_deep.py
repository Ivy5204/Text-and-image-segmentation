"""Extract ultrasound features with the USF-MAE foundation model.

USF-MAE (Megahed et al., Biomedical Signal Processing and Control 2026,
arXiv 2510.22990) is a ViT-Base masked autoencoder pretrained on 370k
ultrasound images from 46 open datasets.  The checkpoint is a plain MAE state
dict whose encoder naming matches timm exactly, so the encoder is rebuilt with
timm and the decoder half of the checkpoint is simply ignored.

Output mirrors extract_features.py so the two feature sets are drop-in
interchangeable:
    features  (N, 1536)   GAP + GMP over the last hidden state
    tokens    (N, 196, 768) the 14x14 patch grid (pooled later by the fusion)
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch

CKPT = (r"E:\模型\torch_cache\usfmae"
        r"\USF-MAE_full_pretrain_43dataset_100epochs.pt")
FEATURE_DIR = r"E:\模型\mm4class\features"
OUT_PATH = os.path.join(FEATURE_DIR, "features_us_usfmae.npz")

IMG_SIZE = 224
BATCH = 8
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_encoder():
    import timm

    model = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0, img_size=IMG_SIZE
    )
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    # Keep only encoder tensors; drop the MAE decoder and the mask token.
    encoder = {
        k: v for k, v in state.items()
        if not k.startswith(("decoder", "mask_token"))
    }
    missing, unexpected = model.load_state_dict(encoder, strict=False)
    print(f"  加载: missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("    意外键:", unexpected[:5])
    if missing:
        print("    缺失键:", missing[:5])
    model.eval()
    return model


def load_us(path):
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))


def main():
    torch.set_flush_denormal(True)
    manifest = list(csv.DictReader(
        open(os.path.join(FEATURE_DIR, "manifest.csv"), encoding="utf-8-sig")))
    rows = [r for r in manifest if r["modality"] == "US"]
    print(f"超声样本 {len(rows)} 条")

    print("构建 USF-MAE 编码器 (ViT-B/16)")
    model = build_encoder()
    n = sum(p.numel() for p in model.parameters())
    print(f"  编码器参数量 {n/1e6:.1f} M")

    # warm up so the timing below is honest
    with torch.no_grad():
        for _ in range(3):
            model(torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE))

    feats, tokens, ids = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            batch = torch.stack([load_us(r["path"]) for r in chunk])
            tok = model.forward_features(batch)      # (B, 197, 768)
            patch = tok[:, 1:, :]                    # drop cls
            gap = patch.mean(dim=1)
            gmp = patch.max(dim=1).values
            feats.append(torch.cat([gap, gmp], dim=1).numpy().astype(np.float32))
            tokens.append(patch.numpy().astype(np.float16))
            ids.extend(int(r["patient_id"]) for r in chunk)
            done = min(start + BATCH, len(rows))
            rate = done / (time.time() - t0)
            eta = (len(rows) - done) / rate if rate else 0
            print(f"  {done}/{len(rows)}  {rate:4.1f} 张/秒  剩余 {eta/60:4.1f} 分钟",
                  end="\r", flush=True)
    print()

    features = np.concatenate(feats, axis=0)
    token_arr = np.concatenate(tokens, axis=0)
    print(f"池化特征 {features.shape}, token {token_arr.shape}")

    np.savez_compressed(
        OUT_PATH,
        patient_id=np.asarray(ids, dtype=np.int64),
        features=features,
        tokens=token_arr,
    )
    print(f"已写出 {OUT_PATH}  "
          f"({os.path.getsize(OUT_PATH)/1048576:.1f} MB)")


if __name__ == "__main__":
    main()
