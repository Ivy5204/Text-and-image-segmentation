"""Extract DR (RAD-DINO) and ultrasound (USF-MAE) features for 安医附院."""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch

FEATURE_DIR = r"E:\模型\mm4class\features"
OUT_PATH = os.path.join(FEATURE_DIR, "features_ani.npz")
RADDINO_DIR = r"E:\模型\torch_cache\rad-dino"
USFMAE_CKPT = (r"E:\模型\torch_cache\usfmae"
               r"\USF-MAE_full_pretrain_43dataset_100epochs.pt")

DR_SIZE = 518
US_SIZE = 224
MEAN_RAD, STD_RAD = 0.5307, 0.2583
MEAN_IM = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_IM = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_dr():
    from transformers import Dinov2Model

    m = Dinov2Model.from_pretrained(RADDINO_DIR, local_files_only=True)
    m.eval()
    return m


def build_us():
    import timm

    m = timm.create_model("vit_base_patch16_224", pretrained=False,
                          num_classes=0, img_size=US_SIZE)
    ckpt = torch.load(USFMAE_CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    enc = {k: v for k, v in state.items()
           if not k.startswith(("decoder", "mask_token"))}
    m.load_state_dict(enc, strict=False)
    m.eval()
    return m


def load_dr(path):
    from PIL import Image

    img = Image.open(path).convert("L").resize((DR_SIZE, DR_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN_RAD) / STD_RAD
    return torch.from_numpy(np.stack([arr] * 3, axis=0))


def load_us(path):
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((US_SIZE, US_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(((arr - MEAN_IM) / STD_IM).transpose(2, 0, 1))


@torch.no_grad()
def dr_feat(model, batch):
    h = model(pixel_values=batch).last_hidden_state
    pooled = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
    tok = h[:, 1:].transpose(1, 2)
    side = int(round(tok.shape[2] ** 0.5))
    small = torch.nn.functional.adaptive_avg_pool2d(
        tok.reshape(-1, 768, side, side), (7, 7))
    return pooled, small.flatten(2).transpose(1, 2)


@torch.no_grad()
def us_feat(model, batch):
    h = model.forward_features(batch)
    p = h[:, 1:, :]
    pooled = torch.cat([p.mean(1), p.max(1).values], dim=1)
    tok = p.transpose(1, 2)
    small = torch.nn.functional.adaptive_avg_pool2d(
        tok.reshape(-1, 768, 14, 14), (7, 7))
    return pooled, small.flatten(2).transpose(1, 2)


def run(rows, key, model, loader, fn, tag):
    subset = [r for r in rows if int(r[key]) == 1]
    print(f"\n[{tag}] {len(subset)} 条")
    feats, toks, ids = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(subset):
            ten = loader(r["dr_path" if key == "has_dr" else "us_path"])
            pooled, tok = fn(model, ten.unsqueeze(0))
            feats.append(pooled[0].numpy().astype(np.float32))
            toks.append(tok[0].to(torch.float16).numpy())
            ids.append(int(r["patient_id"]))
            if (i + 1) % 10 == 0 or i + 1 == len(subset):
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{len(subset)}  {rate:4.2f} 张/秒"
                      f"  剩余 {(len(subset)-i-1)/rate/60:4.1f} 分钟",
                      end="\r", flush=True)
    print()
    return (np.asarray(ids, dtype=np.int64), np.stack(feats), np.stack(toks))


def main():
    torch.set_flush_denormal(True)
    rows = list(csv.DictReader(
        open(os.path.join(FEATURE_DIR, "manifest_ani.csv"), encoding="utf-8-sig")))
    print(f"安医记录 {len(rows)} 条")

    out = {}
    dr = run(rows, "has_dr", build_dr(), load_dr, dr_feat, "DR RAD-DINO")
    out["dr_id"], out["dr_features"], out["dr_tokens"] = dr
    del dr

    us = run(rows, "has_us", build_us(), load_us, us_feat, "超声 USF-MAE")
    out["us_id"], out["us_features"], out["us_tokens"] = us

    np.savez_compressed(OUT_PATH, **out)
    print(f"\n已写出 {OUT_PATH} ({os.path.getsize(OUT_PATH)/1048576:.1f} MB)")
    for k in ("dr_id", "us_id"):
        print(f"  {k}: {out[k].shape}")
    print(f"  dr_features: {out['dr_features'].shape}")
    print(f"  us_features: {out['us_features'].shape}")


if __name__ == "__main__":
    main()
