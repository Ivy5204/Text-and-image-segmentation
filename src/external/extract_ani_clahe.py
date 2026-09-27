"""Re-extract 安医 features from the CLAHE-enhanced images.

Produces both the full-image and centre-cropped variants so the effect of
CLAHE can be compared against the raw baseline on identical footing.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch

RADDINO_DIR = r"E:\模型\torch_cache\rad-dino"
USFMAE_CKPT = (r"E:\模型\torch_cache\usfmae"
               r"\USF-MAE_full_pretrain_43dataset_100epochs.pt")
FDIR = r"E:\模型\mm4class\features"
OUT = os.path.join(FDIR, "features_ani_clahe.npz")

RAW_ROOT = r"E:\学习资料\第一个项目\胎肺数据\安医附院数据"
CLAHE_ROOT = r"E:\学习资料\第一个项目\胎肺数据\安医附院数据_clahe"

DR_SIZE, US_SIZE = 518, 224
KEEP = 0.70
MEAN_RAD, STD_RAD = 0.5307, 0.2583
MEAN_IM = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_IM = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def reroot(path):
    return path.replace(RAW_ROOT, CLAHE_ROOT)


def build_dr():
    from transformers import Dinov2Model
    m = Dinov2Model.from_pretrained(RADDINO_DIR, local_files_only=True)
    m.eval()
    return m


def build_us():
    import timm
    m = timm.create_model("vit_base_patch16_224", pretrained=False,
                          num_classes=0, img_size=US_SIZE)
    ck = torch.load(USFMAE_CKPT, map_location="cpu", weights_only=False)
    st = ck.get("model", ck) if isinstance(ck, dict) else ck
    enc = {k: v for k, v in st.items()
           if not k.startswith(("decoder", "mask_token"))}
    m.load_state_dict(enc, strict=False)
    m.eval()
    return m


def load(path, center, gray):
    from PIL import Image
    img = Image.open(path)
    if center:
        w, h = img.size
        dx, dy = int(w * (1 - KEEP) / 2), int(h * (1 - KEEP) / 2)
        img = img.crop((dx, dy, w - dx, h - dy))
    if gray:
        img = img.convert("L").resize((DR_SIZE, DR_SIZE), 2)
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = (a - MEAN_RAD) / STD_RAD
        return torch.from_numpy(np.stack([a] * 3, axis=0))
    img = img.convert("RGB").resize((US_SIZE, US_SIZE), 2)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(((a - MEAN_IM) / STD_IM).transpose(2, 0, 1))


@torch.no_grad()
def dr_pool(m, b):
    h = m(pixel_values=b).last_hidden_state
    return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)


@torch.no_grad()
def us_pool(m, b):
    h = m.forward_features(b)
    p = h[:, 1:, :]
    return torch.cat([p.mean(1), p.max(1).values], dim=1)


def extract(paths, model, gray, pool_fn, center, tag):
    out, t0 = [], time.time()
    with torch.no_grad():
        for i, p in enumerate(paths):
            out.append(pool_fn(model, load(p, center, gray).unsqueeze(0))[0]
                       .numpy().astype(np.float32))
            if (i + 1) % 20 == 0 or i + 1 == len(paths):
                rate = (i + 1) / (time.time() - t0)
                print(f"    [{tag}] {i+1}/{len(paths)} {rate:4.2f} 张/秒"
                      f" 剩余 {(len(paths)-i-1)/max(rate,1e-6)/60:4.1f} 分",
                      end="\r", flush=True)
    print()
    return np.stack(out)


def main():
    torch.set_flush_denormal(True)
    rows = list(csv.DictReader(
        open(os.path.join(FDIR, "manifest_ani.csv"), encoding="utf-8-sig")))
    dr = [r for r in rows if int(r["has_dr"]) == 1]
    us = [r for r in rows if int(r["has_us"]) == 1]
    print(f"安医 CLAHE 版: DR {len(dr)} 条, 超声 {len(us)} 条")

    out = {"dr_id": np.array([int(r["patient_id"]) for r in dr], dtype=np.int64),
           "us_id": np.array([int(r["patient_id"]) for r in us], dtype=np.int64)}

    print("DR (RAD-DINO)")
    m = build_dr()
    print("  完整 ..."); out["dr_full"] = extract(
        [reroot(r["dr_path"]) for r in dr], m, True, dr_pool, False, "full")
    print("  裁剪 ..."); out["dr_crop"] = extract(
        [reroot(r["dr_path"]) for r in dr], m, True, dr_pool, True, "crop")
    del m

    print("超声 (USF-MAE)")
    m = build_us()
    print("  完整 ..."); out["us_full"] = extract(
        [reroot(r["us_path"]) for r in us], m, False, us_pool, False, "full")
    print("  裁剪 ..."); out["us_crop"] = extract(
        [reroot(r["us_path"]) for r in us], m, False, us_pool, True, "crop")

    np.savez_compressed(OUT, **out)
    print(f"\n已写出 {OUT} ({os.path.getsize(OUT)/1048576:.1f} MB)")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
