"""Centre-crop the images so the model cannot use the acquisition border.

The masking experiment showed the outer 20% ring alone scores 0.746 against
0.770 for the full image, i.e. the model reads framing/acquisition context.
Masking is not the fix -- a constant block is itself an artificial pattern.
The fix is to CROP: take the central region of the original image and resize
it to the encoder input, so the border is simply not there.

Both centres are processed so internal and external numbers stay comparable.
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
FEATURE_DIR = r"E:\模型\mm4class\features"

DR_SIZE, US_SIZE = 518, 224
KEEP = 0.70
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
    ck = torch.load(USFMAE_CKPT, map_location="cpu", weights_only=False)
    st = ck.get("model", ck) if isinstance(ck, dict) else ck
    enc = {k: v for k, v in st.items()
           if not k.startswith(("decoder", "mask_token"))}
    m.load_state_dict(enc, strict=False)
    m.eval()
    return m


def crop(path, center):
    from PIL import Image
    img = Image.open(path)
    if center:
        w, h = img.size
        dx, dy = int(w * (1 - KEEP) / 2), int(h * (1 - KEEP) / 2)
        img = img.crop((dx, dy, w - dx, h - dy))
    return img


def to_dr(img):
    img = img.convert("L").resize((DR_SIZE, DR_SIZE), 2)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - MEAN_RAD) / STD_RAD
    return torch.from_numpy(np.stack([a] * 3, axis=0))


def to_us(img):
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


def extract(paths, model, to_tensor, pool_fn, center):
    feats = []
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(paths):
            feats.append(pool_fn(model, to_tensor(crop(p, center)).unsqueeze(0))[0]
                         .numpy().astype(np.float32))
            if (i + 1) % 20 == 0 or i + 1 == len(paths):
                rate = (i + 1) / (time.time() - t0)
                print(f"    {i+1}/{len(paths)}  {rate:4.2f} 张/秒"
                      f"  剩余 {(len(paths)-i-1)/max(rate,1e-6)/60:4.1f} 分钟",
                      end="\r", flush=True)
    print()
    return np.stack(feats)


def main():
    torch.set_flush_denormal(True)
    wn = list(csv.DictReader(
        open(os.path.join(FEATURE_DIR, "manifest_dual.csv"), encoding="utf-8-sig")))
    an = list(csv.DictReader(
        open(os.path.join(FEATURE_DIR, "manifest_ani.csv"), encoding="utf-8-sig")))
    wn_dr = [r["dr_path"] for r in wn if int(r["has_dr"]) == 1]
    an_dr = [r["dr_path"] for r in an if int(r["has_dr"]) == 1]
    wn_us = [r["us_path"] for r in wn if int(r["has_us"]) == 1]
    an_us = [r["us_path"] for r in an if int(r["has_us"]) == 1]
    print(f"威宁 DR {len(wn_dr)} 超声 {len(wn_us)} | 安医 DR {len(an_dr)} 超声 {len(an_us)}")
    print(f"保留中间 {KEEP:.0%}\n")

    out = {}
    print("DR (RAD-DINO)")
    drm = build_dr()
    print("  威宁 ..."); out["wn_dr"] = extract(wn_dr, drm, to_dr, dr_pool, True)
    print("  安医 ..."); out["an_dr"] = extract(an_dr, drm, to_dr, dr_pool, True)
    del drm

    print("超声 (USF-MAE)")
    usm = build_us()
    print("  威宁 ..."); out["wn_us"] = extract(wn_us, usm, to_us, us_pool, True)
    print("  安医 ..."); out["an_us"] = extract(an_us, usm, to_us, us_pool, True)

    path = os.path.join(FEATURE_DIR, "features_crop.npz")
    np.savez_compressed(path, **out)
    print(f"\n已写出 {path} ({os.path.getsize(path)/1048576:.1f} MB)")
    for k in out:
        print(f"  {k}: {out[k].shape}")


if __name__ == "__main__":
    main()
