"""Histogram matching: align 安医's intensity distribution to 威宁's.

CLAHE equalises local contrast but each image still keeps its own global
intensity distribution.  Histogram matching goes further: every target image's
histogram is warped onto a single reference histogram measured from the source
centre.  If the remaining domain gap is largely a tone-mapping difference,
this should close part of it.

Two variants are produced so the gain can be attributed:
    match      : histogram matching on the raw image
    clahe+match: CLAHE first (matching 威宁's pipeline), then histogram matching
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
OUT = os.path.join(FDIR, "features_ani_match.npz")

WN_CLAHE = r"E:\学习资料\第一个项目\胎肺数据\威宁妇幼数据_clahe"
AN_RAW = r"E:\学习资料\第一个项目\胎肺数据\安医附院数据"

DR_SIZE, US_SIZE = 518, 224
KEEP = 0.70
MEAN_RAD, STD_RAD = 0.5307, 0.2583
MEAN_IM = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_IM = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def reference_histogram(n_per_folder=12):
    """Average intensity histogram of the 威宁 CLAHE images = the target tone."""
    from PIL import Image

    acc = np.zeros(256, dtype=np.float64)
    count = 0
    for folder in sorted(os.listdir(WN_CLAHE)):
        d = os.path.join(WN_CLAHE, folder)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d)
                       if f.lower().endswith((".jpg", ".png", ".bmp", ".jpeg")))
        for f in files[:n_per_folder]:
            arr = np.asarray(Image.open(os.path.join(d, f)).convert("L"))
            acc += np.bincount(arr.ravel(), minlength=256)
            count += 1
    acc /= max(count, 1)
    cdf = np.cumsum(acc)
    cdf /= cdf[-1]
    return cdf


def match_to_ref(gray, ref_cdf):
    """Warp a uint8 image so its histogram follows ref_cdf."""
    src = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(src.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    if cdf[-1] <= 0:
        return src
    cdf /= cdf[-1]
    lut = np.interp(cdf, ref_cdf, np.arange(256)).astype(np.uint8)
    return lut[src]


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


def prepare(path, variant, ref_cdf):
    import cv2
    from PIL import Image

    gray = Image.open(path).convert("L")
    arr = np.asarray(gray, dtype=np.uint8)
    if variant.startswith("clahe"):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe.apply(arr)
    if variant.endswith("match"):
        arr = match_to_ref(arr, ref_cdf)
    return Image.fromarray(arr)


def to_dr(img, center):
    if center:
        w, h = img.size
        dx, dy = int(w * (1 - KEEP) / 2), int(h * (1 - KEEP) / 2)
        img = img.crop((dx, dy, w - dx, h - dy))
    img = img.resize((DR_SIZE, DR_SIZE), 2)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - MEAN_RAD) / STD_RAD
    return torch.from_numpy(np.stack([a] * 3, axis=0))


def to_us(img, center):
    img = img.convert("RGB")
    if center:
        w, h = img.size
        dx, dy = int(w * (1 - KEEP) / 2), int(h * (1 - KEEP) / 2)
        img = img.crop((dx, dy, w - dx, h - dy))
    img = img.resize((US_SIZE, US_SIZE), 2)
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


def run(paths, model, pool_fn, variant, ref_cdf, gray, center, tag):
    out, t0 = [], time.time()
    with torch.no_grad():
        for i, p in enumerate(paths):
            img = prepare(p, variant, ref_cdf)
            ten = to_dr(img, center) if gray else to_us(img, center)
            out.append(pool_fn(model, ten.unsqueeze(0))[0].numpy().astype(np.float32))
            if (i + 1) % 20 == 0 or i + 1 == len(paths):
                rate = (i + 1) / (time.time() - t0)
                print(f"    [{tag}] {i+1}/{len(paths)} {rate:4.2f}/秒"
                      f" 剩余 {(len(paths)-i-1)/max(rate,1e-6)/60:4.1f} 分",
                      end="\r", flush=True)
    print()
    return np.stack(out)


def main():
    torch.set_flush_denormal(True)
    print("计算威宁参考直方图 ...")
    ref = reference_histogram()
    print(f"  完成（参考 CDF 长度 {len(ref)}）")

    rows = list(csv.DictReader(
        open(os.path.join(FDIR, "manifest_ani.csv"), encoding="utf-8-sig")))
    dr = [r for r in rows if int(r["has_dr"]) == 1]
    us = [r for r in rows if int(r["has_us"]) == 1]
    out = {"dr_id": np.array([int(r["patient_id"]) for r in dr], dtype=np.int64),
           "us_id": np.array([int(r["patient_id"]) for r in us], dtype=np.int64)}

    drm, usm = build_dr(), build_us()
    for variant in ("match", "clahematch"):
        nice = "匹配" if variant == "match" else "CLAHE+匹配"
        print(f"\n变体: {nice}")
        print("  DR ...")
        out[f"dr_{variant}_full"] = run(
            [r["dr_path"] for r in dr], drm, dr_pool, variant, ref, True,
            False, "dr-" + variant)
        out[f"dr_{variant}_crop"] = run(
            [r["dr_path"] for r in dr], drm, dr_pool, variant, ref, True,
            True, "dr-" + variant)
        print("  超声 ...")
        out[f"us_{variant}_full"] = run(
            [r["us_path"] for r in us], usm, us_pool, variant, ref, False,
            False, "us-" + variant)
        out[f"us_{variant}_crop"] = run(
            [r["us_path"] for r in us], usm, us_pool, variant, ref, False,
            True, "us-" + variant)

    np.savez_compressed(OUT, **out)
    print(f"\n已写出 {OUT} ({os.path.getsize(OUT)/1048576:.1f} MB)")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
