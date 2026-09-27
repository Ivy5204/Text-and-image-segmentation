"""Frozen-backbone feature extraction for the newborn DR/US + tabular project.

Route 1B: the backbones are never fine-tuned.  Each image is pushed through a
frozen encoder exactly once and the pooled features are cached to disk.  All
downstream experiments (fusion heads, 5x5 CV, ablations) then run on these
feature vectors, which takes seconds instead of hours on this CPU-only laptop.

Encoders (module 1 decision = plan A):
  DR     : DenseNet121 + CheXpert weights (torchxrayvision)
  US     : ConvNeXt-Tiny + ImageNet weights (timm)

Why torchvision's DenseNet instead of torchxrayvision's:
  torchxrayvision ships its own DenseNet implementation which is ~48x slower
  on CPU (4.90 s/img vs 0.101 s/img) for bit-identical outputs.  All 725
  `features.*` keys match exactly, so the CheXpert weights are loaded into
  torchvision's implementation instead.

Outputs (in OUT_DIR):
  manifest.csv      one row per image: patient id, class, modality, split, path
  features_dr.npz   patient ids + pooled features + spatial tokens for DR
  features_us.npz   patient ids + pooled features + spatial tokens for US

Two kinds of descriptor are cached:
  features  (N, 2048 / 1536)  GAP+GMP of the last feature map, for the simple
                              late-fusion heads
  tokens    (N, 49, 1024/768) the 7x7 spatial grid flattened, which is what
                              lets the fusion stage use real cross-attention
"""

from __future__ import annotations

import csv
import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- config ----

IMG_ROOT = r"E:\学习资料\第一个项目\胎肺数据\威宁妇幼数据_clahe"
CSV_DIR = r"E:\模型\PythonProject"
OUT_DIR = r"E:\模型\mm4class\features"
XRV_CACHE = r"E:\模型\torch_cache\torchxrayvision"

IMG_SIZE = 224
BATCH_SIZE = 16
WARMUP_BATCHES = 4
DR_FEAT_DIM = 2048   # DenseNet121: 1024 GAP + 1024 GMP
US_FEAT_DIM = 1536   # ConvNeXt-Tiny last stage: 768 GAP + 768 GMP

CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]          # label 0,1,2,3
CLASS_TO_LABEL = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------- indexing ----

def scan_image_index(root: str):
    """Walk the class folders and return {(modality, patient_id): path}.

    Folder names are inconsistent ("DR  ARDS" uses two spaces, the others
    three), so classes and modalities are detected by substring rather than
    by exact folder name.
    """
    index = {}
    unknown_dirs = []

    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder):
            continue

        modality = "DR" if entry.strip().upper().startswith("DR") else None
        if modality is None and "四腔心" in entry:
            modality = "US"

        cls = None
        for candidate in CLASSES:
            if candidate in entry:
                cls = candidate
                break

        if modality is None or cls is None:
            unknown_dirs.append(entry)
            continue

        for fname in os.listdir(folder):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in IMAGE_EXTS:
                continue
            try:
                patient_id = int(stem)
            except ValueError:
                continue
            index[(modality, patient_id)] = {
                "path": os.path.join(folder, fname),
                "class": cls,
                "label": CLASS_TO_LABEL[cls],
            }

    if unknown_dirs:
        print("  警告: 无法识别用途的目录:", unknown_dirs)
    return index


def build_manifest(index):
    """One row per CSV record, with the image that this sample will use.

    Strategy (unchanged from the existing project): prefer the DR image, fall
    back to ultrasound when a patient has no DR.
    """
    rows = []
    stats = {"DR": 0, "US": 0, "missing": 0}

    for split in ("train", "val", "test"):
        csv_path = os.path.join(CSV_DIR, f"{split}.csv")
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            for record in csv.DictReader(fh):
                patient_id = int(record["住院号"])
                label = int(record["label"])

                chosen = None
                for modality in ("DR", "US"):
                    hit = index.get((modality, patient_id))
                    if hit is not None:
                        chosen = (modality, hit)
                        break

                if chosen is None:
                    stats["missing"] += 1
                    continue

                modality, hit = chosen
                stats[modality] += 1
                rows.append(
                    {
                        "patient_id": patient_id,
                        "split": split,
                        "label": label,
                        "class": hit["class"],
                        "modality": modality,
                        "path": hit["path"],
                        "label_matches_folder": int(label == hit["label"]),
                    }
                )

    return rows, stats


# ------------------------------------------------------------- encoders ----

def build_dr_encoder():
    import torchvision
    import torchxrayvision as xrv

    xrv_model = xrv.models.DenseNet(
        weights="densenet121-res224-chex", cache_dir=XRV_CACHE
    )
    state = {k: v for k, v in xrv_model.state_dict().items()
             if k.startswith("features.")}

    model = torchvision.models.densenet121(weights=None)
    model.features.conv0 = torch.nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"意外的权重键: {unexpected}"
    model.eval()
    return model.features


def build_us_encoder():
    """Returns (feature_map_fn, label).

    timm models expose ``forward_features`` which yields the pre-pooling
    feature map, so the same GAP+GMP pooling used for DR applies here too.
    """
    import timm

    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
    model.eval()
    return (lambda x: model.forward_features(x)), "ConvNeXt-Tiny(forward_features)"


# ------------------------------------------------------------ transforms ----

def load_dr_tensor(path: str) -> torch.Tensor:
    """Grayscale X-ray, scaled into torchxrayvision's [-1024, 1024] range."""
    from PIL import Image

    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)          # 0..255
    arr = arr / 255.0 * 2.0 - 1.0                    # -1..1
    arr = (arr * 1024.0).astype(np.float32)          # -1024..1024
    return torch.from_numpy(arr).unsqueeze(0)        # 1,H,W


def load_us_tensor(path: str) -> torch.Tensor:
    """RGB ultrasound with standard ImageNet normalisation."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))  # 3,H,W


def pooled_features(feature_map: torch.Tensor) -> torch.Tensor:
    """GAP + GMP concatenated, the standard frozen-backbone descriptor."""
    act = F.relu(feature_map)
    gap = F.adaptive_avg_pool2d(act, (1, 1)).flatten(1)
    gmp = F.adaptive_max_pool2d(act, (1, 1)).flatten(1)
    return torch.cat([gap, gmp], dim=1)


def warmup(feature_fn, loader, sample_paths):
    """Run a few throwaway batches before the real loop.

    Measured on this machine: the first passes over a new input shape run at
    ~4-5 s/image, then settle to ~0.1 s/image.  Without this warmup both the
    progress readout and the ETA are meaningless.
    """
    if not sample_paths:
        return
    batch = torch.stack([loader(p) for p in sample_paths[:BATCH_SIZE]])
    with torch.no_grad():
        for _ in range(WARMUP_BATCHES):
            pooled_features(feature_fn(batch))


# ------------------------------------------------------------------ main ----

def extract(rows, modality, feature_fn, label, loader, expected_dim):
    subset = [r for r in rows if r["modality"] == modality]
    print(f"\n[{modality}] 共 {len(subset)} 张图，使用 {label}")
    if not subset:
        return None

    t_warm = time.time()
    warmup(feature_fn, loader, [r["path"] for r in subset])
    print(f"  预热完成 ({time.time() - t_warm:.1f} 秒)")

    feats, tokens, ids = [], [], []
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, len(subset), BATCH_SIZE):
            chunk = subset[start:start + BATCH_SIZE]
            batch = torch.stack([loader(r["path"]) for r in chunk])
            fmap = feature_fn(batch)
            out = pooled_features(fmap)
            feats.append(out.numpy().astype(np.float32))
            # (B, C, H, W) -> (B, H*W, C)
            tok = fmap.flatten(2).transpose(1, 2).contiguous()
            tokens.append(tok.numpy().astype(np.float16))
            ids.extend(r["patient_id"] for r in chunk)

            done = min(start + BATCH_SIZE, len(subset))
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            eta = (len(subset) - done) / rate if rate else 0
            print(f"  {done}/{len(subset)}  {rate:5.1f} 张/秒  剩余 {eta/60:4.1f} 分钟",
                  end="\r", flush=True)

    print()
    features = np.concatenate(feats, axis=0)
    assert features.shape[1] == expected_dim, \
        f"特征维度 {features.shape[1]} != 预期 {expected_dim}"
    token_arr = np.concatenate(tokens, axis=0)
    print(f"  池化特征 {features.shape}, 空间 token {token_arr.shape}")
    return np.asarray(ids, dtype=np.int64), features, token_arr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Cheap insurance: denormal floats can cost 10-100x inside convolutions.
    torch.set_flush_denormal(True)
    print(f"torch 线程数: {torch.get_num_threads()} | flush_denormal: 已开启")

    print("=" * 68)
    print("扫描图像目录")
    index = scan_image_index(IMG_ROOT)
    n_dr = sum(1 for k in index if k[0] == "DR")
    n_us = sum(1 for k in index if k[0] == "US")
    print(f"  找到 DR {n_dr} 张, 超声 {n_us} 张")

    print("\n构建 manifest")
    rows, stats = build_manifest(index)
    mismatch = sum(1 for r in rows if not r["label_matches_folder"])
    print(f"  可用样本 {len(rows)} 条 (DR {stats['DR']}, 超声 {stats['US']}, 缺图 {stats['missing']})")
    print(f"  CSV 标签与文件夹标签不一致: {mismatch} 条")
    if mismatch:
        print("  !! 需要人工确认这些样本，脚本会继续但请检查")

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  已写出 {manifest_path}")

    print("\n构建编码器")
    dr_encoder = build_dr_encoder()
    print("  DR 编码器就绪 (DenseNet121 + CheXpert, torchvision 实现)")

    results = {}
    dr_out = extract(rows, "DR", dr_encoder, "DenseNet121.features", load_dr_tensor, DR_FEAT_DIM)
    if dr_out is not None:
        results["dr"] = dr_out

    # Free the DR encoder before loading the next one; this box is memory tight.
    del dr_encoder
    gc.collect()

    us_encoder, us_label = build_us_encoder()
    print("\n  US 编码器就绪 (ConvNeXt-Tiny + ImageNet)")
    us_out = extract(rows, "US", us_encoder, us_label, load_us_tensor, US_FEAT_DIM)
    if us_out is not None:
        results["us"] = us_out

    print("\n保存特征")
    for name, (ids, features, tokens) in results.items():
        path = os.path.join(OUT_DIR, f"features_{name}.npz")
        np.savez_compressed(path, patient_id=ids, features=features, tokens=tokens)
        size_mb = os.path.getsize(path) / 1048576
        print(f"  {path}  features{features.shape} tokens{tokens.shape}  {size_mb:.1f} MB")

    print("\n完成")


if __name__ == "__main__":
    sys.exit(main())
