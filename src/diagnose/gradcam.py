"""Grad-CAM on both image encoders, to check what the model actually looks at.

The course requirement is explicit: confirm the model attends to lung or
thoracic regions rather than to hospital names, dates and device labels
printed in the image corners.  "Model reads the watermark" is a classic
medical-imaging failure, so this is a gate on trusting anything else.

Method: the frozen encoder is kept, a small linear probe is fitted on the
cached features, and Grad-CAM takes the gradient of the predicted-class logit
with respect to the last block's token activations.

    weights = mean over tokens of d(logit) / d(activation)
    cam     = ReLU(sum_t weights_t * activation_t)   -> reshaped to the grid
"""

from __future__ import annotations

import csv
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIR = r"E:\模型\mm4class\features"
OUT_DIR = r"E:\模型\mm4class\results\gradcam"
RADDINO_DIR = r"E:\模型\torch_cache\rad-dino"
USFMAE_CKPT = (r"E:\模型\torch_cache\usfmae"
               r"\USF-MAE_full_pretrain_43dataset_100epochs.pt")

CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]
DR_SIZE = 518
US_SIZE = 224
MEAN_RAD = 0.5307
STD_RAD = 0.2583
MEAN_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------------------------------------------- encoders ----

def build_dr_encoder():
    from transformers import Dinov2Model

    model = Dinov2Model.from_pretrained(RADDINO_DIR, local_files_only=True)
    model.eval()
    return model


def build_us_encoder():
    import timm

    model = timm.create_model("vit_base_patch16_224", pretrained=False,
                              num_classes=0, img_size=US_SIZE)
    ckpt = torch.load(USFMAE_CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    enc = {k: v for k, v in state.items()
           if not k.startswith(("decoder", "mask_token"))}
    model.load_state_dict(enc, strict=False)
    model.eval()
    return model


# --------------------------------------------------------- probe wrapper ----

class ProbedEncoder(nn.Module):
    """Frozen encoder + linear probe, with the token grid exposed for CAM."""

    def __init__(self, encoder, kind: str, dim: int):
        super().__init__()
        self.encoder = encoder
        self.kind = kind
        self.head = nn.Linear(dim, 4)
        self.activations = None

    def encode(self, pixel_values):
        """Return (cls_token, patch_tokens) with gradients off the encoder."""
        if self.kind == "dinov2":
            out = self.encoder(pixel_values=pixel_values)
            hidden = out.last_hidden_state          # (B, 1+N, 768)
            cls = hidden[:, 0]
            patch = hidden[:, 1:]
        else:                                        # timm ViT
            hidden = self.encoder.forward_features(pixel_values)
            cls = hidden[:, 0]
            patch = hidden[:, 1:]
        return cls, patch

    def forward(self, pixel_values):
        cls, patch = self.encode(pixel_values)
        # The encoder is frozen, so `patch` comes out as a leaf with no grad_fn.
        # Marking it as requiring grad is enough to get d(logit)/d(patch) --
        # nothing upstream of it needs to be differentiated for CAM.
        # `.detach()` first: a plain slice is still a view of the encoder
        # output and PyTorch then refuses to fill in its .grad.
        patch = patch.detach().requires_grad_(True)
        self.activations = patch
        pooled = torch.cat([cls, patch.mean(dim=1)], dim=1)
        return self.head(pooled)


def fit_probe(model: ProbedEncoder, features: np.ndarray, y: np.ndarray,
              epochs: int = 300, lr: float = 0.05):
    """Fit only the linear head on cached features (encoder stays frozen)."""
    import torch

    for p in model.encoder.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.head.parameters(), lr=lr, weight_decay=1e-3)
    X = torch.from_numpy(features).float()
    Y = torch.from_numpy(y).long()
    counts = np.bincount(y, minlength=4).astype(np.float32)
    w = torch.from_numpy((counts.sum() / (4 * np.maximum(counts, 1))).astype(np.float32))
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model.head(X), Y, weight=w)
        loss.backward()
        opt.step()
    return model


# ---------------------------------------------------------------- CAM ----

def grad_cam(model: ProbedEncoder, image: torch.Tensor, class_idx: int,
             grid: int) -> np.ndarray:
    model.eval()
    model.zero_grad(set_to_none=True)
    x = image.unsqueeze(0).clone().requires_grad_(False)
    logits = model(x)
    logits[0, class_idx].backward()

    # `model.activations[0]` is a view, and views never carry .grad, so the
    # gradient has to be read off the whole tensor and indexed afterwards.
    act_all = model.activations
    if act_all.grad is None:
        raise RuntimeError("没有拿到梯度")
    act = act_all[0]                                 # (N, C)
    grad = act_all.grad[0]                           # (N, C)
    weights = grad.mean(dim=0)                       # (C,)
    cam = F.relu((act * weights).sum(dim=-1))        # (N,)
    cam = cam.detach().numpy().reshape(grid, grid)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def load_dr(path):
    from PIL import Image

    img = Image.open(path).convert("L").resize((DR_SIZE, DR_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN_RAD) / STD_RAD
    return torch.from_numpy(np.stack([arr] * 3, axis=0)), np.asarray(img)


def load_us(path):
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((US_SIZE, US_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ten = torch.from_numpy(((arr - MEAN_IMAGENET) / STD_IMAGENET).transpose(2, 0, 1))
    return ten, np.asarray(img.convert("L"))


def overlay(gray: np.ndarray, cam: np.ndarray, size: int) -> np.ndarray:
    cam_t = torch.from_numpy(cam)[None, None].float()
    up = F.interpolate(cam_t, size=(size, size), mode="bilinear",
                       align_corners=False)[0, 0].numpy()
    base = np.stack([gray] * 3, axis=-1).astype(np.float32) / 255.0
    heat = np.zeros_like(base)
    heat[..., 0] = up
    heat[..., 1] = np.clip(up * 2 - 1, 0, 1) * 0.8
    alpha = (up ** 1.5)[..., None] * 0.65
    return np.clip(base * (1 - alpha) + heat * alpha, 0, 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = list(csv.DictReader(
        open(os.path.join(FEATURE_DIR, "manifest_dual.csv"), encoding="utf-8-sig")))

    # ---------- DR ----------
    print("=== DR: RAD-DINO ===")
    rd = np.load(os.path.join(FEATURE_DIR, "features_dr_raddino.npz"))
    dr_ids = list(rd["patient_id"])
    dr_feats = rd["features"]
    dr_manifest = [r for r in rows if int(r["has_dr"]) == 1]
    dr_y = np.array([int(r["label"]) for r in dr_manifest])
    pos = {int(p): i for i, p in enumerate(dr_ids)}
    order = np.array([pos[int(r["patient_id"])] for r in dr_manifest])
    feats = dr_feats[order]

    dr_model = ProbedEncoder(build_dr_encoder(), "dinov2", 1536)
    fit_probe(dr_model, feats, dr_y)

    grid_dr = DR_SIZE // 14
    panel = []
    cams_dr = {}
    for cls_idx, cls_name in enumerate(CLASSES):
        picks = [r for r, y in zip(dr_manifest, dr_y) if y == cls_idx][:3]
        for r in picks:
            ten, gray = load_dr(r["dr_path"])
            try:
                cam = grad_cam(dr_model, ten, cls_idx, grid_dr)
            except RuntimeError as exc:
                print("   跳过", r["patient_id"], exc)
                continue
            cams_dr[f"{cls_name}_{r['patient_id']}"] = cam
            panel.append((cls_name, int(r["patient_id"]), overlay(gray, cam, DR_SIZE)))
            print(f"   DR {cls_name:<4} {r['patient_id']} 完成")

    save_panel(panel, "gradcam_DR.png", DR_SIZE)
    np.savez_compressed(os.path.join(OUT_DIR, "cams_dr.npz"), **cams_dr)

    # ---------- ultrasound ----------
    print("\n=== 超声: USF-MAE ===")
    us = np.load(os.path.join(FEATURE_DIR, "features_us_all.npz"))
    us_ids = list(us["patient_id"])
    us_pos = {int(p): i for i, p in enumerate(us_ids)}
    us_manifest = [r for r in rows if int(r["has_us"]) == 1]
    us_y = np.array([int(r["label"]) for r in us_manifest])
    us_order = np.array([us_pos[int(r["patient_id"])] for r in us_manifest])
    us_feats = us["features"][us_order]

    us_model = ProbedEncoder(build_us_encoder(), "timm", 1536)
    fit_probe(us_model, us_feats, us_y)

    grid_us = US_SIZE // 16
    panel_us = []
    cams_us = {}
    for cls_idx, cls_name in enumerate(CLASSES):
        picks = [r for r, y in zip(us_manifest, us_y) if y == cls_idx][:3]
        for r in picks:
            ten, gray = load_us(r["us_path"])
            try:
                cam = grad_cam(us_model, ten, cls_idx, grid_us)
            except RuntimeError as exc:
                print("   跳过", r["patient_id"], exc)
                continue
            cams_us[f"{cls_name}_{r['patient_id']}"] = cam
            panel_us.append((cls_name, int(r["patient_id"]),
                             overlay(gray, cam, US_SIZE)))
            print(f"   超声 {cls_name:<4} {r['patient_id']} 完成")

    save_panel(panel_us, "gradcam_US.png", US_SIZE)
    np.savez_compressed(os.path.join(OUT_DIR, "cams_us.npz"), **cams_us)


def save_panel(panel, filename, size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not panel:
        print("  无可用样本")
        return
    cols = max(len([p for p in panel if p[0] == c]) for c in CLASSES)
    fig, axes = plt.subplots(4, cols, figsize=(3 * cols, 12))
    if cols == 1:
        axes = axes[:, None]
    for ax in axes.ravel():
        ax.axis("off")
    for i, (cls_name, pid, img) in enumerate(panel):
        row = CLASSES.index(cls_name)
        col = sum(1 for p in panel[:i] if p[0] == cls_name)
        ax = axes[row, col]
        ax.imshow(img)
        ax.set_title(f"{cls_name}  {pid}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Grad-CAM: 红色区域为模型决策依据", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存 {path}")


if __name__ == "__main__":
    main()
