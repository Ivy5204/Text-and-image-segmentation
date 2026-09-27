"""Fine-tune the last block of DenseNet121 on the newborn chest X-rays.

Why only the last block, and why a cached prefix
------------------------------------------------
This is a CPU-only laptop.  Backpropagating through all of DenseNet121 costs
roughly 40 s per 8-image batch, which makes per-fold fine-tuning impossible.

So the network is cut at ``features.denseblock4``:

    conv0 -> ... -> transition3        (frozen, run ONCE, output cached)
    denseblock4 -> norm5 -> pooling    (trained, ~16 layers on a 7x7 map)

The cached tensor is (1024, 7, 7) per image -- about 100 KB in float16 -- so
the whole DR set fits in memory and every training epoch only touches the tail.

Augmentation: because the frozen prefix is cached, augmentation cannot be
applied per epoch.  Instead K augmented copies of each training image are
pushed through the prefix once and cached too; training then samples from
those.  Test-time features always come from the clean copy.

Leakage: the tail is fine-tuned inside each CV fold, on that fold's training
records only.  The encoder never sees a held-out record.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

XRV_CACHE = r"E:\模型\torch_cache\torchxrayvision"
IMG_SIZE = 224


def _build_full_densenet():
    """DenseNet121 with the CheXpert weights (same as extract_features.py)."""
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
    return model


def _augment_tensor(img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Cheap geometric + intensity jitter on a single (1, H, W) X-ray."""
    out = img
    if rng.random() < 0.5:
        out = torch.flip(out, dims=[2])
    if rng.random() < 0.5:
        angle = float(rng.uniform(-8, 8))
        theta = torch.tensor(
            [[np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0.0],
             [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0)
        grid = F.affine_grid(theta, (1, 1, IMG_SIZE, IMG_SIZE), align_corners=False)
        out = F.grid_sample(out.unsqueeze(0), grid, align_corners=False)[0]
    if rng.random() < 0.5:
        out = out * float(rng.uniform(0.9, 1.1))
    return out.clamp(-1024.0, 1024.0)


def load_dr_image(path: str) -> torch.Tensor:
    from PIL import Image

    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy((arr * 1024.0).astype(np.float32)).unsqueeze(0)


class FineTunedDR:
    """Frozen-prefix + trainable-tail DenseNet121 for the DR images."""

    def __init__(self, n_aug: int = 3, seed: int = 0, device: str = "cpu"):
        self.device = device
        self.n_aug = n_aug
        self.seed = seed
        full = _build_full_densenet()
        children = list(full.features.children())
        self.prefix = nn.Sequential(*children[:10]).to(device).eval()
        for p in self.prefix.parameters():
            p.requires_grad_(False)
        self.tail = nn.Sequential(children[10], children[11]).to(device)
        self.cache: Optional[torch.Tensor] = None      # (n_aug+1, N, 1024, 7, 7)

    # ------------------------------------------------------------ caching ----
    def cache_prefix(self, paths: Sequence[str], log_every: int = 60) -> None:
        """Run the frozen prefix once per image (plus K augmented copies)."""
        n = len(paths)
        rng = np.random.default_rng(self.seed)
        holder = []
        # DenseNet121 channel widths: transition3 emits 512, denseblock4
        # expands that back to 1024.  Probe the real width instead of assuming.
        with torch.no_grad():
            probe = self.prefix(torch.zeros(1, 1, IMG_SIZE, IMG_SIZE,
                                            device=self.device))
            c, h, w = probe.shape[1], probe.shape[2], probe.shape[3]
        print(f"  冻结前缀输出形状: (N, {c}, {h}, {w})")
        with torch.no_grad():
            for slot in range(self.n_aug + 1):
                out = torch.empty(n, c, h, w, dtype=torch.float16)
                for i, path in enumerate(paths):
                    img = load_dr_image(path)
                    if slot > 0:
                        img = _augment_tensor(img, rng)
                    feat = self.prefix(img.unsqueeze(0).to(self.device))
                    out[i] = feat[0].to(torch.float16).cpu()
                    if log_every and (i + 1) % log_every == 0:
                        print(f"    缓存 {slot}/{self.n_aug}  {i+1}/{n}",
                              end="\r", flush=True)
                holder.append(out)
                print()
        self.cache = torch.stack(holder)               # (K+1, N, C, H, W)
        print(f"  前缀缓存完成: {tuple(self.cache.shape)} "
              f"({self.cache.numel() * 2 / 1048576:.0f} MB)")

    # ------------------------------------------------------------ training ----
    def fit(self, idx: np.ndarray, y: np.ndarray, epochs: int = 8,
            lr: float = 3e-4, batch_size: int = 16, weight_decay: float = 1e-4,
            class_weights: Optional[torch.Tensor] = None,
            verbose: bool = False) -> None:
        """Fine-tune denseblock4 + norm5 on the given records."""
        for p in self.tail.parameters():
            p.requires_grad_(True)
        self.tail.train()

        head = nn.Linear(2048, 4).to(self.device)
        params = list(self.tail.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        rng = np.random.default_rng(self.seed)

        idx = np.asarray(idx)
        y_t = torch.from_numpy(y).to(self.device)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        for epoch in range(epochs):
            order = rng.permutation(len(idx))
            total = 0.0
            for start in range(0, len(idx), batch_size):
                sel = order[start:start + batch_size]
                if len(sel) < 2:
                    continue
                rows = idx[sel]
                slots = rng.integers(0, self.n_aug + 1, size=len(sel))
                batch = self.cache[slots, rows].to(torch.float32).to(self.device)
                target = y_t[sel]

                feat = F.relu(self.tail(batch))
                pooled = torch.cat([
                    F.adaptive_avg_pool2d(feat, 1).flatten(1),
                    F.adaptive_max_pool2d(feat, 1).flatten(1),
                ], dim=1)
                loss = F.cross_entropy(head(pooled), target,
                                       weight=class_weights)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                total += float(loss) * len(sel)
            sched.step()
            if verbose:
                print(f"      epoch {epoch+1}/{epochs} loss={total/len(idx):.4f}")
        self.head = head

    # ---------------------------------------------------------- inference ----
    @torch.no_grad()
    def transform(self, idx: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """GAP+GMP features from the clean (unaugmented) cached activations."""
        self.tail.eval()
        idx = np.asarray(idx)
        out = []
        for start in range(0, len(idx), batch_size):
            rows = idx[start:start + batch_size]
            batch = self.cache[0, rows].to(torch.float32).to(self.device)
            feat = F.relu(self.tail(batch))
            pooled = torch.cat([
                F.adaptive_avg_pool2d(feat, 1).flatten(1),
                F.adaptive_max_pool2d(feat, 1).flatten(1),
            ], dim=1)
            out.append(pooled.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)
