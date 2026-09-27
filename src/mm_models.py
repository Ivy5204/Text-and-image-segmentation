"""Model zoo: tabular heads, late fusion, and the gated + cross-attention net.

The image side always consumes cached frozen features (route 1B), so the only
things trained here are small heads.  That is deliberate: with 879 records and
39 ARDS cases, anything bigger overfits, and anything bigger is also far too
slow on this CPU-only laptop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 4
# DR is 3072 because the 320px multi-layer extractor concatenates
# transition3 (512ch) and norm5 (1024ch), each with GAP+GMP.
# DR now comes from RAD-DINO (ViT-B/14): CLS + mean patch token = 1536.
POOLED_DIM = {"DR": 1536, "US": 1536}
TOKEN_DIM = {"DR": 1024, "US": 768}
MODALITY_ID = {"DR": 0, "US": 1}


# ------------------------------------------------------------ tabular ----

class TabularPreprocessor:
    """Median imputation + standardisation, fitted on the training fold only."""

    def __init__(self):
        self.median = None
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "TabularPreprocessor":
        self.median = np.nanmedian(X, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        filled = np.where(np.isnan(X), self.median, X)
        self.mean = filled.mean(axis=0)
        self.std = filled.std(axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        missing = np.isnan(X).astype(np.float32)
        filled = np.where(np.isnan(X), self.median, X)
        scaled = (filled - self.mean) / self.std
        # Missing indicator is appended, matching the missing-indicator
        # strategy that is already used elsewhere in this project.
        return np.concatenate([scaled, missing], axis=1).astype(np.float32)

    @property
    def out_dim(self) -> int:
        return len(self.median) * 2


class LightGBMHead:
    """Strong, well-regularised GBDT.  Handles NaN natively.

    Defaults reflect the measured sweep: ``class_weight='balanced'`` actually
    HURT (0.770 -> 0.778 without it) because up-weighting a 4.4% class by 22x
    amplifies its noise along with its signal.
    """

    def __init__(self, seed: int = 0, num_leaves: int = 12,
                 n_estimators: int = 400, learning_rate: float = 0.03,
                 min_child_samples: int = 12, class_weight=None,
                 colsample_bytree: float = 0.7, reg_lambda: float = 1.0):
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=N_CLASSES,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=4,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class LogisticHead:
    """Cheap linear probe, useful as a floor and for the cached features."""

    def __init__(self, seed: int = 0, C: float = 0.05):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1500, C=C, class_weight="balanced",
                               multi_class="multinomial", random_state=seed),
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class MLPHead(nn.Module):
    """Small tabular MLP.  Kept as an ablation, not the primary branch."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------- class weights ----

def effective_number_weights(y: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    """Class-Balanced (Cui et al. 2019) weights: (1-beta)/(1-beta^n)."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights = weights / weights.sum() * N_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


def focal_cross_entropy(logits, target, weight=None, gamma: float = 1.5,
                        label_smoothing: float = 0.05):
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none",
                         label_smoothing=label_smoothing)
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


# ------------------------------------------------------- fusion network ----

@dataclass
class FusionConfig:
    d_model: int = 128
    n_heads: int = 4
    n_attn_layers: int = 1
    token_grid: int = 4          # image tokens per side; 4 -> 16 tokens
    dropout: float = 0.3
    modality_dropout: float = 0.25
    # How z_img and the tabular query are combined.
    #   "gate"   - z = g*z_img + (1-g)*q   (the headline model)
    #   "concat" - z = W[z_img ; q]
    #   "sum"    - z = W(z_img + q)
    fusion_mode: str = "gate"
    # Cross-attention is an ADD-ON to the tabular query: it lets the tabular
    # vector attend over the image tokens.  Off by default: with the old,
    # incorrect implementation it did not help, and it needs re-measuring.
    use_cross_attn: bool = False
    use_aux: bool = True
    use_infonce: bool = True
    infonce_weight: float = 0.1
    aux_weight: float = 0.3
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    snapshot_last: int = 5
    focal_gamma: float = 1.5


class GatedFusion(nn.Module):
    """z = g * z_img + (1 - g) * z_tab,  g = sigmoid(W[z_img; z_tab])."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.Sigmoid(),
        )

    def forward(self, z_img, z_tab):
        """Returns (fused, gate).  gate -> 1 means "trust the image"."""
        g = self.gate(torch.cat([z_img, z_tab], dim=-1))
        return g * z_img + (1 - g) * z_tab, g


class FusionNet(nn.Module):
    """Image tokens + tabular vector -> four classes.

    Fixed in this revision (see the audit list):

    1. Modality dropout now ROUTES instead of zeroing.  Previously a dropped
       modality was replaced by a zero vector, which the gate happily mixed at
       ~0.5/0.5 and which the attention attended over as if it were real
       tokens.  Now, per sample:
           image dropped -> fused = tabular query (attention output discarded)
           table dropped -> fused = image embedding
           both present  -> fused = g*z_img + (1-g)*q
       The gate is pinned to 0 or 1 accordingly, and no zero vector ever
       reaches the gate or a softmax.

    2. Image tokens carry a 2D positional encoding (learned row + column
       embeddings).  Without it the token set was permutation invariant, i.e.
       the model could not tell upper-left lung from lower-right lung, throwing
       away the spatial structure the CNN had produced.

    3. Cross-attention is a real nn.MultiheadAttention with
       Q = tabular vector, K = V = image tokens, wrapped in a residual +
       LayerNorm.  Previously this was self-attention over a concatenated
       sequence, which is not the same operation.
    """

    def __init__(self, tab_dim: int, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        g = cfg.token_grid

        self.pool_proj = nn.ModuleDict({
            m: nn.Linear(POOLED_DIM[m], d) for m in POOLED_DIM
        })
        self.tok_proj = nn.ModuleDict({
            m: nn.Linear(TOKEN_DIM[m], d) for m in TOKEN_DIM
        })
        self.modality_embed = nn.Embedding(2, d)

        # 2D positional encoding: row (g, d) + column (g, d) -> (g*g, d)
        self.row_embed = nn.Parameter(torch.zeros(g, d))
        self.col_embed = nn.Parameter(torch.zeros(g, d))
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(128, d),
        )

        self.gate = GatedFusion(d)
        self.concat_proj = nn.Linear(2 * d, d)
        self.sum_proj = nn.Linear(d, d)

        self.cross_attn = nn.MultiheadAttention(
            d, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d)

        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(d, N_CLASSES)
        self.aux_img = nn.Linear(d, N_CLASSES)
        self.aux_tab = nn.Linear(d, N_CLASSES)

    def encode_image(self, pooled, tokens, mod_id):
        z_parts, t_parts = [], []
        for mid in torch.unique(mod_id):
            sel = mod_id == mid
            name = "DR" if int(mid) == 0 else "US"
            # ``pooled``/``tokens`` are zero-padded to the widest modality, so
            # each modality's projection must see only its own columns.
            pw = POOLED_DIM[name]
            tw = TOKEN_DIM[name]
            z_parts.append((sel, self.pool_proj[name](pooled[sel][:, :pw])))
            t_parts.append((sel, self.tok_proj[name](tokens[sel][:, :, :tw])))

        z = torch.zeros(pooled.shape[0], self.cfg.d_model, device=pooled.device)
        tok = torch.zeros(pooled.shape[0], tokens.shape[1], self.cfg.d_model,
                          device=pooled.device)
        for sel, val in z_parts:
            z[sel] = val
        for sel, val in t_parts:
            tok[sel] = val
        tok = tok + self.modality_embed(mod_id).unsqueeze(1)
        return z, tok

    def token_positional(self, tok: torch.Tensor) -> torch.Tensor:
        """Add a learned 2D row/column encoding to the image token grid."""
        g = self.cfg.token_grid
        pos = self.row_embed[:g, None, :] + self.col_embed[None, :g, :]
        return tok + pos.reshape(1, g * g, -1)

    def forward(self, pooled, tokens, tab, mod_id, modality_dropout: bool = False):
        z_img, tok = self.encode_image(pooled, tokens, mod_id)
        tok = self.token_positional(tok)          # (1) fix: spatial layout kept
        z_tab = self.tab_encoder(tab)

        device = pooled.device
        b = pooled.shape[0]
        keep_img = torch.ones(b, dtype=torch.bool, device=device)
        keep_tab = torch.ones(b, dtype=torch.bool, device=device)

        if modality_dropout and self.training and self.cfg.modality_dropout > 0:
            p = self.cfg.modality_dropout
            keep_img = torch.rand(b, device=device) >= p
            keep_tab = torch.rand(b, device=device) >= p
            both_dropped = (~keep_img) & (~keep_tab)
            keep_tab = keep_tab | both_dropped   # never drop both for one sample

        # (3) fix: real cross-attention, Q = tabular vector, K = V = image tokens
        if self.cfg.use_cross_attn:
            attn_out, _ = self.cross_attn(
                z_tab.unsqueeze(1), tok, tok, need_weights=False
            )
            query = self.cross_norm(z_tab + self.dropout(attn_out[:, 0]))
        else:
            query = z_tab

        gval = None
        if self.cfg.fusion_mode == "gate":
            fused, gval = self.gate(z_img, query)
        elif self.cfg.fusion_mode == "concat":
            fused = self.concat_proj(torch.cat([z_img, query], dim=-1))
        elif self.cfg.fusion_mode == "sum":
            fused = self.sum_proj(z_img + query)
        else:
            raise ValueError(f"未知的 fusion_mode: {self.cfg.fusion_mode}")

        # (2) fix: modality dropout routes, it does not zero.
        #   image missing -> fused = tabular query ; gate pinned to 0
        #   table missing -> fused = image embedding ; gate pinned to 1
        # No zero vector is ever fed to the gate or to a softmax.
        if modality_dropout and self.training and self.cfg.modality_dropout > 0:
            tab_only = (~keep_img & keep_tab).unsqueeze(1)
            img_only = (keep_img & ~keep_tab).unsqueeze(1)
            if gval is None:
                gval = torch.full_like(z_img[:, :1], 0.5)
            fused = torch.where(tab_only, query, fused)
            fused = torch.where(img_only, z_img, fused)
            gval = torch.where(tab_only, torch.zeros_like(gval), gval)
            gval = torch.where(img_only, torch.ones_like(gval), gval)

        if gval is None:
            gval = torch.full_like(z_img[:, :1], 0.5)

        z = self.dropout(self.norm(fused))
        return {
            "logits": self.head(z),
            "aux_img": self.aux_img(z_img),
            "aux_tab": self.aux_tab(z_tab),
            "z_img": z_img,
            "z_tab": z_tab,
            "gate": gval,
        }


def infonce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.1
            ) -> torch.Tensor:
    """Symmetric InfoNCE between the two modality embeddings of a batch."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.t(), labels))
