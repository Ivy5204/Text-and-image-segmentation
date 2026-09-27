"""Training loop for the fusion network, plus array assembly helpers."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from mm_models import (
    MODALITY_ID, POOLED_DIM, TOKEN_DIM, FusionConfig, FusionNet,
    TabularPreprocessor, effective_number_weights, focal_cross_entropy, infonce,
)

MAX_POOL_DIM = max(POOLED_DIM.values())
MAX_TOKEN_DIM = max(TOKEN_DIM.values())
TOKEN_GRID = 4        # pooled down to 4x4 before the fusion transformer


def _pool_tokens(tokens: np.ndarray, grid: int) -> np.ndarray:
    """(N, src*src, C) -> (N, grid*grid, C) via adaptive average pooling.

    The source grid is inferred from the token count, so encoders with
    different output strides can be mixed: the 224px DR/US features give
    7x7 = 49 tokens, the 320px DR features give 10x10 = 100.
    """
    n, ntok, c = tokens.shape
    src = int(round(ntok ** 0.5))
    if src * src != ntok:
        raise ValueError(f"token 数 {ntok} 不是完全平方数, 无法推断网格")
    if src == grid:
        return tokens
    x = torch.from_numpy(tokens).permute(0, 2, 1).reshape(n, c, src, src)
    x = torch.nn.functional.adaptive_avg_pool2d(x, (grid, grid))
    return x.flatten(2).permute(0, 2, 1).contiguous().numpy()

def build_arrays(cohort, token_grid: int = TOKEN_GRID):
    """Pad per-modality features into dense arrays.

    DR and US produce different widths (2048/1536 pooled, 1024/768 per token).
    The model projects each modality with its own layer and ignores the zero
    padding, so a single dense tensor keeps the batching simple.
    """
    n = len(cohort)
    n_tokens = token_grid * token_grid
    pooled = np.zeros((n, MAX_POOL_DIM), dtype=np.float32)
    # float32 up front: casting per batch cost more than it saved, and the
    # whole array is only ~176 MB.
    tokens = np.zeros((n, n_tokens, MAX_TOKEN_DIM), dtype=np.float32)
    mod_id = np.zeros(n, dtype=np.int64)

    for i in range(n):
        m = cohort.modality[i]
        pooled[i, :POOLED_DIM[m]] = cohort.pooled[i]
        tokens[i, :, :TOKEN_DIM[m]] = _pool_tokens(
            cohort.tokens[i][None], token_grid
        )[0]
        mod_id[i] = MODALITY_ID[m]
    return pooled, tokens, mod_id


def _batch(pooled, tokens, tab, mod, idx, device):
    return (
        torch.from_numpy(pooled[idx]).to(device),
        torch.from_numpy(tokens[idx]).to(device),
        torch.from_numpy(tab[idx]).to(device),
        torch.from_numpy(mod[idx]).to(device),
    )


def train_fusion(
    train_arrays,
    test_arrays,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: FusionConfig,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Train on one fold and return averaged test probabilities.

    Predictions are averaged over the last ``snapshot_last`` epochs instead of
    early-stopping on the test fold, which would leak.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pooled_tr, tokens_tr, tab_tr, mod_tr = train_arrays
    pooled_te, tokens_te, tab_te, mod_te = test_arrays

    prep = TabularPreprocessor().fit(tab_tr)
    tab_tr_p = prep.transform(tab_tr)
    tab_te_p = prep.transform(tab_te)

    model = FusionNet(tab_dim=tab_tr_p.shape[1], cfg=cfg).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.epochs
    )
    weights = effective_number_weights(y_train).to(device)

    n = len(y_train)
    batch_size = min(cfg.batch_size, n)
    snapshot: List[Dict[str, np.ndarray]] = []

    for epoch in range(cfg.epochs):
        model.train()
        order = np.random.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            # BatchNorm needs more than one sample.
            if len(idx) < 2:
                continue
            args = _batch(pooled_tr, tokens_tr, tab_tr_p, mod_tr, idx, device)
            target = torch.from_numpy(y_train[idx]).to(device)

            out = model(*args, modality_dropout=True)
            loss = focal_cross_entropy(
                out["logits"], target, weight=weights, gamma=cfg.focal_gamma
            )
            if cfg.use_aux:
                loss = loss + cfg.aux_weight * (
                    focal_cross_entropy(out["aux_img"], target, weight=weights,
                                        gamma=cfg.focal_gamma)
                    + focal_cross_entropy(out["aux_tab"], target, weight=weights,
                                          gamma=cfg.focal_gamma)
                )
            if cfg.use_infonce:
                loss = loss + cfg.infonce_weight * infonce(
                    out["z_img"], out["z_tab"]
                )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            epoch_loss += float(loss) * len(idx)
        scheduler.step()

        if epoch >= cfg.epochs - cfg.snapshot_last:
            model.eval()
            with torch.no_grad():
                out = model(*_batch(pooled_te, tokens_te, tab_te_p, mod_te,
                                    np.arange(len(y_test)), device))
                snapshot.append({
                    "proba": torch.softmax(out["logits"], -1).cpu().numpy(),
                    "gate": out["gate"].squeeze(-1).cpu().numpy(),
                })
        if verbose:
            print(f"    epoch {epoch+1}/{cfg.epochs} loss={epoch_loss/n:.4f}")

    proba = np.mean([s["proba"] for s in snapshot], axis=0)
    gate = np.mean([s["gate"] for s in snapshot], axis=0)
    return {"proba": proba, "gate": gate}
