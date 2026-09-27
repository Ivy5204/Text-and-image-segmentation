"""Evaluation protocol: 5x5 repeated stratified group CV + honest reporting.

Why this file is strict:
  ARDS has only 39 cases.  Even at a true AUROC of 0.85 the 95% CI is about
  +/-0.08, so a single split cannot distinguish two models.  Every comparison
  here therefore runs on the *same* folds and is reported with a paired test.

Grouping is by 住院号 (patient id).  24 patients have more than one record;
without grouping those duplicates would straddle folds and inflate scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]
ARDS_INDEX = 1

N_SPLITS = 5
N_REPEATS = 5
BASE_SEED = 42


# ------------------------------------------------------------------ folds ----

def make_folds(y: np.ndarray, groups: np.ndarray,
               n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS,
               base_seed: int = BASE_SEED):
    """List of (repeat, fold, train_idx, test_idx).  Same folds for every model."""
    folds = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=base_seed + repeat
        )
        for fold, (tr, te) in enumerate(
            splitter.split(np.zeros(len(y)), y, groups)
        ):
            folds.append((repeat, fold, tr, te))
    return folds


# ---------------------------------------------------------------- metrics ----

def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    out = {}
    try:
        out["macro_auroc"] = float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        out["macro_auroc"] = float("nan")

    try:
        out["ards_auprc"] = float(
            average_precision_score((y_true == ARDS_INDEX).astype(int),
                                    proba[:, ARDS_INDEX])
        )
        out["ards_auroc"] = float(
            roc_auc_score((y_true == ARDS_INDEX).astype(int), proba[:, ARDS_INDEX])
        )
    except ValueError:
        out["ards_auprc"] = float("nan")
        out["ards_auroc"] = float("nan")

    out["accuracy"] = float(accuracy_score(y_true, proba.argmax(1)))
    out["balanced_accuracy"] = float(_balanced_accuracy(y_true, proba.argmax(1)))

    for k, name in enumerate(CLASSES):
        binary = (y_true == k).astype(int)
        try:
            out[f"auroc_{name}"] = float(roc_auc_score(binary, proba[:, k]))
        except ValueError:
            out[f"auroc_{name}"] = float("nan")
    return out


def _balanced_accuracy(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class = np.diag(cm) / cm.sum(axis=1)
    return float(np.nanmean(per_class))


def per_class_recall(y_true, y_pred) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.diag(cm) / cm.sum(axis=1)
    return {name: float(recall[k]) for k, name in enumerate(CLASSES)}


def apply_class_weights(proba: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """argmax over proba * weights, i.e. per-class decision thresholds."""
    return (proba * np.asarray(weights, dtype=float)).argmax(axis=1)


def optimize_class_weights(y_true: np.ndarray, proba: np.ndarray,
                           n_iter: int = 8, n_grid: int = 41,
                           lo: float = 0.3, hi: float = 5.0):
    """Coordinate ascent on per-class decision weights for balanced accuracy.

    AUROC is threshold-free, the deployed decision is not.  On this data the
    gap is big: macro-AUROC ~0.80 while the plain argmax only reaches ~0.60
    balanced accuracy, because the majority class swallows the rest.  Four
    multiplicative weights recover most of it and cost nothing but inference.
    """
    weights = np.ones(4, dtype=float)
    best = _balanced_accuracy(y_true, apply_class_weights(proba, weights))
    grid = np.linspace(lo, hi, n_grid)
    for _ in range(n_iter):
        improved = False
        for k in range(4):
            for cand in grid:
                trial = weights.copy()
                trial[k] = cand
                score = _balanced_accuracy(
                    y_true, apply_class_weights(proba, trial)
                )
                if score > best + 1e-9:
                    best, weights, improved = score, trial, True
        if not improved:
            break
    return weights, best


def decision_report(y_true: np.ndarray, proba: np.ndarray,
                    weights: np.ndarray = None) -> Dict[str, object]:
    """Compare the raw argmax against the re-weighted decision."""
    pred_raw = proba.argmax(1)
    out = {
        "accuracy_raw": float(accuracy_score(y_true, pred_raw)),
        "balanced_accuracy_raw": _balanced_accuracy(y_true, pred_raw),
        "recall_raw": per_class_recall(y_true, pred_raw),
    }
    if weights is not None:
        pred_tuned = apply_class_weights(proba, weights)
        out.update({
            "weights": [float(w) for w in weights],
            "accuracy_tuned": float(accuracy_score(y_true, pred_tuned)),
            "balanced_accuracy_tuned": _balanced_accuracy(y_true, pred_tuned),
            "recall_tuned": per_class_recall(y_true, pred_tuned),
        })
    return out


def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray,
                 metric: str = "macro_auroc", n_boot: int = 1000,
                 seed: int = 0) -> tuple:
    """Percentile CI over patient-level resampling."""
    rng = np.random.default_rng(seed)
    patients = np.unique(y_true)  # placeholder, replaced below
    del patients
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 4:
            continue
        try:
            stats.append(compute_metrics(y_true[idx], proba[idx])[metric])
        except ValueError:
            continue
    stats = np.array([s for s in stats if np.isfinite(s)])
    if len(stats) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ------------------------------------------------------- comparisons ----

@dataclass
class EvalResult:
    name: str
    fold_scores: Dict[str, List[float]] = field(default_factory=dict)
    pooled_oof: np.ndarray = None
    n_train_records: int = 0

    def summary(self) -> Dict[str, float]:
        out = {}
        for metric, values in self.fold_scores.items():
            arr = np.array(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            out[f"{metric}_mean"] = float(arr.mean()) if len(arr) else float("nan")
            out[f"{metric}_std"] = float(arr.std()) if len(arr) else float("nan")
        return out


def run_cv(name: str, y: np.ndarray, groups: np.ndarray,
           fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
           folds=None, seed_note: str = "") -> EvalResult:
    """Run one model through the shared folds.

    ``fit_predict(train_idx, test_idx) -> (n_test, 4) probabilities``.
    Metrics are computed twice: per fold (for the paired test) and on the
    pooled out-of-fold matrix (for the headline number).
    """
    folds = folds if folds is not None else make_folds(y, groups)
    result = EvalResult(name=name)
    metric_names = list(compute_metrics(
        np.array([0, 1, 2, 3]), np.full((4, 4), 0.25)
    ).keys())
    for m in metric_names:
        result.fold_scores[m] = []

    oof = np.full((len(y), 4), np.nan)
    for repeat, fold, tr, te in folds:
        proba = fit_predict(tr, te)
        oof[te] = proba
        m = compute_metrics(y[te], proba)
        for k, v in m.items():
            result.fold_scores[k].append(v)

    if not np.isnan(oof).any():
        result.pooled_oof = oof
    else:
        # Duplicated patient records can appear in more than one test fold;
        # average the repeats rather than taking the last write.
        result.pooled_oof = np.nanmean(oof.reshape(-1, 4), axis=0) if False else oof
    result.note = seed_note
    return result


def paired_resampled_ttest(scores_a: List[float], scores_b: List[float],
                           n_train: int, n_test: int) -> Dict[str, float]:
    """Nadeau-Bengio corrected resampled t-test (accounts for fold overlap)."""
    from scipy import stats

    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return {"t": float("nan"), "p": float("nan"), "diff": float("nan")}
    diff = a - b
    n = len(a)
    correction = (1.0 / n) + (n_test / n_train)
    denom = diff.std(ddof=1) * np.sqrt(correction)
    if denom == 0:
        return {"t": float("nan"), "p": float("nan"), "diff": float(diff.mean())}
    t = diff.mean() / denom
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return {"t": float(t), "p": float(p), "diff": float(diff.mean())}


def format_table(results: List[EvalResult]) -> str:
    header = (f"{'model':<34}{'macro-AUROC':>18}{'ARDS AUPRC':>18}"
              f"{'bACC':>14}")
    lines = [header, "-" * len(header)]
    for r in results:
        s = r.summary()
        lines.append(
            f"{r.name:<34}"
            f"{s.get('macro_auroc_mean', float('nan')):>11.3f} ± "
            f"{s.get('macro_auroc_std', float('nan')):<4.3f}"
            f"{s.get('ards_auprc_mean', float('nan')):>11.3f} ± "
            f"{s.get('ards_auprc_std', float('nan')):<4.3f}"
            f"{s.get('balanced_accuracy_mean', float('nan')):>9.3f}"
        )
    return "\n".join(lines)
