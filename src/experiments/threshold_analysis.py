"""How much of the "accuracy" gap is just an untuned decision rule?

The fusion model reaches macro-AUROC ~0.80 but only ~0.60 balanced accuracy
with a plain argmax, because 57% of the cohort is the negative class.

Two numbers are reported:
  in-sample   tune and evaluate on the same out-of-fold predictions.  This is
              the optimistic upper bound and is NOT a valid estimate.
  split-half  tune the weights on half the patients, evaluate on the other
              half, repeat and average.  This is the honest number.
"""

import glob
import os
import sys

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mm_eval

RESULTS_DIR = r"E:\模型\mm4class\results"


def split_half(y, oof, groups, seed=0, n_rep=20):
    rng = np.random.default_rng(seed)
    patients = np.unique(groups)
    scores_raw, scores_tuned = [], []
    for _ in range(n_rep):
        perm = rng.permutation(patients)
        half = len(perm) // 2
        a, b = set(perm[:half]), set(perm[half:])
        mask_a = np.array([g in a for g in groups])
        mask_b = ~mask_a
        if len(np.unique(y[mask_a])) < 4 or len(np.unique(y[mask_b])) < 4:
            continue
        w, _ = mm_eval.optimize_class_weights(y[mask_a], oof[mask_a])
        for mask in (mask_b, mask_a):
            pred_raw = oof[mask].argmax(1)
            pred_tuned = mm_eval.apply_class_weights(oof[mask], w)
            from sklearn.metrics import balanced_accuracy_score
            scores_raw.append(balanced_accuracy_score(y[mask], pred_raw))
            scores_tuned.append(balanced_accuracy_score(y[mask], pred_tuned))
    return (float(np.mean(scores_raw)), float(np.mean(scores_tuned)),
            float(np.std(scores_tuned)))


files = sorted(glob.glob(os.path.join(RESULTS_DIR, "oof_*.npz")))
if not files:
    raise SystemExit("没有找到 oof_*.npz")

for path in files:
    z = np.load(path)
    y, oof, groups = z["y"], z["oof"], z["patient_id"]
    name = os.path.basename(path).replace("oof_", "").replace(".npz", "")
    print("=" * 74)
    print(name)
    print("=" * 74)
    print(f"  样本 {len(y)}, ARDS {int((y == 1).sum())}")

    pred_raw = oof.argmax(1)
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    acc_raw = accuracy_score(y, pred_raw)
    bacc_raw = balanced_accuracy_score(y, pred_raw)
    print(f"\n  未调阈值: accuracy {acc_raw:.3f}   平衡正确率 {bacc_raw:.3f}")

    w, best = mm_eval.optimize_class_weights(y, oof)
    pred_t = mm_eval.apply_class_weights(oof, w)
    acc_t = accuracy_score(y, pred_t)
    bacc_t = balanced_accuracy_score(y, pred_t)
    print(f"  阈值优化(同数据): accuracy {acc_t:.3f}   平衡正确率 {bacc_t:.3f}")
    print(f"    权重 {np.round(w, 2).tolist()}")
    print(f"    各类召回 原始 {mm_eval.per_class_recall(y, pred_raw)}")
    print(f"    各类召回 优化 {mm_eval.per_class_recall(y, pred_t)}")

    raw_h, tuned_h, std_h = split_half(y, oof, groups)
    print(f"\n  半数据划分（无偏估计）: 平衡正确率 {raw_h:.3f} -> {tuned_h:.3f}"
          f"  ({tuned_h - raw_h:+.3f}) ± {std_h:.3f}")
    print()
