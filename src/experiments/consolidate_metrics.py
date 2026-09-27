"""Recompute every headline metric from the saved out-of-fold predictions.

Reading them back from the .npz files (rather than quoting numbers out of
transcripts) keeps the table internally consistent: same definition of
accuracy, same class order, same treatment of the decision rule.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score,
)

RESULTS = r"E:\模型\mm4class\results"
CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]


def metrics(y, oof):
    pred = oof.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = np.diag(cm) / cm.sum(axis=1)
    spec = []
    for k in range(4):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec.append(tn / (tn + fp) if (tn + fp) else np.nan)
    return {
        "macro_auroc": roc_auc_score(y, oof, multi_class="ovr", average="macro"),
        "ards_auroc": roc_auc_score((y == 1).astype(int), oof[:, 1]),
        "ards_auprc": average_precision_score((y == 1).astype(int), oof[:, 1]),
        "acc": accuracy_score(y, pred),
        "bacc": balanced_accuracy_score(y, pred),
        "sens": sens,
        "spec": np.array(spec),
    }


GROUPS = {
    "单模态（威宁内部）": [
        "oof_table_TabPFN_v2__n_est_4_.npz",
        "oof_stack_TabPFN_表格.npz",
        "oof_model_placeholder",
    ],
}

files = sorted(glob.glob(os.path.join(RESULTS, "oof_*.npz")))
print(f"找到 {len(files)} 个 OOF 文件\n")

rows = []
for path in files:
    z = np.load(path, allow_pickle=True)
    y, oof = z["y"], z["oof"]
    if np.isnan(oof).any():
        continue
    m = metrics(y, oof)
    rows.append((os.path.basename(path)[4:-4], m, len(y),
                 int((y == 1).sum())))

hdr = ("模型", "n", "ARDS", "macroAUROC", "ARDS AUROC", "ARDS AUPRC",
       "准确率", "平衡正确率")
print(f"{hdr[0]:<34}{hdr[1]:>5}{hdr[2]:>6}{hdr[3]:>12}{hdr[4]:>12}"
      f"{hdr[5]:>12}{hdr[6]:>9}{hdr[7]:>11}")
print("-" * 101)
for name, m, n, nards in rows:
    print(f"{name[:33]:<34}{n:>5}{nards:>6}{m['macro_auroc']:>12.3f}"
          f"{m['ards_auroc']:>12.3f}{m['ards_auprc']:>12.3f}"
          f"{m['acc']:>9.3f}{m['bacc']:>11.3f}")

print()
print("=" * 101)
print("各类灵敏度（召回）")
print("=" * 101)
print(f"{'模型':<34}" + "".join(f"{c:>14}" for c in CLASSES))
for name, m, _, _ in rows:
    print(f"{name[:33]:<34}" + "".join(f"{v:>14.3f}" for v in m["sens"]))

print()
print("=" * 101)
print("各类特异度")
print("=" * 101)
print(f"{'模型':<34}" + "".join(f"{c:>14}" for c in CLASSES))
for name, m, _, _ in rows:
    print(f"{name[:33]:<34}" + "".join(f"{v:>14.3f}" for v in m["spec"]))
