"""Modality x class confound analysis (the P0 check from the decision doc).

Question: is the choice of imaging modality entangled with the label?

Every sample uses DR when a DR exists and falls back to ultrasound otherwise,
so "modality" is really encoding "does this patient have a chest X-ray".
If ARDS babies disproportionately lack a DR (plausible: sicker infants get
bedside ultrasound), then a model can learn "ultrasound -> ARDS" without ever
learning a radiographic sign.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

FEATURE_DIR = r"E:\模型\mm4class\features"
CSV_DIR = r"E:\模型\PythonProject"
CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = chi2_contingency(table.to_numpy())[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


print("=" * 72)
print("模态 × 类别 交叉表")
print("=" * 72)

manifest = pd.read_csv(os.path.join(FEATURE_DIR, "manifest.csv"))
# Use the CSV label (clinical ground truth), not the folder name: 25 records
# sit in a folder that disagrees with their label.
manifest["truth"] = manifest["label"].map(lambda i: CLASSES[int(i)])
manifest["truth"] = pd.Categorical(manifest["truth"], categories=CLASSES, ordered=True)
manifest["folder_class"] = pd.Categorical(
    manifest["class"], categories=CLASSES, ordered=True
)

print(f"\nCSV 标签分布:    {dict(manifest['truth'].value_counts().reindex(CLASSES))}")
print(f"文件夹类别分布:  {dict(manifest['folder_class'].value_counts().reindex(CLASSES))}")
print(f"两者不一致:      {int((manifest['truth'] != manifest['folder_class']).sum())} 条")

print(f"\n记录总数: {len(manifest)}   唯一住院号: {manifest['patient_id'].nunique()}")

# ---------------------------------------------------------------- counts ----
ct = pd.crosstab(manifest["modality"], manifest["truth"])
print("\n--- 计数 ---")
print(ct.to_string())

row_pct = pd.crosstab(manifest["modality"], manifest["truth"], normalize="index") * 100
print("\n--- 行百分比 (每种模态内部的类别构成) ---")
print(row_pct.round(1).to_string())

col_pct = pd.crosstab(manifest["modality"], manifest["truth"], normalize="columns") * 100
print("\n--- 列百分比 (每个类别内部的模态构成) ---")
print(col_pct.round(1).to_string())

# ------------------------------------------------------------ statistics ----
chi2, p, dof, expected = chi2_contingency(ct.to_numpy())
v = cramers_v(ct)
print("\n--- 独立性检验 ---")
print(f"  chi2 = {chi2:.2f}  dof = {dof}  p = {p:.4f}")
print(f"  Cramer's V = {v:.3f}   (0 = 无关, 1 = 完全相关)")
print(f"  最小期望频数 = {expected.min():.2f}  (<5 时卡方检验不可靠)")

# ------------------------------------------- modality-only "model" AUROC ----
print("\n--- 只用模态标签能预测到什么程度 ---")
print("（把 modality 当成唯一特征，one-hot 后取每类的分数，等于直接用先验）")
prior = ct.to_numpy() / ct.to_numpy().sum(axis=1, keepdims=True)
mod_index = {m: i for i, m in enumerate(ct.index)}
label_index = {c: i for i, c in enumerate(CLASSES)}

scores = np.array([prior[mod_index[m]] for m in manifest["modality"]])
y = np.array([label_index[c] for c in manifest["truth"]])

print(f"  macro-AUROC = {roc_auc_score(y, scores, multi_class='ovr', average='macro'):.3f}")
for k, name in enumerate(CLASSES):
    binary = (y == k).astype(int)
    auc = roc_auc_score(binary, scores[:, k])
    print(f"    {name:<6} AUROC = {auc:.3f}")

# -------------------------------------------------------- per split view ----
print("\n--- 各划分的模态构成 ---")
print(pd.crosstab(manifest["split"], manifest["modality"]).to_string())
print("\n--- 各划分的类别构成 ---")
print(pd.crosstab(manifest["split"], manifest["truth"]).to_string())

# ---------------------------------------------- availability (the cause) ----
print("\n" + "=" * 72)
print("根本原因：为什么会有这种模态分布")
print("=" * 72)

index = {}
for folder in os.listdir(r"E:\学习资料\第一个项目\胎肺数据\威宁妇幼数据_clahe"):
    full = os.path.join(r"E:\学习资料\第一个项目\胎肺数据\威宁妇幼数据_clahe", folder)
    if not os.path.isdir(full):
        continue
    modality = "DR" if folder.strip().upper().startswith("DR") else "US"
    for fname in os.listdir(full):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        try:
            pid = int(stem)
        except ValueError:
            continue
        index.setdefault(pid, set()).add(modality)

avail = pd.Series({pid: "".join(sorted(v)) for pid, v in index.items()}, name="available")
rec = pd.DataFrame({"patient_id": manifest["patient_id"], "class": manifest["truth"]})
rec["modality"] = manifest["modality"].to_numpy()
rec = rec.join(avail, on="patient_id")

print("\n--- 每个类别中，患者拥有哪些模态 ---")
print(pd.crosstab(rec["class"], rec["available"]).to_string())

both = rec[rec["available"] == "DRUS"]["patient_id"].nunique()
only_dr = rec[rec["available"] == "DR"]["patient_id"].nunique()
only_us = rec[rec["available"] == "US"]["patient_id"].nunique()
print(f"\n  只有 DR: {only_dr} 人")
print(f"  只有超声: {only_us} 人")
print(f"  两者都有: {both} 人")

print("\n--- 只有超声的患者，类别分布 ---")
only_us_rows = rec[rec["available"] == "US"]
print(only_us_rows["class"].value_counts().to_string())
print("\n--- 只有 DR 的患者，类别分布 ---")
print(rec[rec["available"] == "DR"]["class"].value_counts().to_string())

# ------------------------------------------------------ clean subset check ----
print("\n" + "=" * 72)
print("补救方案评估：只用同时有两种模态的患者（模态不再混杂）")
print("=" * 72)
clean = rec[rec["available"] == "DRUS"]
print(f"  可用样本数: {len(clean)}")
print("  类别分布:")
print(clean["class"].value_counts().reindex(CLASSES).to_string())
print(f"\n  ARDS 例数 {int((clean['class'] == 'ARDS').sum())} "
      f"-> {'太少, 无法支撑单独建模' if (clean['class'] == 'ARDS').sum() < 15 else '勉强可用'}")
