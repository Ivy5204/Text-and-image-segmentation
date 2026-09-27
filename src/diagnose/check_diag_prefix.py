"""Does the free-text diagnosis lead with ARDS, and does that resolve the conflict?"""

import csv
import os

import pandas as pd

CSV_DIR = r"E:\模型\PythonProject"
FEATURE_DIR = r"E:\模型\mm4class\features"
CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]

frames = []
for split in ("train", "val", "test"):
    df = pd.read_csv(os.path.join(CSV_DIR, f"{split}.csv"))
    df["__split"] = split
    frames.append(df)
allc = pd.concat(frames, ignore_index=True)
manifest = list(csv.DictReader(
    open(os.path.join(FEATURE_DIR, "manifest.csv"), encoding="utf-8-sig")))

ARDS = "ARDS"
PNA = "\u80ba\u708e"
TTN = "\u6e7f\u80ba"


def head(text, n=6):
    return "".join(f"\\u{ord(c):04x}" for c in text[:n])


def a(s):
    return str(s).encode("ascii", "replace").decode("ascii")


rows = []
for i, m in enumerate(manifest):
    csv_label = CLASSES[int(m["label"])]
    folder_label = m["class"]
    if csv_label == folder_label:
        continue
    text = str(allc.iloc[i].get("临床诊断", "") or "")
    rows.append({
        "patient": int(m["patient_id"]),
        "split": m["split"],
        "csv": csv_label,
        "folder": folder_label,
        "starts_with_ARDS": text.startswith(ARDS),
        "has_pna": PNA in text,
        "has_ttn": TTN in text,
        "head_escaped": head(text),
        "text": text,
    })

print(f"冲突 {len(rows)} 条\n")
print(f"{'patient':>8} {'csv':<6} {'folder':<6} {'首词ARDS':<9} "
      f"{'含肺炎':<7} {'含湿肺':<7} 前6字")
print("-" * 92)
for r in rows:
    print(f"{r['patient']:>8} {a(r['csv']):<6} {a(r['folder']):<6} "
          f"{str(r['starts_with_ARDS']):<9} {str(r['has_pna']):<7} "
          f"{str(r['has_ttn']):<7} {r['head_escaped']}")

n_starts = sum(r["starts_with_ARDS"] for r in rows)
print(f"\n以 ARDS 开头的: {n_starts}/{len(rows)}")

print()
print("=" * 92)
print("若规则为『以 ARDS 开头即 ARDS』，标签变化：")
print("=" * 92)
base = {c: sum(1 for m in manifest if int(m["label"]) == CLASSES.index(c))
        for c in CLASSES}
converted = dict(base)
for r in rows:
    if r["starts_with_ARDS"]:
        converted[r["csv"]] -= 1
        converted["ARDS"] += 1
print(f"  当前 CSV 标签:   {base}")
print(f"  按诊断文本重标:  {converted}")
print(f"  ARDS: {base['ARDS']} -> {converted['ARDS']} 例")

out = os.path.join(FEATURE_DIR, "label_conflicts_text.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n已写出 {out}（完整文本，Excel 打开可读）")
