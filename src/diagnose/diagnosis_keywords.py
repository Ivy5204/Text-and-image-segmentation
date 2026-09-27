"""Which class does the free-text 临床诊断 actually name?

The console here is GBK and mangles Chinese, so instead of printing the text
this counts class keywords inside it -- that is the signal we actually need.
"""

import csv
import os
import sys

import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

CSV_DIR = r"E:\模型\PythonProject"
FEATURE_DIR = r"E:\模型\mm4class\features"
OUT = os.path.join(FEATURE_DIR, "label_conflicts_checked.csv")

CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]
# terms that unambiguously name a class in the free text
KEYWORDS = {
    "ARDS": ["ARDS", "呼吸窘迫"],
    "肺炎": ["肺炎"],
    "湿肺": ["湿肺"],
    "阴性": [],
}

frames = []
for split in ("train", "val", "test"):
    df = pd.read_csv(os.path.join(CSV_DIR, f"{split}.csv"))
    df["__split"] = split
    frames.append(df)
allc = pd.concat(frames, ignore_index=True)

manifest = list(csv.DictReader(
    open(os.path.join(FEATURE_DIR, "manifest.csv"), encoding="utf-8-sig")))

rows = []
for i, m in enumerate(manifest):
    csv_label = CLASSES[int(m["label"])]
    folder_label = m["class"]
    if csv_label == folder_label:
        continue
    text = str(allc.iloc[i].get("临床诊断", "") or "")
    hits = {c: sum(1 for kw in kws if kw in text)
            for c, kws in KEYWORDS.items()}
    named = [c for c, n in hits.items() if n > 0]
    rows.append({
        "patient": int(m["patient_id"]),
        "split": m["split"],
        "csv": csv_label,
        "folder": folder_label,
        "modality": m["modality"],
        "text_names": "|".join(named) if named else "(none)",
        "text_len": len(text),
        "text": text,
    })


def ascii_safe(s):
    return str(s).encode("ascii", "replace").decode("ascii")


print(f"冲突 {len(rows)} 条\n")
print(f"{'patient':>8} {'split':<6} {'csv':<6} {'folder':<6} "
      f"{'modality':<4} 诊断文本提到的类别")
print("-" * 78)
for r in rows:
    print(f"{r['patient']:>8} {r['split']:<6} {ascii_safe(r['csv']):<6} "
          f"{ascii_safe(r['folder']):<6} {r['modality']:<4} "
          f"{ascii_safe(r['text_names'])}")

print()
print("=" * 78)
print("按冲突类型汇总：诊断文本支持哪一方")
print("=" * 78)
from collections import Counter
summary = {}
for r in rows:
    key = (r["csv"], r["folder"])
    summary.setdefault(key, Counter())
    named = r["text_names"].split("|")
    if r["csv"] in named:
        summary[key]["支持 CSV"] += 1
    if r["folder"] in named:
        summary[key]["支持文件夹"] += 1
    if r["csv"] in named and r["folder"] in named:
        summary[key]["两者都提到"] += 1
    if r["text_names"] == "(none)":
        summary[key]["文本无线索"] += 1

for (c, f), cnt in sorted(summary.items(), key=lambda kv: -sum(kv[1].values())):
    total = sum(cnt.values())
    print(f"\nCSV={ascii_safe(c)} -> 文件夹={ascii_safe(f)}  (共 {total} 条)")
    for k, v in cnt.items():
        print(f"    {k:<14} {v}")

with open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n已写出 {OUT}（含完整诊断文本，用 Excel 打开即可读）")
