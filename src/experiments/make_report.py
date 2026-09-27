"""Turn results/summary_*.json into a readable markdown report."""

from __future__ import annotations

import argparse
import json
import os

RESULTS_DIR = r"E:\模型\mm4class\results"
CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]

GROUPS = [
    ("表格单模态", ["T-only"]),
    ("图像单模态", ["I-only"]),
    ("融合", ["T+I", "FULL"]),
    ("消融", ["abl:"]),
]


def load(tag: str):
    path = os.path.join(RESULTS_DIR, f"summary_{tag}.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="quick", choices=["quick", "full"])
    args = parser.parse_args()

    data = load(args.tag)
    settings = data["settings"]
    results = data["results"]

    lines = []
    lines.append(f"# 实验结果（{args.tag}）")
    lines.append("")
    lines.append(f"- 样本数：{settings['n_samples']}")
    lines.append(f"- 折数：{settings['n_folds']}"
                 f"（{settings['repeats']} 重复 × 5 折）")
    lines.append(f"- 训练轮数：{settings['epochs']}")
    lines.append(f"- 图像 token 网格：{settings['token_grid']}×{settings['token_grid']}")
    lines.append(f"- 已完成：{settings['completed']}/{settings['planned']}")
    lines.append("")
    lines.append("> ARDS 仅 39 例，macro-AUROC 的 95% 置信区间约 ±0.08。")
    lines.append("> 差异小于该幅度时，请以配对检验的 p 值为准。")
    lines.append("")

    header = ("| 模型 | macro-AUROC | ARDS AUPRC | ARDS AUROC | bACC |"
              " 阴性 AUROC | 湿肺 AUROC | 肺炎 AUROC |")
    sep = "|" + "---|" * 8

    for title, prefixes in GROUPS:
        subset = [r for r in results
                  if any(r["name"].startswith(p) for p in prefixes)]
        if not subset:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for r in subset:
            lines.append(
                f"| {r['name']} "
                f"| {fmt(r.get('macro_auroc_mean'))} ± {fmt(r.get('macro_auroc_std'))} "
                f"| {fmt(r.get('ards_auprc_mean'))} "
                f"| {fmt(r.get('ards_auroc_mean'))} "
                f"| {fmt(r.get('balanced_accuracy_mean'))} "
                f"| {fmt(r.get('auroc_阴性_mean'))} "
                f"| {fmt(r.get('auroc_湿肺_mean'))} "
                f"| {fmt(r.get('auroc_肺炎_mean'))} |"
            )
        lines.append("")

    comparisons = data.get("comparisons_vs_full") or []
    if comparisons:
        lines.append("## 与完整模型的配对比较（macro-AUROC）")
        lines.append("")
        lines.append("| 模型 | Δmacro-AUROC | t | p | 显著 (p<0.05) |")
        lines.append("|---|---|---|---|---|")
        for row in comparisons:
            lines.append(
                f"| {row['model']} | {fmt(row['delta_macro_auroc'], 4)} "
                f"| {fmt(row['t'], 3)} | {fmt(row['p_value'], 4)} "
                f"| {'是' if row['significant_at_0.05'] else '否'} |"
            )
        lines.append("")

    out_path = os.path.join(RESULTS_DIR, f"report_{args.tag}.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"已写出 {out_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
