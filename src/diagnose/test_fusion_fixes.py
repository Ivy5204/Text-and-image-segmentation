"""Smoke test for the three fixes in FusionNet."""

import os
import sys

import torch

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mm_models import FusionConfig, FusionNet

torch.manual_seed(0)
B, TAB, NT = 6, 20, 16
pooled = torch.randn(B, 2048)
tokens = torch.randn(B, NT, 1024)
tab = torch.randn(B, TAB)
mod = torch.tensor([0, 0, 0, 1, 1, 1])

print("=== 三种融合模式 x cross-attn 开关 ===")
for mode in ("gate", "concat", "sum"):
    for ca in (False, True):
        cfg = FusionConfig(fusion_mode=mode, use_cross_attn=ca, token_grid=4)
        m = FusionNet(tab_dim=TAB, cfg=cfg).eval()
        with torch.no_grad():
            out = m(pooled, tokens, tab, mod)
        logits = tuple(out["logits"].shape)
        gmean = float(out["gate"].mean())
        print("  %-7s cross_attn=%-5s logits %s  gate mean %.3f"
              % (mode, ca, logits, gmean))

print()
print("=== 位置编码 ===")
cfg = FusionConfig(token_grid=4)
m = FusionNet(tab_dim=TAB, cfg=cfg)
print("  row_embed %s  col_embed %s"
      % (tuple(m.row_embed.shape), tuple(m.col_embed.shape)))
with torch.no_grad():
    pos = m.token_positional(torch.zeros(1, NT, 128))[0]
print("  16 个位置中互不相同的个数:", int(torch.unique(pos, dim=0).shape[0]))
print("  位置 0 与位置 1 相同?", bool(torch.allclose(pos[0], pos[1])))
print("  位置 0 与位置 5 相同?", bool(torch.allclose(pos[0], pos[5])))

print()
print("=== 模态丢弃路由 ===")
cfg = FusionConfig(fusion_mode="gate", modality_dropout=1.0)
m = FusionNet(tab_dim=TAB, cfg=cfg).train()
with torch.no_grad():
    out = m(pooled, tokens, tab, mod, modality_dropout=True)
uniq = sorted({round(float(v), 3) for v in out["gate"].flatten()})
print("  p=1.0 时 gate 的取值:", uniq, " (应只含 0.0 和 1.0)")
print("  logits 含 NaN?", bool(torch.isnan(out["logits"]).any()))

cfg = FusionConfig(fusion_mode="gate", modality_dropout=0.5)
m = FusionNet(tab_dim=TAB, cfg=cfg).train()
vals = []
for _ in range(200):
    with torch.no_grad():
        out = m(pooled, tokens, tab, mod, modality_dropout=True)
    vals.append(out["logits"])
print("  p=0.5 时 200 次前向全部有限?", bool(all(torch.isfinite(v).all() for v in vals)))

print()
print("=== 旧实现 vs 新实现的对照（同一随机种子） ===")
cfg = FusionConfig(fusion_mode="gate", modality_dropout=0.5)
m = FusionNet(tab_dim=TAB, cfg=cfg).eval()
with torch.no_grad():
    a = m(pooled, tokens, tab, mod, modality_dropout=False)
print("  推理时(关闭丢弃) logits 均值 %.4f" % float(a["logits"].mean()))
print("  gate 均值 %.3f  (0.5=完全对称, 接近说明门控还没学会偏向)" % float(a["gate"].mean()))
