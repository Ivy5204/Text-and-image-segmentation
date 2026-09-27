# 新生儿 DR / 超声 + 临床表格多模态四分类

威宁妇幼保健院与安医附院新生儿呼吸系统疾病四分类项目
（阴性 / ARDS / 湿肺 / 肺炎）

---

## 一、项目概述

本项目的目标是用新生儿胸部 X 光片（DR）、心脏超声四腔心切面与临床表格数据，
建立四分类模型，并通过跨中心外部验证检验其泛化能力。

**主要结论：**

| 评估场景 | macro-AUROC | ARDS AUROC |
| --- | --- | --- |
| 内部（威宁，5×5 交叉验证） | **0.834** | **0.960** |
| 外部（威宁训练 → 安医测试） | 0.653 | — |

跨中心性能下降明显，Grad-CAM 与遮蔽实验定位到原因是图像分支部分依赖
图像边缘的采集特征而非肺部病灶。这一发现是本项目的主要贡献。

---

## 二、数据

| 数据 | 位置 | 规模 |
| --- | --- | --- |
| 威宁妇幼（内部） | `E:\学习资料\第一个项目\胎肺数据\威宁妇幼数据_clahe` | 1365 张图，879 条记录 |
| 安医附院（外部） | `E:\学习资料\第一个项目\胎肺数据\安医附院数据` | 342 张图，233 条记录 |
| 安医 CLAHE 版 | `E:\学习资料\第一个项目\胎肺数据\安医附院数据_clahe` | 同上（预处理对齐后） |

**类别分布：**

| 类别 | 威宁 | 安医 |
| --- | --- | --- |
| 阴性 | 501 | 88 |
| ARDS | 57（修正后） | 82 |
| 湿肺 | 151 | 47 |
| 肺炎 | 165 | 16 |

**图片可得性：** 威宁 879 条记录中 483 条同时有 DR 和超声，396 条只有超声，
没有任何一条只有 DR。安医 233 条中 104 条有 DR。

---

## 三、环境

```powershell
# 解释器
E:\模型\mm\Scripts\python.exe      # Python 3.9.13

# 主要依赖
torch 2.8.0+cpu, torchvision 0.23.0+cpu
timm 1.0.11, transformers 4.57.6, safetensors 0.7.0
numpy 1.24.4, pandas 2.3.3, scipy 1.13.1, scikit-learn 1.3.2
lightgbm 4.6.0, tabpfn 8.0.7
opencv-python-headless 4.10.0.84, scikit-image 0.19.2
torchxrayvision 1.5.4
```

**预训练权重缓存：**

| 权重 | 路径 |
| --- | --- |
| RAD-DINO（DR 编码器） | `E:\模型\torch_cache\rad-dino` |
| USF-MAE（超声编码器） | `E:\模型\torch_cache\usfmae` |
| TabPFN v2（表格分支） | `E:\模型\torch_cache\tabpfn` |
| CheXpert DenseNet121（对照） | `E:\模型\torch_cache\torchxrayvision` |

环境变量（已写入 `sitecustomize.py`，无需手动设置）：
`HF_ENDPOINT=https://hf-mirror.com`，模型缓存指向 `E:\模型\hf_cache`。

---

## 四、目录结构

```
mm4class/
├── README.md                    本文件
├── docs/
│   └── 模型结构说明.md           架构详解
├── src/
│   ├── mm_data.py               数据层：读 manifest、表格编码、样本顺序
│   ├── mm_models.py             模型定义：融合网络、表格头、损失函数
│   ├── mm_train.py              训练循环、特征数组组装
│   ├── mm_eval.py               评估协议：分组 CV、指标、配对检验、阈值优化
│   ├── extract/                 特征提取（每个编码器一个脚本）
│   ├── experiments/             主实验矩阵与结果汇总
│   ├── diagnose/                诊断分析（Grad-CAM、遮蔽实验、混杂检验等）
│   └── external/                跨中心外部验证全流程
├── features/                    缓存的特征与数据清单
│   └── archive_versions/        历史版本特征（对照实验用）
└── results/                     实验输出（预测、指标、图）
```

---

## 五、执行流程

所有脚本均从 `mm4class/` 根目录执行，脚本内部自动定位 `src/`。

### 步骤 1：确认数据清单

```powershell
E:\模型\mm\Scripts\python.exe src\extract\extract_features.py
```

首次运行会生成 `features/manifest.csv`（每条记录的患者号、标签、模态、图片路径）。
后续所有实验都基于这份清单，保证样本顺序一致。

### 步骤 2：提取特征（每个编码器只跑一次）

```powershell
# DR：RAD-DINO，518px，输出 1536 维
E:\模型\mm\Scripts\python.exe src\extract\extract_dr_raddino.py

# 超声：USF-MAE，224px，输出 1536 维（覆盖全部 879 条）
E:\模型\mm\Scripts\python.exe src\extract\extract_us_all.py
```

特征缓存在磁盘上，之后所有实验都在特征上跑，不需要重复前向。

### 步骤 3：内部实验

```powershell
# 主实验矩阵（基线 + 融合 + 消融）
E:\模型\mm\Scripts\python.exe src\experiments\run_experiments.py --quick

# 最终模型：OOF Stacking（表格概率 + 图像概率）
E:\模型\mm\Scripts\python.exe src\experiments\stacking.py

# 汇总全部指标
E:\模型\mm\Scripts\python.exe src\experiments\consolidate_metrics.py
```

### 步骤 4：诊断分析

```powershell
# 模态 × 类别交叉表（检查捷径）
E:\模型\mm\Scripts\python.exe src\diagnose\check_modality_confound.py

# Grad-CAM 热力图
E:\模型\mm\Scripts\python.exe src\diagnose\gradcam.py

# 边缘遮蔽实验（验证 Grad-CAM 的发现）
E:\模型\mm\Scripts\python.exe src\diagnose\mask_experiment.py
E:\模型\mm\Scripts\python.exe src\diagnose\eval_masks.py
```

### 步骤 5：跨中心外部验证

```powershell
# 解析安医表格数据
E:\模型\mm\Scripts\python.exe src\external\prepare_ani.py

# 给安医做 CLAHE（与威宁预处理对齐）
E:\模型\mm\Scripts\python.exe src\external\extract_ani_clahe.py

# 外部验证
E:\模型\mm\Scripts\python.exe src\external\external_validation.py

# 预处理变体全对比（原始 / CLAHE / 直方图匹配）
E:\模型\mm\Scripts\python.exe src\external\final_all_methods.py
```

---

## 六、模型配置

```
DR  编码器   RAD-DINO（ViT-B/14，DINOv2 自监督，80 万张胸片）
             518×518，冻结，CLS + patch 均值 = 1536 维

超声 编码器   USF-MAE（ViT-B/16，MAE，37 万张超声）
             224×224，冻结，GAP + GMP = 1536 维

表格 分支     TabPFN v2（表格基础模型），20 维，中位数填充

融合          OOF Stacking：[图像概率(4) ⊕ 表格概率(4)] → Logistic 组合器

决策          类别权重优化（坐标上升，最大化平衡正确率）
```

详细说明见 `docs/模型结构说明.md`。

---

## 七、评估协议

| 设计 | 说明 |
| --- | --- |
| 分组方式 | `StratifiedGroupKFold`，groups = 住院号 |
| 重复次数 | 3 至 5 次（默认 3） |
| 对比方式 | 所有模型跑同一套折，配对重采样 t 检验 |
| 主指标 | macro-AUROC；准确率在 57% 阴性占比下不可靠 |
| 区间估计 | 患者级自助检验，2000 次重采样 |

**重要提示：** ARDS 仅 39 至 57 例，macro-AUROC 的 95% 置信区间约 ±0.08。
小于此幅度的模型差异在单次划分上无法分辨。

---

## 八、数据质量问题

### 已查清

1. **41 组图像跨文件夹重复**（MD5 完全相同）。根因是 41 个患者同时被诊断
   ARDS 和肺炎，整理数据时在两个类别目录各放了一份拷贝。
2. **23 条记录的标签需修正**。这些记录的 `临床诊断` 自由文本以 ARDS 开头
   （如「ARDS、新生儿肺炎」），说明 ARDS 是首位诊断。修正后 ARDS 从 39 例
   增至 57 例，ARDS AUROC 从 0.935 提升到 0.960。
3. **两家医院标签命名不一致**。安医的「新生儿呼吸窘迫综合征」对应 ARDS、
   「新生儿湿肺」对应湿肺，已在外部验证中建立映射。



## 九、结果文件说明

| 文件 | 内容 |
| --- | --- |
| `results/oof_*.npz` | 各模型的折外预测概率（可用于任意指标复算） |
| `results/summary_*.json` | 实验矩阵的指标汇总 |
| `results/gradcam/*.png` | Grad-CAM 热力图 |
| `results/gradcam/cams_*.npz` | 热力图原始数值（用于定量分析） |
| `results/modality_confound.json` | 模态混杂检验结果 |
| `results/external_*.npz` | 外部验证预测 |

---

## 十、已知局限

1. **图像分支跨中心迁移能力弱**（0.770 → 0.615）。Grad-CAM 与遮蔽实验证实
   模型部分依赖图像边缘的采集特征，而非肺部病灶。
2. **超声分支判别力有限**（单模态 macro-AUROC 0.564）。四腔心切面主要用于
   观察心脏，对 ARDS、湿肺、肺炎等肺部疾病的解剖针对性不足。
3. **特征空间域自适应不可行**。特征维度 1536 而安医 DR 仅 104 条，
   样本数远小于维度时协方差无法估计（实测中心内标准化与 CORAL 均导致性能下降）。
4. **两中心病例谱系差异大**。安医 ARDS 占比 35%，威宁仅 4 至 7%，
   提示两家纳入标准不同。
