# 实验作业五：SmolVLM-256M 的 QLoRA 高效微调与 DPO 偏好对齐

本目录包含按《作业五_SmolVLM-256M_QLoRA与DPO对齐实验_实验指南》实现的完整实验：单文件脚本
`hw5_dpo.py`（QLoRA SFT + DPO）、合成图像集、17 组偏好对（指南 12 组 + 自拟 5 组）、三阶段评测、
训练曲线、实测日志与实验报告。

## 实测环境（本机）

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU（8 GB，驱动 566.07，CUDA 12.4） |
| Python | 3.12.14 |
| torch / transformers / peft / bitsandbytes | 2.6.0+cu124 / 4.57.6 / 0.21.2 / 0.50.2 |
| 模型 | HuggingFaceTB/SmolVLM-256M-Instruct（safetensors 513 MB，已下载到项目内） |

## 一键复现

```powershell
.\install_deps.ps1          # 依赖装到项目内 .deps（torch 走 cu124 wheel）
python download_model.py    # 下载/校验模型权重（失败自动回退 hf-mirror.com）
python check_env.py         # 环境自检：CUDA、bitsandbytes、processor
.\run_all.ps1               # make_data → infer → sft → dpo → eval → dpo_check
python plot_curves.py       # 由 logs/*.json 生成 figs/fig1~fig4
python make_screenshot.py params; python make_screenshot.py dpo_check
python build_report.py      # 生成 output/学号_姓名_实验作业五.docx
```

## 常用命令

```powershell
python hw5_dpo.py --mode make_data                 # 生成 15 张合成图像（种子 42）
python hw5_dpo.py --mode infer                     # 基线推理 15 题
python hw5_dpo.py --mode sft                       # QLoRA SFT（4-bit NF4 + LoRA r=8）
python hw5_dpo.py --mode dpo                       # 在 SFT 之上做 DPO（17 组偏好对）
python hw5_dpo.py --mode eval                      # 三阶段 × 15 题对比（fp32 口径）
python hw5_dpo.py --mode eval --eval_4bit          # 三阶段对比（4-bit 口径，与训练一致）
python hw5_dpo.py --dpo_check 1.2,0.8,0.9,0.6,0.1  # DPO loss 手算复核
python hw5_dpo.py --mode dpo --beta 0.5 --tag beta0.5   # β 敏感性扫描（任务五 a）
python hw5_dpo.py --mode sft --r 16 --tag r16           # 秩敏感性扫描（任务五 b）
```

## 关键实测结果

- 可训练参数：1,363,968 / 257,848,896 = **0.5290%**（r=8，α=16，q/k/v/o_proj）
- SFT 30 步（2 轮）loss：8.85 → 1.50（最低），耗时 152 秒
- DPO 17 步（1 轮，β=0.1）：loss 0.599→0.540，margin 持续为正，耗时 211 秒
- 三阶段正确率（fp32 口径，15 题）：基线 1 → SFT 4 → SFT+DPO 1
- 三阶段正确率（4-bit 口径，与训练一致）：基线 1 → SFT 3 → SFT+DPO 4（exist 首次命中 1 题）
- `--dpo_check 1.2,0.8,0.9,0.6,0.1` → margin=0.0100，loss=0.6882（指南正文的 ≈0.693 是 margin→0 的近似；
  指南代码的 `-log(1+e^-m)` 符号写反，本实现已修正为 `+log(1+e^-m)`）
- 延伸选做：β 扫描（margin 均值 0.246 / 0.485 / 2.146 对应 β=0.05 / 0.1 / 0.5）、秩扫描
  （r=4/8/16 → 可训练参数 681,984 / 1,363,968 / 2,727,936，占比 0.2652% / 0.5290% / 1.0524%，
  SFT 正确率 3/4/3（fp32）、3/3/3（4-bit）——参数涨 4 倍，正确率不涨）

## 目录说明

- `hw5_dpo.py`：主脚本（含全部改造与注释，改造点见文件头 docstring）
- `logs/`：`*.json` 为结构化结果（每题输出、逐步 loss/margin），`*.log` 为控制台原始输出
- `figs/`：报告用图（SFT loss、DPO loss/margin、三阶段对比、参数占比、控制台截图）
- `sft_adapter/`、`dpo_adapter/`：LoRA 适配器权重（DPO 在 SFT 之上训练）
- `output/`：最终提交物（DOCX / PDF）
- `tmp/`：QA 中间产物（PDF 渲染检查、模板结构 dump 等）
