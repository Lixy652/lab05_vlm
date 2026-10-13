# -*- coding: utf-8 -*-
"""由 logs/*.json 生成报告用图：SFT loss、DPO loss/margin、三阶段正确率、可训练参数占比。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体（Windows 自带微软雅黑）
for cand in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
             r"C:\Windows\Fonts\simsun.ttc"]:
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

LOG = "logs"
OUT = "figs"


def load(name):
    with open(os.path.join(LOG, name), encoding="utf-8") as f:
        return json.load(f)


def plot_sft():
    d = load("sft_log.json")
    h = d["history"]
    steps = [x["step"] for x in h]
    losses = [x["loss"] for x in h]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=144)
    ax.plot(steps, losses, color="#1f77b4", lw=1.6, label="SFT loss（逐步）")
    # 滑动平均，便于看趋势
    k = 5
    if len(losses) >= k:
        ma = [sum(losses[max(0, i - k + 1):i + 1]) / len(losses[max(0, i - k + 1):i + 1])
              for i in range(len(losses))]
        ax.plot(steps, ma, color="#d62728", lw=1.8, ls="--", label=f"滑动平均(窗口{k})")
    ax.set_xlabel("训练步数 step")
    ax.set_ylabel("损失 loss")
    ax.set_title("图 1  SFT（QLoRA，仅答案 token 计损失）损失曲线")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_sft_loss.png")
    plt.close(fig)


def plot_dpo():
    d = load("dpo_log.json")
    h = d["history"]
    steps = [x["step"] for x in h]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=144)
    axes[0].plot(steps, [x["loss"] for x in h], color="#2ca02c", lw=1.6)
    axes[0].set_xlabel("训练步数 step")
    axes[0].set_ylabel("DPO 损失 loss")
    axes[0].set_title("(a) DPO loss")
    axes[0].axhline(0.6931, color="gray", ls=":", lw=1)
    axes[0].text(steps[0], 0.6975, "随机基线 ln2≈0.693", fontsize=8, color="gray")
    axes[0].grid(alpha=0.3)
    axes[1].plot(steps, [x["margin"] for x in h], color="#ff7f0e", lw=1.6)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xlabel("训练步数 step")
    axes[1].set_ylabel("margin = β·Δlogπ")
    axes[1].set_title("(b) DPO margin（>0 表示偏好方向正确）")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"图 2  DPO 训练曲线（β={d['beta']}，{d['n_pairs']} 组偏好对，{len(h)} 步）")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_dpo_curves.png")
    plt.close(fig)


def plot_eval():
    d = load("eval_results.json")
    tags = [t for t in ("基线", "SFT", "SFT+DPO") if t in d]
    types = ["count", "attr", "exist"]
    import numpy as np
    x = np.arange(len(tags))
    w = 0.22
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=144)
    colors = {"count": "#4c72b0", "attr": "#dd8452", "exist": "#55a868"}
    for i, tp in enumerate(types):
        vals = [d[t]["by_type"][tp][0] for t in tags]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=tp, color=colors[tp])
        ax.bar_label(bars, fontsize=8)
    totals = [d[t]["total"] for t in tags]
    ax.plot(x, totals, "k--o", ms=5, lw=1.2, label="总计 /15")
    for xi, t in zip(x, totals):
        ax.annotate(str(t), (xi, t), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(总 {d[t]['total']}/15)" for t in tags])
    ax.set_ylabel("正确题数 /5")
    ax.set_ylim(0, 6)
    ax.set_title("图 3  三阶段 × 题型正确率对比（15 题，宽松判分）")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_three_stage.png")
    plt.close(fig)


def plot_params():
    d = load("sft_params.json")["params"]
    tr, tot = d["trainable"], d["total"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=144)
    ax.barh(["冻结基座", "可训练适配器"], [tot - tr, tr],
            color=["#b0b0b0", "#d62728"])
    ax.set_xscale("log")
    ax.set_xlabel("参数量（对数刻度）")
    ax.set_title(f"图 4  可训练参数占比 {d['ratio_pct']:.4f}%\n"
                 f"（{tr:,} / {tot:,}）", fontsize=10)
    for i, v in enumerate([tot - tr, tr]):
        ax.text(v, i, f" {v:,}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_trainable_params.png")
    plt.close(fig)


def plot_beta_scan():
    """β 敏感性：不同 β / 轮数下 DPO 的 margin 与 loss 轨迹对比。"""
    runs = [("dpo_log_beta0.05.json", "β=0.05（17 步）", "#1f77b4"),
            ("dpo_log.json", "β=0.10（17 步，主实验）", "#2ca02c"),
            ("dpo_log_beta0.5.json", "β=0.50（17 步）", "#d62728"),
            ("dpo_log_b01e3.json", "β=0.10（51 步，3 轮）", "#9467bd")]
    data = []
    for fn, label, color in runs:
        p = os.path.join(LOG, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        h = d["history"]
        data.append((label, color, [x["step"] for x in h],
                     [x["margin"] for x in h], [x["loss"] for x in h]))
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), dpi=144)
    for label, color, steps, margins, losses in data:
        axes[0].plot(steps, margins, color=color, lw=1.6, label=label)
        axes[1].plot(steps, losses, color=color, lw=1.6, label=label)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("训练步数 step")
    axes[0].set_ylabel("margin")
    axes[0].set_title("(a) margin 随步数变化")
    axes[1].axhline(0.6931, color="gray", ls=":", lw=1)
    axes[1].text(1, 0.70, "ln2≈0.693", fontsize=8, color="gray")
    axes[1].set_xlabel("训练步数 step")
    axes[1].set_ylabel("DPO loss")
    axes[1].set_title("(b) loss 随步数变化")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.suptitle("图 7  β 与训练轮数对 DPO 的影响（同一 SFT 起点）")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_beta_scan.png")
    plt.close(fig)


def plot_rank_scan():
    """秩敏感性：r=4/8/16 的可训练参数占比与两种口径下的 SFT 正确率。"""
    import numpy as np
    rs, ratios, acc_fp32, acc_4bit = [], [], [], []
    for r in (4, 8, 16):
        pf = f"{LOG}/sft_params.json" if r == 8 else f"{LOG}/sft_params_r{r}.json"
        ef = f"{LOG}/eval_results.json" if r == 8 else f"{LOG}/eval_results_r{r}.json"
        e4 = f"{LOG}/eval_results_4bit.json" if r == 8 else f"{LOG}/eval_results_r{r}_4bit.json"
        if not os.path.exists(pf):
            continue
        with open(pf, encoding="utf-8") as f:
            p = json.load(f)["params"]
        rs.append(r)
        ratios.append(p["ratio_pct"])
        for path, bucket in ((ef, acc_fp32), (e4, acc_4bit)):
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    d = json.load(f)
                bucket.append(d.get("SFT", {}).get("total", None))
            else:
                bucket.append(None)
    if not rs:
        return
    x = np.arange(len(rs))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), dpi=144)
    b = axes[0].bar(x, ratios, 0.5, color="#4c72b0")
    axes[0].bar_label(b, fmt="%.4f%%", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"r={r}" for r in rs])
    axes[0].set_ylabel("可训练参数占比 /%")
    axes[0].set_title("(a) 可训练参数占比随秩变化")
    axes[0].grid(axis="y", alpha=0.3)
    if any(v is not None for v in acc_fp32):
        axes[1].plot(x, [v if v is not None else float("nan") for v in acc_fp32],
                     "o-", color="#2ca02c", label="fp32 口径")
    if any(v is not None for v in acc_4bit):
        axes[1].plot(x, [v if v is not None else float("nan") for v in acc_4bit],
                     "s--", color="#d62728", label="4-bit 口径")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"r={r}" for r in rs])
    axes[1].set_ylabel("SFT 正确题数 /15")
    axes[1].set_ylim(0, 6)
    axes[1].set_title("(b) SFT 正确率随秩变化")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("图 8  LoRA 秩敏感性（同一数据与步数，仅改 r）")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_rank_scan.png")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{LOG}/sft_log.json"):
        plot_sft()
        print("fig1 done")
    if os.path.exists(f"{LOG}/dpo_log.json"):
        plot_dpo()
        print("fig2 done")
    if os.path.exists(f"{LOG}/eval_results.json"):
        plot_eval()
        print("fig3 done")
    if os.path.exists(f"{LOG}/sft_params.json"):
        plot_params()
        print("fig4 done")
    plot_beta_scan()
    print("fig7 done")
    plot_rank_scan()
    print("fig8 done")
