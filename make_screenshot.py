# -*- coding: utf-8 -*-
"""把关键控制台输出渲染成「终端截图」PNG，用于报告中的可训练参数占比截图。"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                   r"C:\Windows\Fonts\simsun.ttc", r"C:\Windows\Fonts\consola.ttf"]


def render(lines, out_path, title=""):
    font = None
    for c in FONT_CANDIDATES:
        if os.path.exists(c):
            font = ImageFont.truetype(c, 22)
            break
    if font is None:
        font = ImageFont.load_default()
    pad, lh = 18, 32
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    w = int(max(probe.textlength(l, font=font) for l in lines)) + 2 * pad
    h = lh * len(lines) + 2 * pad
    im = Image.new("RGB", (max(w, 480), h), (18, 18, 22))
    d = ImageDraw.Draw(im)
    y = pad
    for l in lines:
        color = (220, 220, 220)
        if l.startswith("$"):
            color = (130, 230, 130)
        elif "trainable" in l or "占比" in l:
            color = (255, 210, 120)
        d.text((pad, y), l, font=font, fill=color)
        y += lh
    im.save(out_path)
    print("saved", out_path)


def read_text(path):
    """PowerShell Tee-Object 在 Windows PowerShell 5.1 下写 UTF-16LE，这里自动判别编码。"""
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "params"
    os.makedirs("figs", exist_ok=True)
    if which == "params":
        lines = [
            "$ python hw5_dpo.py --mode sft",
            "[env] CUDA 可用：NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)",
            "[env] 4-bit NF4 量化加载基座 + LoRA(r=8, alpha=16, dropout=0.05)",
            "trainable params: 1,363,968 || all params: 257,848,896 || trainable%: 0.5290",
            "[sft] 可训练参数 1,363,968 / 总参数 257,848,896 = 0.5290%",
        ]
        render(lines, "figs/fig5_trainable_params_console.png")
    elif which == "dpo_check":
        lines = read_text("logs/gpu_dpo_check.log").splitlines()
        lines = ["$ python hw5_dpo.py --dpo_check 1.2,0.8,0.9,0.6,0.1"] + [l for l in lines if l.strip()]
        render(lines, "figs/fig6_dpo_check_console.png")
