# -*- coding: utf-8 -*-
"""
SmolVLM-256M QLoRA 微调 + DPO 偏好对齐 —— 实验作业五实现
对应教材「参数高效微调」与「偏好对齐 / DPO」相关章节

以指南第七节参考代码为骨架，本文件在其上做了如下改造（报告中均有说明）：
  [改造1] 训练设备自适应：有 NVIDIA GPU 且 bitsandbytes 可用时走 4-bit NF4 + LoRA
          （真正的 QLoRA）；否则自动回退 fp32 基座 + LoRA（只训适配器的思想不变）。
  [改造2] 新增 5 组自拟偏好对 MY_PAIRS（任务三），其中 1 组为「冗长废话型」rejected。
  [改造3] 判分函数 judge() 修正指南中的「有」误判（预测“没有”含“有”会被判对）。
  [改造4] 全过程数据落盘：logs/*.json 记录每题输出、SFT/DPO 逐步 loss 与 margin，
          供 plot_curves.py 画训练曲线，报告数据全部由脚本输出直接抄录。
  [改造5] 新增 --r / --lr / --seed / --sft_dir / --dpo_dir / --pairs / --no_4bit
          / --tag 等参数，便于任务五的 β 与秩敏感性扫描且不用改代码。

常用命令：
python hw5_dpo.py --mode make_data            # 生成 15 张合成图像（首次运行自动执行）
python hw5_dpo.py --mode infer                # 基线推理 15 题，记录基线正确率
python hw5_dpo.py --mode sft                  # QLoRA SFT（产出 ./sft_adapter）
python hw5_dpo.py --mode dpo                  # 在 SFT 之上做 DPO（产出 ./dpo_adapter）
python hw5_dpo.py --mode eval                 # 三阶段（基线/SFT/SFT+DPO）× 15 题对比
python hw5_dpo.py --dpo_check 1.2,0.8,0.9,0.6,0.1   # DPO loss 手算复核
python hw5_dpo.py --mode dpo --beta 0.5 --tag beta0.5   # β 敏感性扫描（任务五 a）
python hw5_dpo.py --mode sft --r 16 --tag r16           # 秩敏感性扫描（任务五 b）
"""
import argparse, gc, json, math, os, random, time

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_MODEL = os.path.join(_HERE, "SmolVLM-256M-Instruct")
# [改造] 若已把权重下载到项目内，则优先离线加载（等价权重），否则走 HuggingFace 在线/镜像
MODEL_ID = os.environ.get("HW5_MODEL_ID") or (
    _LOCAL_MODEL if os.path.isfile(os.path.join(_LOCAL_MODEL, "config.json"))
    else "HuggingFaceTB/SmolVLM-256M-Instruct")
IMG_DIR = os.path.join(_HERE, "hw5_imgs")
SFT_DIR, DPO_DIR = os.path.join(_HERE, "sft_adapter"), os.path.join(_HERE, "dpo_adapter")
LOG_DIR = os.path.join(_HERE, "logs")

# ============================================================
# 1. 合成图像库「几何图形」：15 张图，三类问题的答案均已埋入
# 每张图 = 白底 256x256，画若干「颜色-形状」对（位置固定可复现）
# ============================================================
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}
# img_id -> [(颜色, 形状, 个数), ...]（形状：圆=ellipse，方=rectangle）
IMG_SPEC = {
    "img01": [("红", "圆", 2), ("蓝", "方", 1)],
    "img02": [("绿", "圆", 3)],
    "img03": [("蓝", "圆", 1), ("黄", "方", 2)],
    "img04": [("红", "方", 3)],
    "img05": [("绿", "方", 2), ("红", "圆", 1)],
    "img06": [("黄", "圆", 4)],
    "img07": [("蓝", "方", 2), ("绿", "圆", 1)],
    "img08": [("红", "圆", 4)],
    "img09": [("绿", "方", 1), ("黄", "圆", 2)],
    "img10": [("蓝", "圆", 3)],
    "img11": [("黄", "方", 1)],                    # 判断题素材：只有一个黄方
    "img12": [("红", "方", 1), ("蓝", "方", 1)],   # 没有圆形 → 「有没有红圆」答「没有」
    "img13": [("绿", "圆", 2), ("黄", "圆", 1)],
    "img14": [("蓝", "方", 3), ("红", "方", 1)],
    "img15": [("黄", "圆", 2), ("绿", "圆", 2)],
}


def make_images():
    os.makedirs(IMG_DIR, exist_ok=True)
    rnd = random.Random(42)
    for img_id, spec in IMG_SPEC.items():
        im = Image.new("RGB", (256, 256), "white")
        d = ImageDraw.Draw(im)
        for color, shape, n in spec:
            for _ in range(n):
                x, y = rnd.randint(10, 200), rnd.randint(10, 200)
                s = rnd.randint(30, 50)
                box = [x, y, x + s, y + s]
                if shape == "圆":
                    d.ellipse(box, outline=COLORS[color], width=4)
                else:
                    d.rectangle(box, outline=COLORS[color], width=4)
        im.save(f"{IMG_DIR}/{img_id}.png")
    print(f"[data] 已生成 {len(IMG_SPEC)} 张图像到 {IMG_DIR}/")


def load_image(img_id):
    p = f"{IMG_DIR}/{img_id}.png"
    if not os.path.exists(p):
        make_images()
    return Image.open(p).convert("RGB")


# ============================================================
# 2. 问题集：3 类 × 5 题（gold = 标准答案；与参考答案逐字匹配计分）
# count=数数（局部事实） attr=属性识别 exist=存在性/组合判断
# ============================================================
QUESTIONS = [
    {"id": "P01", "type": "count", "img": "img01", "q": "图中有几个红色圆形？", "gold": "2"},
    {"id": "P02", "type": "count", "img": "img02", "q": "图中有几个绿色圆形？", "gold": "3"},
    {"id": "P03", "type": "count", "img": "img03", "q": "图中有几个黄色方形？", "gold": "2"},
    {"id": "P04", "type": "count", "img": "img06", "q": "图中有几个黄色圆形？", "gold": "4"},
    {"id": "P05", "type": "count", "img": "img10", "q": "图中有几个蓝色圆形？", "gold": "3"},
    {"id": "P06", "type": "attr", "img": "img01", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P07", "type": "attr", "img": "img05", "q": "图中红色的图形是什么形状？", "gold": "圆形"},
    {"id": "P08", "type": "attr", "img": "img09", "q": "图中绿色的图形是什么形状？", "gold": "方形"},
    {"id": "P09", "type": "attr", "img": "img13", "q": "图中黄色的图形是什么形状？", "gold": "圆形"},
    {"id": "P10", "type": "attr", "img": "img14", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P11", "type": "exist", "img": "img11", "q": "图里有没有黄色方形？", "gold": "有"},
    {"id": "P12", "type": "exist", "img": "img12", "q": "图里有没有红色圆形？", "gold": "没有"},
    {"id": "P13", "type": "exist", "img": "img07", "q": "图里有没有绿色圆形？", "gold": "有"},
    {"id": "P14", "type": "exist", "img": "img08", "q": "图里有没有绿色方形？", "gold": "没有"},
    {"id": "P15", "type": "exist", "img": "img15", "q": "图里有没有蓝色图形？", "gold": "没有"},
]

# SFT 数据：图像 + 问题 → 简短标准答案（教模型「答题格式 + 事实」）
SFT_DATA = [{"img": q["img"], "prompt": q["q"],
             "answer": f"{q['gold']}。" if q["type"] != "count" else f"{q['gold']}个。"}
            for q in QUESTIONS]

# 偏好对（12 组）：chosen = 正确回答；rejected = 典型幻觉/错误
# rejected 的三类错误：数错（off-by-one / 混淆）、形状幻觉、对不存在的物体硬答「有」
PRE_PAIRS = [
    {"img": "img01", "prompt": "图中有几个红色圆形？",
     "chosen": "2 个。", "rejected": "3 个。图中有三个红色圆形整齐排列。"},
    {"img": "img02", "prompt": "图中有几个绿色圆形？",
     "chosen": "3 个。", "rejected": "2 个。"},
    {"img": "img03", "prompt": "图中有几个黄色方形？",
     "chosen": "2 个。", "rejected": "没有看到黄色图形。"},
    {"img": "img06", "prompt": "图中有几个黄色圆形？",
     "chosen": "4 个。", "rejected": "3 个。"},
    {"img": "img10", "prompt": "图中有几个蓝色圆形？",
     "chosen": "3 个。", "rejected": "图中有三个蓝色方形。"},
    {"img": "img01", "prompt": "图中蓝色的图形是什么形状？",
     "chosen": "方形。", "rejected": "圆形。"},
    {"img": "img05", "prompt": "图中红色的图形是什么形状？",
     "chosen": "圆形。", "rejected": "方形，而且有两个红色方形。"},
    {"img": "img09", "prompt": "图中绿色的图形是什么形状？",
     "chosen": "方形。", "rejected": "图中没有绿色图形。"},
    {"img": "img11", "prompt": "图里有没有黄色方形？",
     "chosen": "有。", "rejected": "没有。"},
    {"img": "img12", "prompt": "图里有没有红色圆形？",
     "chosen": "没有。", "rejected": "有，图左上角有一个红色圆形。"},
    {"img": "img08", "prompt": "图里有没有绿色方形？",
     "chosen": "没有。", "rejected": "有，一共两个绿色方形。"},
    {"img": "img15", "prompt": "图里有没有蓝色图形？",
     "chosen": "没有。图中只有黄色和绿色的圆形。", "rejected": "有，两个蓝色圆形。"},
]

# ------------------------------------------------------------
# [改造2] 任务三：自拟 5 组偏好对（MY_PAIRS，编号 E01~E05）
# 每组 rejected 对应一种失败模式，chosen 一律取自标准答案；
# 覆盖指南未含的题目（P02/P04/P10/P13/P14 之外的新题），并各注明失败模式。
# ------------------------------------------------------------
MY_PAIRS = [
    # E01 失败模式：数错（off-by-one，且把「属性对」数成另一个颜色）——img10 蓝圆=3
    {"id": "E01", "mode": "数错·off-by-one",
     "img": "img10", "prompt": "图中有几个蓝色圆形？",
     "chosen": "3 个。", "rejected": "4 个。图中蓝色圆形排成两行，上一行 2 个下一行 2 个。"},
    # E02 失败模式：形状幻觉（把圆说成方，与颜色绑定错）——img13 黄圆=1
    {"id": "E02", "mode": "形状幻觉·圆形说成方形",
     "img": "img13", "prompt": "图中黄色的图形是什么形状？",
     "chosen": "圆形。", "rejected": "方形，是两个黄色方形。"},
    # E03 失败模式：硬答「有」（对不存在的物体编造位置）——img14 无绿圆
    {"id": "E03", "mode": "硬答有·对不存在物体编造",
     "img": "img14", "prompt": "图里有没有绿色圆形？",
     "chosen": "没有。", "rejected": "有，图右下角有一个绿色圆形。"},
    # E04 失败模式：硬答「有」+ 计数幻觉（存在性判断被数量带偏）——img12 无红圆
    {"id": "E04", "mode": "硬答有·计数幻觉",
     "img": "img12", "prompt": "图里有没有红色圆形？",
     "chosen": "没有。", "rejected": "有，有两个红色圆形。"},
    # E05 失败模式：冗长废话型（通篇描述与问题无关的视觉细节，不给结论，指南 Q5 提示的退化回答）
    {"id": "E05", "mode": "冗长废话·不给结论",
     "img": "img06", "prompt": "图中有几个黄色圆形？",
     "chosen": "4 个。",
     "rejected": "这是一张白色背景的图片，图中绘制了一些几何图形，它们由线条勾勒而成，"
                 "分布在不同位置，整体构图上疏密有致，可以看出作者希望考察观察者对图形的辨识能力。"},
]
PRE_PAIRS_ALL = PRE_PAIRS + MY_PAIRS


# ============================================================
# 3. 模型加载：QLoRA（教材：NF4 量化 + LoRA 低秩适配器）
# 256M 小模型按理可全参微调，但本实验统一走 QLoRA 流程以体会方法
# ============================================================
def _bnb_available():
    """[改造1] 检测 bitsandbytes 是否可用（CPU/无 bnb 时自动回退 fp32+LoRA）。"""
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception as e:  # pragma: no cover
        print(f"[env] bitsandbytes 不可用（{type(e).__name__}: {e}），回退 fp32 基座 + LoRA")
        return False


def _auto_model_cls():
    import transformers
    if hasattr(transformers, "AutoModelForVision2Seq"):
        return transformers.AutoModelForVision2Seq
    return transformers.AutoModelForImageTextToText


def load_model(mode="infer", r=8, lora_alpha=16, lora_dropout=0.05, use_4bit=True,
               grad_ckpt=True, device=None, quantize=None):
    """mode: infer=原精度推理 / train=4bit 量化+LoRA（无 GPU 时自动回退 fp32+LoRA）
    quantize: 显式指定是否 4-bit 量化（评测阶段可设为 True 以与训练量化口径一致）"""
    from transformers import AutoProcessor, BitsAndBytesConfig
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kwargs, quantized = {}, False
    want_quant = (mode == "train" and use_4bit) if quantize is None else quantize
    if want_quant:
        if device.startswith("cuda") and _bnb_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = {"": device}
            quantized = True
        else:
            print("[env] 无 NVIDIA GPU 或 bitsandbytes 不可用 → 按指南 Q2 回退 fp32 基座 + LoRA")
    if not quantized:
        kwargs["torch_dtype"] = torch.float32
    model = _auto_model_cls().from_pretrained(MODEL_ID, **kwargs)
    if mode == "train":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if quantized:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=grad_ckpt)
        else:
            # [改造1] fp32 回退路径同样「冻结基座、只训适配器」
            for p in model.parameters():
                p.requires_grad = False
            if grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        lora = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        # 训练时关闭 KV cache（梯度检查点与 use_cache 不兼容）
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        model.print_trainable_parameters()   # ← 报告 2.1 节要抄的可训练参数占比
        if not quantized:
            model.to(device)
    else:
        model.to(device)
    model.eval() if mode == "infer" else None
    return processor, model


def count_trainable(model):
    """[改造4] 记录可训练参数统计，供报告表 1 抄录。
    注意：bnb 的 Params4bit 用 U8 打包存储，numel() 只有真实参数量的一半，
    故量化模型直接复用 peft 的 get_nb_trainable_parameters（它会乘回 2×element_size）。"""
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable,
            "ratio": trainable / total, "ratio_pct": 100.0 * trainable / total}


def build_prompt(processor, q):
    msgs = [{"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


@torch.no_grad()
def answer(processor, model, img, q, max_new=24):
    inputs = processor(text=[build_prompt(processor, q)], images=[img],
                       return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return text.strip()


def judge(q, pred):
    """宽松判分：标准答案出现在预测里即算对。
    [改造3] 修正指南原式：gold=「有」时预测「没有」也含「有」，会被误判为正确，
            故对「有」加否定词排除；对「没有」须整词出现（“没有”本身已包含“有”）。"""
    p, g = pred.replace(" ", ""), q["gold"]
    if g == "没有":
        return "没有" in p
    if g == "有":
        return ("有" in p) and ("没有" not in p) and ("无" not in p)
    return g in p


# ============================================================
# 4. SFT（教材：监督微调）：只对答案 token 计损失，全流程手写便于理解
# ============================================================
def sft_step(model, processor, opt, epochs=2, steps=60, log_every=10,
             out_dir=SFT_DIR, log_path=None):
    dev = next(model.parameters()).device
    model.train()
    history, t0 = [], time.time()
    for ep in range(epochs):
        rnd = random.Random(ep)
        order = list(range(len(SFT_DATA)))
        rnd.shuffle(order)
        for i, idx in enumerate(order):
            if ep * len(order) + i >= steps:
                print(f"[sft] 已达 {steps} 步，提前结束")
                _finish_sft(model, out_dir, history, log_path, t0)
                return history
            s = SFT_DATA[idx]
            full = processor(text=[build_prompt(processor, s["prompt"]) + s["answer"]],
                             images=[load_image(s["img"])], return_tensors="pt").to(dev)
            with torch.no_grad():
                prompt_len = processor(text=[build_prompt(processor, s["prompt"])],
                                       images=[load_image(s["img"])],
                                       return_tensors="pt")["input_ids"].shape[1]
            labels = full["input_ids"].clone()
            labels[:, :prompt_len] = -100          # 只学答案
            full["labels"] = labels
            loss = model(**full).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            history.append({"ep": ep + 1, "step": ep * len(order) + i + 1,
                            "loss": loss.item(), "t": time.time() - t0})
            if (i + 1) % log_every == 0:
                print(f"[sft] ep{ep+1} step{i+1}/{len(order)} loss={loss.item():.4f}")
    _finish_sft(model, out_dir, history, log_path, t0)
    return history


def _finish_sft(model, out_dir, history, log_path, t0):
    model.save_pretrained(out_dir)
    print(f"[sft] 适配器已保存到 ./{out_dir}（耗时 {time.time()-t0:.1f}s）")
    if log_path:
        dump_json(log_path, {"history": history, "seconds": time.time() - t0,
                             "params": count_trainable(model)})


# ============================================================
# 5. DPO（教材：直接偏好优化）
# loss = -log σ( β·[(πθ(yw)-πref(yw)) - (πθ(yl)-πref(yl))] )
# 参考模型 = 关闭 LoRA 适配器后的基座（disable_adapter 上下文）
# ============================================================
def seq_logprob(model, processor, img, prompt, resp):
    """返回答案 token 的序列总对数概率（只对答案位置求和）"""
    dev = next(model.parameters()).device
    full = processor(text=[build_prompt(processor, prompt) + resp],
                     images=[img], return_tensors="pt").to(dev)
    plen = processor(text=[build_prompt(processor, prompt)], images=[img],
                     return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :plen] = -100
    logits = model(**full).logits[:, :-1, :]
    tgt = labels[:, 1:]
    lp = torch.log_softmax(logits.float(), dim=-1)
    mask = tgt.ne(-100)
    return lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)[mask].sum()


def dpo_step(model, processor, opt, beta=0.1, epochs=1, steps=40, log_every=5,
             pairs=None, out_dir=DPO_DIR, log_path=None):
    pairs = pairs if pairs is not None else PRE_PAIRS_ALL
    dev = next(model.parameters()).device
    model.train()
    history, t0 = [], time.time()
    for ep in range(epochs):
        rnd = random.Random(ep)
        order = list(range(len(pairs)))
        rnd.shuffle(order)
        for i, idx in enumerate(order):
            if ep * len(order) + i >= steps:
                print(f"[dpo] 已达 {steps} 步，提前结束")
                _finish_dpo(model, out_dir, history, log_path, t0, beta, pairs)
                return history
            p = pairs[idx]
            img = load_image(p["img"])
            pol_w = seq_logprob(model, processor, img, p["prompt"], p["chosen"])
            pol_l = seq_logprob(model, processor, img, p["prompt"], p["rejected"])
            with model.disable_adapter():          # 冻结的参考模型 π_ref
                with torch.no_grad():              # [改造] 参考路径不需要梯度
                    ref_w = seq_logprob(model, processor, img, p["prompt"], p["chosen"])
                    ref_l = seq_logprob(model, processor, img, p["prompt"], p["rejected"])
            margin = beta * ((pol_w - ref_w) - (pol_l - ref_l))
            loss = -F.logsigmoid(margin)
            loss.backward()
            opt.step()
            opt.zero_grad()
            history.append({"ep": ep + 1, "step": ep * len(order) + i + 1,
                            "pair": p.get("id", idx), "loss": loss.item(),
                            "margin": margin.item(),
                            "pol_w": pol_w.item(), "pol_l": pol_l.item(),
                            "ref_w": ref_w.item(), "ref_l": ref_l.item(),
                            "t": time.time() - t0})
            if (i + 1) % log_every == 0:
                print(f"[dpo] ep{ep+1} step{i+1}/{len(order)} loss={loss.item():.4f} "
                      f"margin={margin.item():.3f}")
    _finish_dpo(model, out_dir, history, log_path, t0, beta, pairs)
    return history


def _finish_dpo(model, out_dir, history, log_path, t0, beta, pairs):
    model.save_pretrained(out_dir)
    print(f"[dpo] 适配器已保存到 ./{out_dir}（耗时 {time.time()-t0:.1f}s）")
    if log_path:
        dump_json(log_path, {"history": history, "seconds": time.time() - t0,
                             "beta": beta, "n_pairs": len(pairs)})


def dpo_check(arg):
    """手算复核：依次给 pol_w, ref_w, pol_l, ref_l, beta"""
    pw, rw, pl, rl, beta = [float(x) for x in arg.split(",")]
    margin = beta * ((pw - rw) - (pl - rl))
    # [改造] 指南原式 loss = -math.log(1 + math.exp(-margin)) 符号写反（会输出负损失）；
    # 因为 -log σ(m) = +log(1 + e^(-m))，下面用数值稳定写法给出正值。
    loss = math.log(1 + math.exp(-margin)) if margin >= 0 else -margin + math.log(1 + math.exp(margin))
    print(f"margin = β·[(πθ(yw)-πref(yw)) - (πθ(yl)-πref(yl))] = {margin:.4f}")
    print(f"DPO loss = -log σ(margin) = {loss:.4f}"
          f"（margin 越大 loss 越小；margin<0 说明模型更偏好 rejected，需要继续训练）")
    return {"pol_w": pw, "ref_w": rw, "pol_l": pl, "ref_l": rl, "beta": beta,
            "margin": margin, "loss": loss}


# ============================================================
# 6. 三阶段对比评测：基线 / SFT / SFT+DPO × 15 题（对照实验作业三阶段表）
# ============================================================
def evaluate(processor, model, tag, verbose=True):
    ok_by_type, records = {}, []
    for tp in ("count", "attr", "exist"):
        qs = [q for q in QUESTIONS if q["type"] == tp]
        ok = 0
        for q in qs:
            pred = answer(processor, model, load_image(q["img"]), q["q"])
            good = judge(q, pred)
            ok += int(good)
            records.append({"id": q["id"], "type": tp, "q": q["q"], "gold": q["gold"],
                            "pred": pred, "ok": bool(good)})
            if verbose:
                print(f"[{tag}][{q['id']}·{tp}] {'√' if good else '×'} Q: {q['q']} "
                      f"A: {pred}（gold: {q['gold']}）")
        ok_by_type[tp] = [ok, len(qs)]
    total = sum(o for o, _ in ok_by_type.values())
    if verbose:
        for tp, (o, n) in ok_by_type.items():
            print(f"{tag:<10}{tp:<8}正确 {o}/{n}")
        print(f"{tag:<10}总计 正确 {total}/15\n")
    return {"tag": tag, "by_type": ok_by_type, "total": total, "records": records}


def load_adapter_for_infer(model, adapter_dir):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, adapter_dir)


def run_eval(stages, out_json=None, verbose=True, quantize=None):
    """stages: [(tag, adapter_dir_or_None), ...]"""
    results = {}
    for tag, adapter in stages:
        if adapter and not os.path.exists(adapter):
            print(f"[eval] 跳过 {tag}：未找到 {adapter}")
            continue
        processor, model = load_model("infer", quantize=quantize)
        if adapter:
            model = load_adapter_for_infer(model, adapter)
            model.eval()
        results[tag] = evaluate(processor, model, tag, verbose=verbose)
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if out_json:
        dump_json(out_json, results)
    return results


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[log] 已写入 {path}")


# ============================================================
# 7. 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="实验作业五：SmolVLM-256M QLoRA + DPO")
    ap.add_argument("--mode", choices=["make_data", "infer", "sft", "dpo", "eval"],
                    default="infer")
    ap.add_argument("--steps", type=int, default=None, help="覆盖默认训练步数（CPU 演示用 5~10）")
    ap.add_argument("--epochs", type=int, default=None,
                    help="训练轮数（指南默认 1 轮；1 轮=len(pairs) 步，故 DPO 默认 17 步而非 40 步）")
    ap.add_argument("--eval_4bit", action="store_true",
                    help="评测时也用 4-bit NF4 基座（与训练量化口径一致）")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO 温度系数 β")
    ap.add_argument("--dpo_check", default=None, help="如 1.2,0.8,0.9,0.6,0.1")
    # ---- [改造5] 以下参数便于任务五的敏感性扫描，不改代码即可复现 ----
    ap.add_argument("--r", type=int, default=8, help="LoRA 秩 r")
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="覆盖默认学习率")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_4bit", action="store_true", help="禁用 4-bit 量化（fp32 + LoRA）")
    ap.add_argument("--pairs", choices=["all", "base"], default="all",
                    help="all=指南 12 组 + 自拟 5 组；base=仅指南 12 组")
    ap.add_argument("--sft_dir", default=SFT_DIR)
    ap.add_argument("--dpo_dir", default=DPO_DIR)
    ap.add_argument("--tag", default="", help="日志文件名后缀，如 beta0.5 / r16")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(LOG_DIR, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    pairs = PRE_PAIRS_ALL if args.pairs == "all" else PRE_PAIRS
    # SFT 输出目录带 tag（便于秩敏感性扫描）；DPO 输入优先用同 tag 的 SFT 目录，没有则用主 SFT 目录
    sft_out = f"{args.sft_dir}_{args.tag}" if args.tag else args.sft_dir
    sft_in = sft_out if os.path.exists(sft_out) else args.sft_dir
    dpo_dir = f"{args.dpo_dir}_{args.tag}" if args.tag else args.dpo_dir

    if args.dpo_check:
        res = dpo_check(args.dpo_check)
        dump_json(f"{LOG_DIR}/dpo_check{suffix}.json", res)
        return
    if args.mode == "make_data":
        make_images()
        return
    if args.mode == "infer":
        t0 = time.time()
        processor, model = load_model("infer")
        res = evaluate(processor, model, "基线", verbose=True)
        res["seconds"] = time.time() - t0
        dump_json(f"{LOG_DIR}/infer_baseline{suffix}.json", res)
        return
    if args.mode == "sft":
        processor, model = load_model("train", r=args.r, lora_alpha=args.lora_alpha,
                                      use_4bit=not args.no_4bit)
        stats = count_trainable(model)
        print(f"[sft] 可训练参数 {stats['trainable']:,} / 总参数 {stats['total']:,} "
              f"= {stats['ratio_pct']:.4f}%")
        dump_json(f"{LOG_DIR}/sft_params{suffix}.json",
                  {"params": stats, "r": args.r, "lora_alpha": args.lora_alpha,
                   "quantized": not args.no_4bit, "lr": args.lr or 1e-4,
                   "steps": args.steps or 60})
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr or 1e-4)
        sft_step(model, processor, opt, epochs=args.epochs or 2, steps=args.steps or 60,
                 out_dir=sft_out, log_path=f"{LOG_DIR}/sft_log{suffix}.json")
        return
    if args.mode == "dpo":
        if not os.path.exists(sft_in):
            raise SystemExit(f"先运行 --mode sft 生成 SFT 适配器（{sft_in}）")
        processor, model = load_model("train", r=args.r, lora_alpha=args.lora_alpha,
                                      use_4bit=not args.no_4bit)
        # [改造] 把 SFT 适配器权重装进当前可训练的 LoRA（指南原式对 4-bit 基座调 .float() 会破坏量化）
        load_sft_weights(model, sft_in)
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr or 5e-5)
        dpo_step(model, processor, opt, beta=args.beta, epochs=args.epochs or 1,
                 steps=args.steps or 10 ** 6, pairs=pairs, out_dir=dpo_dir,
                 log_path=f"{LOG_DIR}/dpo_log{suffix}.json")
        return
    if args.mode == "eval":
        stages = [("基线", None)]
        for tag, d in (("SFT", args.sft_dir), ("SFT+DPO", args.dpo_dir)):
            dd = f"{d}_{args.tag}" if (args.tag and os.path.exists(f"{d}_{args.tag}")) else d
            stages.append((tag, dd))
        out = f"{LOG_DIR}/eval_results{suffix}{'_4bit' if args.eval_4bit else ''}.json"
        run_eval(stages, out_json=out, quantize=True if args.eval_4bit else False)
        print("提示：把三阶段结果抄进报告表 3；另抽题做人工忠实度评分（答案是否仅陈述图中事实）。")
        return


def load_sft_weights(model, sft_dir):
    """把已保存的 SFT LoRA 权重加载进当前 PeftModel（不新建适配器、不动基座）。"""
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    path = os.path.join(sft_dir, "adapter_model.safetensors")
    if os.path.exists(path):
        set_peft_model_state_dict(model, load_file(path))
        print(f"[dpo] 已从 {sft_dir} 载入 SFT 适配器权重作为 DPO 起点（参考模型 = 关闭该适配器的基座）")
    else:  # 兼容 bin 格式
        from peft import PeftModel
        model.load_adapter(sft_dir, adapter_name="default", is_trainable=True)
        print(f"[dpo] 通过 load_adapter 载入 {sft_dir}")


if __name__ == "__main__":
    main()
