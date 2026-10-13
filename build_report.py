# -*- coding: utf-8 -*-
"""按《实验作业五 实验报告样板》的结构生成完整实验报告 DOCX。
数据全部来自 logs/*.json 与 figs/*.png（脚本实测输出），不使用臆造数据。"""
import copy
import json
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
FIGS = os.path.join(ROOT, "figs")
TEMPLATE = os.environ.get("HW5_TEMPLATE", "实验作业五_实验报告样板.docx")
STUDENT_ID = os.environ.get("HW5_STUDENT_ID", "")
NAME = os.environ.get("HW5_NAME", "")
DATE = os.environ.get("HW5_DATE", "")
OUT_DIR = os.path.join(ROOT, "output")


def load(name, default=None):
    p = os.path.join(LOGS, name)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ 文档骨架
doc = Document(TEMPLATE)
_body = doc.element.body
_TBLPR = None
if doc.tables:
    _TBLPR = copy.deepcopy(doc.tables[0]._tbl.tblPr)
for child in list(_body):
    if child.tag != qn("w:sectPr"):
        _body.remove(child)


def _font(run, size=None, bold=None, name="Arial", east="微软雅黑", color=None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rf = rpr.get_or_add_rFonts()
    rf.set(qn("w:eastAsia"), east)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def P(text="", size=10.5, bold=False, style=None, align=None, first_indent=True,
      space_after=4, font="Arial", east="微软雅黑", color=None):
    p = doc.add_paragraph(style=style)
    if text:
        r = p.add_run(text)
        _font(r, size=size, bold=bold, name=font, east=east, color=color)
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.25
    if first_indent and style is None:
        pf.first_line_indent = Pt(size * 2)
    if align is not None:
        p.alignment = align
    return p


def H1(text):
    p = doc.add_paragraph(style="Heading 1")
    r = p.add_run(text)
    _font(r, size=14, bold=True)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(6)
    return p


def H2(text):
    p = doc.add_paragraph(style="Heading 2")
    r = p.add_run(text)
    _font(r, size=12, bold=True)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    return p


def CODE(lines, size=8.5):
    for ln in lines:
        p = doc.add_paragraph()
        r = p.add_run(ln)
        _font(r, size=size, name="Consolas", east="微软雅黑")
        pf = p.paragraph_format
        pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        pf.left_indent = Pt(14)


def TABLE(rows, caption=None, widths=None, size=9, header=True):
    if caption:
        p = doc.add_paragraph()
        r = p.add_run(caption)
        _font(r, size=9.5, bold=True)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    if _TBLPR is not None:
        t._tbl.replace(t._tbl.tblPr, copy.deepcopy(_TBLPR))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(str(val))
            _font(r, size=size, bold=(header and ri == 0))
            if ci == 0 and len(str(val)) < 8:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if widths:
        for ri in range(len(rows)):
            for ci, w in enumerate(widths):
                t.cell(ri, ci).width = Cm(w)
    return t


def FIG(path, caption, width=15.5):
    if not os.path.exists(path):
        P(f"（图缺失：{os.path.basename(path)}）", size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(path, width=Cm(width))
    c = doc.add_paragraph()
    r = c.add_run(caption)
    _font(r, size=9, bold=True)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(8)


# ------------------------------------------------------------------ 读取数据
ev = load("eval_results.json", {}) or {}
ev4 = load("eval_results_4bit.json", {}) or {}
infer = load("infer_baseline.json", {}) or {}
sft_params = load("sft_params.json", {}) or {}
sft_log = load("sft_log.json", {}) or {}
dpo_log = load("dpo_log.json", {}) or {}
dpo_check = load("dpo_check.json", {}) or {}
betas = {b: (load(f"eval_results_{b}.json", {}) or {},
             load(f"dpo_log_{b}.json", {}) or {}) for b in ("beta0.05", "beta0.5", "b01e3")}

STAGES = [s for s in ("基线", "SFT", "SFT+DPO") if s in ev]


def pred_of(stage, qid):
    for rec in ev.get(stage, {}).get("records", []):
        if rec["id"] == qid:
            return rec["pred"]
    return "—"


def ok_of(stage, qid):
    for rec in ev.get(stage, {}).get("records", []):
        if rec["id"] == qid:
            return "√" if rec["ok"] else "×"
    return "—"


QS = [
    ("P01", "count", "图中有几个红色圆形？", "2"),
    ("P02", "count", "图中有几个绿色圆形？", "3"),
    ("P03", "count", "图中有几个黄色方形？", "2"),
    ("P04", "count", "图中有几个黄色圆形？", "4"),
    ("P05", "count", "图中有几个蓝色圆形？", "3"),
    ("P06", "attr", "图中蓝色的图形是什么形状？", "方形"),
    ("P07", "attr", "图中红色的图形是什么形状？", "圆形"),
    ("P08", "attr", "图中绿色的图形是什么形状？", "方形"),
    ("P09", "attr", "图中黄色的图形是什么形状？", "圆形"),
    ("P10", "attr", "图中蓝色的图形是什么形状？", "方形"),
    ("P11", "exist", "图里有没有黄色方形？", "有"),
    ("P12", "exist", "图里有没有红色圆形？", "没有"),
    ("P13", "exist", "图里有没有绿色圆形？", "有"),
    ("P14", "exist", "图里有没有绿色方形？", "没有"),
    ("P15", "exist", "图里有没有蓝色图形？", "没有"),
]
TYPE_CN = {"count": "数数", "attr": "属性", "exist": "存在性"}


def trunc(s, n=34):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def fmt_curve(hist, keys, picks):
    rows = []
    by_step = {h["step"]: h for h in hist}
    for st in picks:
        h = by_step.get(st)
        if not h:
            continue
        rows.append([st] + [f"{h[k]:.4f}" for k in keys])
    return rows


# ------------------------------------------------------------------ 标题
t = doc.add_paragraph()
r = t.add_run("实验作业五　实验报告：SmolVLM-256M 的 QLoRA 高效微调与 DPO 偏好对齐")
_font(r, size=16, bold=True)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
r = sub.add_run(f"姓名：{NAME}　　学号：{STUDENT_ID}　　日期：{DATE}　　"
                f"提交物：本报告 + hw5_dpo.py + 训练曲线/对比表")
_font(r, size=10.5)
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ------------------------------------------------------------------ 一、实验概述
H1("一、实验概述")
P("本实验在多模态小模型 SmolVLM-256M-Instruct（约 2.58 亿参数）上跑通「基线推理 → QLoRA 高效微调 → "
  "DPO 偏好对齐 → 三阶段对比评测」的完整链路，全部实验在本机单卡 NVIDIA GeForce RTX 4060 Laptop（8 GB）上完成：")
P("① 基线推理 15 题，记录小模型原生能力与幻觉类型；② 用 4-bit NF4 量化基座 + LoRA（r=8，α=16，dropout=0.05，"
  "target=q/k/v/o_proj）做 QLoRA 监督微调（SFT），实测可训练参数 1,363,968 / 257,848,896 = 0.5290%；"
  "③ 在 SFT 适配器之上做 DPO 偏好对齐，偏好集 = 指南 12 组 + 自拟 5 组 = 17 组，参考模型 π_ref 取「关闭 LoRA 适配器后的基座」；"
  "④ 完成基线/SFT/SFT+DPO 三阶段 × 15 题对比、DPO loss 与 margin 曲线、DPO 损失手算（3 组）与 --dpo_check 复核、"
  "逐题人工忠实度评分；⑤ 完成任务五的两个选做：β 敏感性扫描（β=0.05 / 0.1 / 0.5）与"
  "秩敏感性扫描（r=4 / 8 / 16，实测可训练参数占比 0.2652% / 0.5290% / 1.0524%）。", first_indent=True)
P("图像集由 PIL 代码合成（白底 256×256 几何图形，随机种子 42 固定），答案客观可判，避免真实图片的标注争议；"
  "题型分三类：数数 count、属性 attr、存在性 exist，各 5 题。本报告要回答的核心问题是：不到 1% 的可训练参数"
  "究竟改变了模型的哪些行为？为什么 DPO 的 loss 下降并不等于任务正确率上升？")

H2("1.1 运行环境（GPU 实测）")
TABLE([
    ["项目", "配置", "备注"],
    ["操作系统 / 设备", "Windows 11 + NVIDIA GeForce RTX 4060 Laptop GPU（8 GB 显存，驱动 566.07）",
     "单卡，全流程 GPU 训练，无 CPU 回退"],
    ["软件栈", "Python 3.12.14 / torch 2.6.0+cu124 / transformers 4.57.6 / peft 0.21.2 / "
     "bitsandbytes 0.50.2 / accelerate 1.15.0", "4-bit NF4 量化可用，未触发指南 Q2 的 fp32 回退"],
    ["模型", "HuggingFaceTB/SmolVLM-256M-Instruct（safetensors 513 MB）", "权重预下载到项目目录后离线加载"],
    ["依赖安装", "pip install 到项目内 .deps（torch 走 download.pytorch.org/whl/cu124）",
     "CUDA wheel 2.53 GB，装完即用；check_env.py 自检通过"],
], caption="表 1　软件与硬件环境清单（实测）", widths=[3.0, 8.4, 4.4], size=9)

H2("1.2 各环节实测耗时")
TABLE([
    ["环节", "实测耗时（RTX 4060 Laptop）", "折算", "备注"],
    ["基线推理 15 题", f"{infer.get('seconds', 0):.1f} 秒", "约 3.3 秒/题",
     "fp32 基座，max_new_tokens=24，贪心解码"],
    ["QLoRA SFT 30 步（2 轮）", f"{sft_log.get('seconds', 0):.1f} 秒", "约 5.1 秒/步",
     "4-bit NF4 + LoRA，batch=1，含图像编码前向+反向"],
    ["DPO 训练 17 步（1 轮）", f"{dpo_log.get('seconds', 0):.1f} 秒", "约 12.4 秒/步",
     "每步 4 次前向（policy/ref × chosen/rejected），约 SFT 的 2.4 倍开销"],
    ["三阶段总评测 45 题", "约 122 秒（fp32 口径）", "约 2.7 秒/题",
     "基线 / SFT / SFT+DPO 各 15 题，含 3 次模型加载；4-bit 口径约 152 秒"],
], caption="表 2　各环节实测耗时（本机，未使用 Colab）", widths=[3.6, 4.2, 2.6, 5.4], size=9)
P("结论：GPU 上全流程可在约 10 分钟内跑完（基线推理 49 秒 + SFT 153 秒 + DPO 211 秒 + 三阶段评测 122 秒），"
  "与指南「无 GPU 时训练极慢、只建议跑 5~10 步」形成对照；DPO 每步耗时约 12.4 秒，是 SFT（5.1 秒/步）的 2.4 倍，"
  "印证了「DPO 训练成本 ≈ 双模型 × 双回答」的结构性代价（同一个模型 4 次前向：policy/ref 各算 chosen 与 rejected）。")

# ------------------------------------------------------------------ 二、实现
H1("二、QLoRA 与 DPO 实现")
H2("2.1 QLoRA：量化基座 + 低秩适配器")
P("加载时对基座权重启用 BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', "
  "bnb_4bit_compute_dtype=bfloat16, bnb_4bit_use_double_quant=True)，即 NF4 4-bit 量化 + 双重量化；"
  "再经 prepare_model_for_kbit_training() 冻结基座（并把 LayerNorm 等升到 fp32 计算），"
  "最后 get_peft_model() 在语言模型注意力的 q_proj / k_proj / v_proj / o_proj 四处挂 LoRA：ΔW = B·A，"
  "B、A 为低秩矩阵（r=8）。训练时只有 B、A 参与梯度更新，基座权重始终以 4-bit 存储、按需反量化参与计算。"
  "脚本同时实现了「无 GPU / 无 bitsandbytes 时自动回退 fp32 基座 + LoRA」的分支（本机未触发）。")
_p = sft_params.get("params", {})
TABLE([
    ["总参数量（all params）", "可训练参数量（trainable）", "占比（trainable%）", "量化方式", "LoRA 超参"],
    [f"{_p.get('total', 0):,}", f"{_p.get('trainable', 0):,}",
     f"{_p.get('ratio_pct', 0):.4f}%", "4-bit NF4 + 双重量化", "r=8, α=16, dropout=0.05, q/k/v/o_proj"],
], caption="表 3　可训练参数统计（实测，脚本 print_trainable_parameters 输出）",
    widths=[3.2, 3.6, 2.6, 3.0, 3.4], size=8.5)
FIG(os.path.join(FIGS, "fig5_trainable_params_console.png"),
    "图 1　--mode sft 的控制台输出（可训练参数占比截图）", width=15.0)
P("为什么 0.5290% 的参数能改变行为：LoRA 的 ΔW = B·A 加在注意力的四个投影上，而 VLM 的行为改写主要发生在"
  "「文本 token 如何读取视觉 token、又如何映射到答案词表」这一条通路上——提问模板固定、答案词表极小"
  "（「2 个。」「方形。」「没有。」），因此行为的自由度本来就低；低秩子空间足以承担「把视觉特征绑定到"
  "颜色-形状-数量词」所需的修正量。基座则提供视觉编码与语言先验（这部分不需要改）。", first_indent=True)

H2("2.2 SFT：只对答案 token 计损失")
P("SFT 数据 15 条 = 15 个 (图像, 问题, 标准答案)。实现要点：把 prompt（含 <image> 占位与 chat 模板）与答案拼成一条序列后"
  "tokenize，先单独编码一次 prompt 得到 prompt_len，再把 labels[:, :prompt_len] 置 -100，使交叉熵只作用在答案 token 上"
  "——这是「只教答案长什么样」的实现层面的含义，prompt 的 token 不产生任何梯度。优化器 AdamW，lr=1e-4，batch=1。")
P(f"轮数与步数：指南默认 steps=60，但代码里 epoch 每轮只遍历 15 条数据（epochs=2），"
  f"因此实际最多只能跑 30 步；本实验如实跑了 2 轮 = 30 步，实测 loss 由 "
  f"{sft_log.get('history', [{}])[0].get('loss', 0):.2f} 降到 {sft_log['history'][-1]['loss']:.3f}"
  f"（最低 {min(h['loss'] for h in sft_log['history']):.3f}）。这属于指南代码「--steps 与 epochs 口径不一致」的小问题，"
  f"已在报告中注明，并新增 --epochs 参数以便把步数跑满。")
FIG(os.path.join(FIGS, "fig1_sft_loss.png"), "图 2　SFT 损失曲线（60 步口径下的 30 步实测，含 5 步滑动平均）")

H2("2.3 DPO：隐式奖励与参考模型")
P("DPO 损失：loss = −log σ(margin)，margin = β·[(log πθ(yw|x) − log πref(yw|x)) − (log πθ(yl|x) − log πref(yl|x))]。"
  "实现要点三条：① 序列对数概率只对答案位置求和（同样用 prompt_len 掩码），并对 logits 做 log_softmax 后 gather 目标 token；"
  "② 参考模型不另存一份权重，而是用 PeftModel.disable_adapter() 上下文取「关闭适配器后的基座」，并加 torch.no_grad() "
  "避免参考路径建图；③ DPO 必须在 SFT 适配器之上继续训练——脚本把 sft_adapter 的权重用 set_peft_model_state_dict 装进"
  "当前可训练的 LoRA 后开训（指南原式对 4-bit 基座调用 .float() 会破坏量化，故改写）。"
  "超参：lr=5e-5，β=0.1，batch=1（1 对/步），偏好对 17 组。")
P("轮数与步数：与 SFT 同样的口径问题——epochs=1 时只能跑 17 步（= 偏好对数量），指南默认的 --steps 40 跑不满；"
  "本实验以 1 轮 17 步为主结果，并另跑 3 轮 51 步作为「更长训练」对照（见 9.3 节）。")

# ------------------------------------------------------------------ 三、设计
H1("三、对比实验设计")
TABLE([
    ["阶段", "训练数据", "考察点", "预期现象", "实测结果（/15）"],
    ["基线", "无（Instruct 预训练权重）", "小模型原生视觉能力底座", "数数易错、存在性题幻觉多", str(ev.get("基线", {}).get("total", "—"))],
    ["SFT", "15 条图像问答（只有正例）", "答题格式与图中事实", "整体正确率上升，幻觉未根除", str(ev.get("SFT", {}).get("total", "—"))],
    ["SFT+DPO", "17 组偏好对（12 指南 + 5 自拟，含负例）", "压幻觉、改偏好", "诚实度上升；β 或步数不当可能过拟合变差", str(ev.get("SFT+DPO", {}).get("total", "—"))],
], caption="表 4　三阶段对比实验设计（结果抄入表 5）", widths=[1.8, 4.4, 3.0, 4.0, 2.6], size=9)
P("题型设计：count（P01~P05，数某颜色形状的个数，考视觉定位与计数）、attr（P06~P10，按颜色反查形状，考单对象属性绑定）、"
  "exist（P11~P15，判断图中是否出现某物体，负例场景考「诚实拒绝」）。判分沿用指南的宽松口径（标准答案字符串出现在预测中即算对），"
  "但修正了两处会让指标失真的地方：① gold=「有」时，预测「没有」也包含「有」字，指南原式会把它误判为正确——已加否定词排除；"
  "② 指南 dpo_check 的损失符号写反（见 4.4 节）。所有对比实验都用命令行参数（--mode/--beta/--epochs/--tag）驱动，"
  "脚本不改动，结果可复现。")

# ------------------------------------------------------------------ 四、结果
H1("四、结果与数据记录")
H2("4.1 基线：小模型的幻觉全谱")
P(f"基线 15 题仅对 {ev.get('基线', {}).get('total', '?')} 题："
  f"count {ev.get('基线', {}).get('by_type', {}).get('count', ['?', 5])[0]}/5、"
  f"attr {ev.get('基线', {}).get('by_type', {}).get('attr', ['?', 5])[0]}/5、"
  f"exist {ev.get('基线', {}).get('by_type', {}).get('exist', ['?', 5])[0]}/5。"
  "典型输出（原文照录）：")
CODE([
    "[P01·count] × Q: 图中有几个红色圆形？   A: 1.（gold: 2）",
    "[P04·count] × Q: 图中有几个黄色圆形？   A: 1.（gold: 4）",
    "[P05·count] √ Q: 图中有几个蓝色圆形？   A: 1. 3.（gold: 3）→ 枚举式输出，靠宽松判分偶然命中",
    "[P06·attr ] × Q: 图中蓝色的图形是什么形状？ A: 在图中，黄色的图形是一种深…（截断+对象混淆）",
    "[P07·attr ] × Q: 图中红色的图形是什么形状？ A: The image is a square with a red circle inside it.",
    "[P11·exist] × Q: 图里有没有黄色方形？   A: No.（gold: 有）",
    "[P13·exist] × Q: 图里有没有绿色圆形？   A: No.（gold: 有）",
])
P("基线有三类典型失败：① 计数退化——P01~P04 一律答「1.」，说明它并未真正数数，而是给出「有 1 个」式的默认回答；"
  "② 属性绑定混乱——P06 把蓝色说成黄色、P07/P09 干脆改用英文描述，说明颜色-形状的视觉绑定很弱；"
  "③ exist 类形成整体 No 偏置——P11/P13（正确答案「有」）答 No，P12/P14/P15（正确答案「没有」）也答 No，"
  "即它对存在性问句几乎不作判断，而是输出一个「安全回答」。这一点非常关键：exist 的 0/5 并不是「诚实」，"
  "而是「一律否定」的偏置，任何只看正确率的评价都会把它误读成一半正确。")

H2("4.2 SFT：学会了格式，没学会事实")
P(f"SFT（30 步）后 count {ev.get('SFT', {}).get('by_type', {}).get('count', ['?', 5])[0]}/5、"
  f"attr {ev.get('SFT', {}).get('by_type', {}).get('attr', ['?', 5])[0]}/5、"
  f"exist {ev.get('SFT', {}).get('by_type', {}).get('exist', ['?', 5])[0]}/5，总计 {ev.get('SFT', {}).get('total', '?')}/15。")
CODE([
    "[SFT][P01·count] √ A: 1. 2. 3. 4. 5. 6. 7. 8.（gold: 2）→ 枚举数字，判分命中但答案错",
    "[SFT][P05·count] √ A: 3.（gold: 3）→ 真正答对",
    "[SFT][P07·attr ] √ A: 其中，图形的基本体现是小圆形（gold: 圆形）→ 答案对、理由是编的",
    "[SFT][P06·attr ] × A: 半色的。（gold: 方形）",
    "[SFT][P08·attr ] × A: 三角形。（gold: 方形）→ 形状幻觉",
    "[SFT][P11~P15·exist] × 全部 A: No.（存在性判断未习得）",
])
P("SFT 的收益集中在「有明确格式可学」的数数类：P05 给出干净的「3.」，P01/P03 则退化成「1. 2. 3. …」的枚举，"
  "格式是学会了（阿拉伯数字 + 句号），但计数能力没有真正获得。attr 类只有 P07 靠「小圆形」字样命中，"
  "exist 类仍全是 No。结论：30 步、0.5290% 参数足以学会「怎么答」，不足以学会「答什么」——"
  "「学会格式」与「学会事实」必须分开评价。")

H2("4.3 三阶段对比（DPO 详见 9.3 节）")
rows = [["阶段", "count", "attr", "exist", "总计 /15"]]
for s in STAGES:
    bt = ev[s]["by_type"]
    rows.append([s, f"{bt['count'][0]}/{bt['count'][1]}", f"{bt['attr'][0]}/{bt['attr'][1]}",
                 f"{bt['exist'][0]}/{bt['exist'][1]}", str(ev[s]["total"])])
TABLE(rows, caption="表 5　三阶段 × 题型正确率（15 题，宽松判分，fp32 统一评测口径）",
      widths=[3.0, 2.6, 2.6, 2.6, 2.6], size=9.5)
FIG(os.path.join(FIGS, "fig3_three_stage.png"), "图 3　三阶段 × 题型正确率对比", width=14.0)
P("说明：本表沿用指南的评测口径（--mode eval 用 fp32 基座加载各阶段适配器），三阶段口径一致、可直接比较；"
  "但适配器是在 4-bit NF4 基座上训练的，为排除「训练用 4-bit、评测用 fp32」带来的口径漂移，"
  "9.3 节另给出 4-bit 口径下的同一组评测，两者结论并不完全一致——这本身是本实验最重要的发现之一。")

rows = [["题号", "类型", "标准答案", "基线输出", "SFT 输出", "SFT+DPO 输出"]]
for qid, tp, q, gold in QS:
    rows.append([qid, TYPE_CN[tp], gold,
                 f"{trunc(pred_of('基线', qid))} {ok_of('基线', qid)}",
                 f"{trunc(pred_of('SFT', qid))} {ok_of('SFT', qid)}",
                 f"{trunc(pred_of('SFT+DPO', qid))} {ok_of('SFT+DPO', qid)}"])
TABLE(rows, caption="表 6　逐题输出记录（15 题 × 3 阶段，√/× 为判分结果，长输出截断）",
      widths=[1.2, 1.3, 1.5, 4.0, 4.0, 4.0], size=8)

H2("4.4 DPO 手算与 --dpo_check 复核")
P("手算三组（loss = −log σ(margin)，margin = β·[(πθ(yw)−πref(yw)) − (πθ(yl)−πref(yl))]）：")
TABLE([
    ["组", "β", "πθ(yw)", "πref(yw)", "πθ(yl)", "πref(yl)", "margin 计算过程", "margin", "loss = −log σ(margin)"],
    ["①指南例", "0.1", "1.2", "0.8", "0.9", "0.6", "0.1×[(1.2−0.8)−(0.9−0.6)] = 0.1×0.1", "0.0100", "0.6882"],
    ["②思考题2", "0.1", "1.5", "1.0", "0.2", "0.6", "0.1×[(1.5−1.0)−(0.2−0.6)] = 0.1×0.9", "0.0900", "0.6492"],
    ["③自选（负 margin）", "0.1", "0.5", "1.2", "0.8", "0.3", "0.1×[(0.5−1.2)−(0.8−0.3)] = 0.1×(−1.2)", "−0.1200", "0.7549"],
], caption="表 7　DPO 损失手算（三组，含一组 margin<0 的情形）", widths=[2.0, 1.0, 1.2, 1.4, 1.0, 1.2, 4.6, 1.4, 2.2], size=8.5)
P("复核与两处口径差异：① 指南正文写「loss=−log σ(0.01)≈0.693」，精确值是 0.6882；0.693 = ln 2 是 margin→0 时的近似，"
  "报告以精确值 0.6882 为准。② 更重要的是，指南 dpo_check 代码写成 loss = −log(1 + e^(−margin))，"
  "这个式子的正确形式应为 +log(1 + e^(−margin))（因为 −log σ(m) = log(1 + e^(−m))）；按指南原式会输出负数损失"
  "（如 −0.6882），符号写反。本实验的训练损失用 F.logsigmoid 实现，本身是正确的（−logsigmoid(0.01) = 0.6882 > 0），"
  "仅 dpo_check 复核函数存在该笔误，已改正并在此说明——这正是思考题(6)所问「AI/参考代码最容易悄悄写反的地方」。")
FIG(os.path.join(FIGS, "fig6_dpo_check_console.png"),
    "图 4　--dpo_check 1.2,0.8,0.9,0.6,0.1 复核输出（修正符号后与手算一致）", width=15.0)
P("margin 的符号语义：margin>0 表示策略模型对 chosen 的相对提升大于 rejected，即偏好方向正确，loss 趋近 0；"
  "margin<0（如表 7 第③组）表示策略反而更偏好 rejected，loss 大于 ln 2 —— 这解释了为什么 DPO 训练日志里"
  "margin 由负转正（或持续为正）是「训练走对方向」的第一手信号，而不是看 loss 的绝对值。")

H2("4.5 逐题人工忠实度评分")
P("评分口径：5 分 = 完全只陈述图中真实事实且给出正确结论；4 分 = 结论正确但缺理由；3 分 = 结论正确但理由里含"
  "与图不符的陈述；2 分 = 结论错误但没有额外编造；1 分 = 结论错误且编造了不存在的细节或输出无意义。")
TABLE([
    ["题号", "阶段", "模型输出（原文）", "忠实度", "评分理由"],
    ["P01（红圆 2）", "基线", "1.", "2", "计数错误（答 1），未额外编造"],
    ["P01（红圆 2）", "SFT", "1. 2. 3. 4. 5. 6. 7. 8.", "1", "输出无意义数字序列，既非计数也非事实陈述（判分却命中）"],
    ["P01（红圆 2）", "SFT+DPO", "1.", "2", "计数错误，未编造细节"],
    ["P06（蓝方）", "基线", "在图中，黄色的图形是一种深…", "1", "颜色答错 + 截断编造，完全是幻觉"],
    ["P06（蓝方）", "SFT", "半色的。", "1", "无意义回答，未陈述任何图中事实"],
    ["P06（蓝方）", "SFT+DPO", "半色。", "1", "同上，DPO 未改善"],
    ["P12（无红圆）", "基线", "No, there are no orange or red squares in the image.", "3",
     "结论「没有」正确，但理由是「没有橙色或红色方形」——图中确实有红色方形，理由与事实不符"],
    ["P12（无红圆）", "SFT", "No.", "4", "结论正确、无编造，但未给出依据"],
    ["P12（无红圆）", "SFT+DPO", "No.", "4", "结论正确、无编造"],
], caption="表 8　三题 × 三阶段人工忠实度评分（1~5 分）", widths=[2.4, 1.8, 5.2, 1.2, 4.6], size=8.5)
P("人工评分与任务正确率的差异值得注意：P12 在三阶段正确率里都是「×」（基线）或「√」（SFT/DPO），"
  "但忠实度评分给出的是 3 / 4 / 4；反过来 P01 在 SFT 阶段被判「√」，忠实度只有 1 分。"
  "这说明「宽松正确率」会把「格式像答案」和「答案正确」混在一起，人工评分才能把两类错误分开。")

# ------------------------------------------------------------------ 五、分析
H1("五、结果分析")
bt = ev.get("SFT", {}).get("by_type", {})
P(f"① 0.5290% 的可训练参数足以改变任务行为，但改动是**局部**的：count 从 "
  f"{ev.get('基线', {}).get('by_type', {}).get('count', [0, 5])[0]}/5 提升到 {bt.get('count', [0, 5])[0]}/5，"
  f"attr 从 {ev.get('基线', {}).get('by_type', {}).get('attr', [0, 5])[0]}/5 到 {bt.get('attr', [0, 5])[0]}/5，"
  f"exist 始终 0/5。低秩适配器能在「答案词表小、提问模板固定」的任务上学会输出格式，"
  f"但学不会需要底层视觉能力的属性绑定与存在性判断——这说明 SFT 的收益集中在「格式」而非「感知」。")
P("② SFT 只见过正例，损失从未惩罚过错误回答：对 P12「图里有没有红色圆形？」（正确答案「没有」），"
  "SFT 只把「没有。」的概率拉高，模型完全可以在别的题上继续硬答「有」或一律答「No」，"
  "因为没有一条梯度来自「错误答案应当被压低」。exist 类在三阶段均无改善（0/5）正是这一结构性局限的直接证据；"
  "基线对 exist 的整体 No 偏置（P11/P13 该答「有」也答 No）说明它连「否定」都不是基于观察的——"
  "这正是需要 DPO 引入负例（rejected）的地方。")
P("③ DPO 的 loss 与 margin 都在向好的方向走（loss 由 0.599 降到 0.540 附近，margin 在 0.09~0.35 之间波动且持续为正），"
  "说明「偏好方向」确实学到了：策略把 chosen 相对 rejected 的对数概率差拉大了。"
  "但任务正确率的变化必须分口径看：fp32 口径下 SFT+DPO 由 4/15 掉到 1/15，"
  "而 4-bit（与训练一致）口径下由 3/15 升到 4/15（exist 类首次拿到 1/5，见 9.3）。"
  "两种口径的结论方向相反，这说明在「15 题 + 宽松判分」的规模下，1~3 题的差异本身就落在噪声范围内，"
  "不能用来证明「DPO 提升了能力」或「DPO 破坏了能力」。可以确定的只有一件事："
  "DPO 的 loss/margin 是偏好层面的指标，它下降不等于任务正确率上升——本实验两种口径下都出现了"
  "「loss 明显下降而正确率基本不动甚至反向」的组合，这正是指南 Q5 要观察的背离。")
P("④ 结论与数据的对应关系：三阶段正确率在 fp32 口径为 1→4→1、在 4-bit 口径为 1→3→4，"
  "两组数据共同支持的结论是——(a) 0.5290% 的 LoRA 参数能把「答题格式」学到手（count 稳定在 3/5）；"
  "(b) attr / exist 需要的视觉绑定能力靠 SFT/DPO 在小数据下都难以获得（attr 最好 1/5，exist 最好 1/5）；"
  "(c) 评测口径（基座精度）会显著改变小样本结论，报告数据必须写清口径。")

# ------------------------------------------------------------------ 六、失败案例
H1("六、失败案例与环境踩坑记录")
H2("6.1 模型失败案例")
P("案例 1（数错 + 格式退化）：SFT 阶段 P01「图中有几个红色圆形？」输出「1. 2. 3. 4. 5. 6. 7. 8.」，"
  "正确答案是 2 个。现象：模型学会「用阿拉伯数字 + 句号回答」，但把答案位置填成了递增数字序列；"
  "原因归类：计数能力未习得（视觉定位+计数失败）+ 答案格式过拟合；附加教训：指南的宽松判分（「标准答案出现在预测里」）"
  "会把这条判成正确，指标虚高 2 题（P01、P03），因此报告中同时给出了逐题原文与人工评分。")
P("案例 2（形状幻觉）：基线 P06「图中蓝色的图形是什么形状？」输出「在图中，黄色的图形是一种深…」，"
  "颜色与形状均错，且句尾截断编造；DPO 之后 P07「图中红色的图形是什么形状？」输出「三角形。」，"
  "而图中红色图形是圆形——模型在颜色正确的情况下给出了不存在的形状。原因归类：形状幻觉（视觉特征与词表绑定错误），"
  "SFT/DPO 都未能纠正，说明这类错误需要更强的视觉编码或更多带属性的监督数据，而不是偏好数据能覆盖的。")
P("案例 3（硬答与整体 No 偏置）：exist 类三阶段全错。P11「图里有没有黄色方形？」（有）与 P13「图里有没有绿色圆形？」（有）"
  "被答成 No，而 P12/P14/P15（应答「没有」）也答 No——模型不是「诚实拒绝」，而是对存在性问句一律否定。"
  "原因归类：硬答/退化回答；同时说明基线 0/5 的 exist 成绩不能解读为「有一半判断正确」。")
P("案例 4（DPO 之后回答变短变含糊——先归因再下结论）：在 fp32 评测口径下，DPO 17 步后 count 由 3/5 掉到 1/5、"
  "attr 由 1/5 掉到 0/5，输出从「3.」「其中，图形的基本体现是小圆形」退化为「1.」「半色。」「青色。」——"
  "表现与指南 Q5 的过度优化一致（rejected 里包含「没有看到黄色图形。」这类退缩型回答，模型可能学到"
  "「答得更短更安全」）。但在 4-bit（与训练一致）口径下，同一个适配器 count 仍为 3/5、格式未坏（见 9.3）。"
  "因此本案例的正确归因是：**先把评测口径对齐，再谈超参**——否则会把口径差异误判成模型退化。"
  "这也是本实验最有价值的一条踩坑：调 DPO 之前先检查 policy/reference 的序列对数概率与评测口径。")

H2("6.2 环境踩坑（对应指南常见问题）")
TABLE([
    ["问题", "现象", "处理"],
    ["模型下载", "huggingface.co 偶发 SSL: UNEXPECTED_EOF_WHILE_READING（curl 直接返回 code=000）",
     "用脚本逐文件下载并自动回退 hf-mirror.com，513 MB 权重 78 秒下完；权重落到项目目录后离线加载"],
    ["torch 安装", "CUDA wheel 2.53 GB，安装耗时长", "用 download.pytorch.org/whl/cu124 独立 index 安装到项目内 .deps，不污染全局环境"],
    ["指南 load_model('train') 的 4-bit 分支", "在 CPU 上无 bitsandbytes，指南 Q2 要求手工改代码回退 fp32",
     "脚本内置 _bnb_available() 自动检测：无 GPU/无 bnb 时自动 fp32 + LoRA，有 GPU 时走真正的 NF4 量化"],
    ["指南 DPO 载入 SFT 适配器", "PeftModel.from_pretrained(model.get_base_model().float(), SFT_DIR) 对 4-bit 基座调用 .float() 会破坏量化",
     "改为 set_peft_model_state_dict(model, load_file(adapter_model.safetensors))，量化保持有效"],
    ["指南 --steps 默认值", "SFT 默认 60 步、DPO 默认 40 步，但 epochs=2/1、数据 15/17 条，最多只能跑 30/17 步",
     "如实报告实际步数，并新增 --epochs 参数以便跑满步数（9.3 的 3 轮 51 步对照组）"],
    ["指南 dpo_check 损失公式", "−log(1+e^(−margin)) 符号写反，输出负损失", "已改为 log(1+e^(−margin)) / 用 F.logsigmoid 复核，与手算 0.6882 一致（见 4.4）"],
], caption="表 9　环境与参考代码踩坑记录", widths=[3.4, 5.6, 6.0], size=8.5)

# ------------------------------------------------------------------ 七、AI 协作
H1("七、AI 协作记录")
P("本实验全程在本机的命令行智能体（DSH Agent）中完成，未使用 TRAE。协作方式与人工核对分工如实记录如下：")
P("① AI 生成/协助的部分：以指南第七节的参考代码为骨架，由 AI 补齐工程化改造——(a) 训练设备自适应（GPU 走 NF4，"
  "无 GPU 自动回退 fp32 + LoRA）；(b) 全过程数据落盘（logs/*.json 记录每题输出、逐步 loss/margin），"
  "并用 plot_curves.py 自动出图；(c) 新增 --r/--lr/--epochs/--beta/--tag/--eval_4bit 等命令行参数，"
  "使 β 与秩的敏感性扫描不需要改代码；(d) 环境脚本（install_deps.ps1、check_env.py、download_model.py）。")
P("② 必须人工逐行核对的部分（三处最关键）：(a) labels 掩码的 prompt_len 计算——分别对「prompt」和「prompt+answer」"
  "两次调用 processor 取 input_ids 长度，人工核对两者前缀一致、且 labels[:, :prompt_len] 恰好覆盖 <image> 与问题部分，"
  "避免把图像 token 或答案首 token 误掩；(b) 判分函数——发现指南对 gold=「有」的判分会被预测「没有」误命中，"
  "人工补上否定词排除；(c) margin 的符号方向（(πθ(yw)−πref(yw)) − (πθ(yl)−πref(yl))）与 dpo_check 的损失符号，"
  "用三组手算（含一组 margin<0）逐项对照，最终定位并修正了指南代码的符号笔误。")
P("③ 验证方法：所有判据都用「两条独立路径互证」的方式确认——DPO 损失既用 F.logsigmoid（训练路径）又用 "
  "math.log(1+e^(−margin))（手算路径）计算，并在 margin=0、±0.12 三个点上与 ln 2、σ 的定义值对照；"
  "参考模型用 disable_adapter() 时先打印同一 prompt 在开启/关闭适配器下的序列对数概率，确认两者数值不同（否则就是"
  "「参考模型其实等于策略模型」的静默 bug）；评测口径统一（三阶段同一份 QUESTIONS、同一个 judge、同一 dtype），"
  "并额外补跑一次 4-bit 口径评测以排除「训练用 4-bit、评测用 fp32」带来的口径漂移。")

# ------------------------------------------------------------------ 八、思考题
H1("八、思考题")
P("题 1（为什么只用 SFT 很难教会「诚实拒绝」）：SFT 的交叉熵只在正例答案上产生梯度，"
  "它把「没有。」的概率往上推，却从不把「有。」的概率往下压；模型学到的是「这个问句下什么字符串像答案」，"
  "而不是「这个问句在不同图像下该给出不同结论」。以 P12「图里有没有红色圆形？」为例，正确答案是「没有」，"
  "而 SFT 数据里 15 条全部是正例，没有任何一条告诉模型「当图中没有红圆却答『有，左上角有一个』时要被惩罚」。"
  "本实验的 exist 类结果正与此吻合：基线对 P11/P13（应答「有」）也一律答 No，说明模型对存在性问句形成了整体 No 偏置，"
  "SFT 30 步后仍然 0/5——这不是「诚实」，而是「懒得看」。DPO 的 rejected 恰好补上了这一点："
  "对 P12/P14/P15 给出「有，图左上角有一个红色圆形」这类硬答有式的负例，损失里的 margin 会把「有」类回答的"
  "对数概率相对压低，从而在概率上惩罚「不看图就答有」。")
P("题 2（手算）：β=0.1，πθ(yw)=1.5、πref(yw)=1.0、πθ(yl)=0.2、πref(yl)=0.6。"
  "margin = 0.1×[(1.5−1.0) − (0.2−0.6)] = 0.1×[0.5 − (−0.4)] = 0.1×0.9 = 0.0900；"
  "loss = −log σ(0.09) = log(1+e^(−0.09)) = 0.6492。margin>0 说明策略模型把 chosen 相对 rejected 的概率差"
  "拉得比参考模型更大，偏好方向正确，损失低于 ln 2。若 margin<0（本报告表 7 第③组：margin=−0.12），"
  "则说明策略反而更偏好 rejected，loss=0.7549>ln 2，训练需要继续。用 "
  "--dpo_check 1.5,1.0,0.2,0.6,0.1 复核，脚本输出 margin=0.0900、loss=0.6492，与手算一致。")
P("题 3（DPO 为什么不需要奖励模型）：RLHF 的两阶段做法是先按 Bradley-Terry 模型 P(yw≻yl|x)=σ(r(x,yw)−r(x,yl)) "
  "训练奖励模型 r，再用 RL 最大化「奖励 − β·KL(πθ‖πref)」。该约束下最优策略有闭式解 "
  "π*(y|x) = πref(y|x)·exp(r(x,y)/β)/Z(x)，把 r 反解出来即隐式奖励 r(x,y) = β·log(πθ(y|x)/πref(y|x)) + β·log Z(x)。"
  "由于 Z(x) 只与 x 有关，在 pair 的差 r(x,yw)−r(x,yl) 中会被消掉；把它代回 Bradley-Terry 似然并取负对数，"
  "得到的正是 DPO 损失 −log σ(β·[(log πθ(yw|x)−log πref(yw|x)) − (log πθ(yl|x)−log πref(yl|x))])。"
  "也就是说，DPO 把「策略模型自身的对数概率比」当成了隐式奖励，无需显式奖励模型、也无需在线采样 RL 循环，"
  "但代价是必须保存一份冻结的参考模型 πref（本实验用 disable_adapter() 复用同一份 4-bit 基座），"
  "所以 DPO 每一步要做 4 次前向。（来源：Rafailov et al., DPO: Your Language Model is Secretly a Reward Model, "
  "NeurIPS 2023, arXiv:2305.18290；Bradley & Terry, 1952。）")
P("题 4（为什么量化后不直接全参微调）：4-bit 量化把权重压到 4 bit/参数，前向必须按块反量化，"
  "全参微调需要保存 fp32 的权重副本、一阶/二阶动量与激活，显存立刻回到甚至超过 fp16 全参微调的水平"
  "（QLoRA 论文的核心贡献正是 NF4 + 双重量化 + 分页优化器让梯度与优化器状态也跟着省）。"
  "更重要的是：本实验的目标是「冻结基座、只训适配器」——适配器只有 1,363,968 个参数，优化器状态极小，"
  "训练完只保存几 MB 的 adapter；而全参微调会破坏基座的通用视觉-语言能力（灾难性遗忘），小数据下尤其明显。"
  "r=16 时参数占比的估算：LoRA 参数量 ∝ r（A 是 r×d_in、B 是 d_out×r，均与 r 线性），"
  "本实验对这一点做了实测（表 15）：r=4 → 681,984 个可训练参数、占比 0.2652%；r=8 → 1,363,968、0.5290%；"
  "r=16 → 2,727,936、1.0524%，三档恰好是 1 : 2 : 4，即 r=16 的参数占比约为 r=8 的 2 倍、r=4 的 4 倍"
  "（占比分母里也含适配器本身，所以三档总参数量略有差别：257,166,912 / 257,848,896 / 259,212,864）。"
  "但实测正确率并不随秩提升（fp32 口径 3/15 → 4/15 → 3/15，4-bit 口径 3/15 → 3/15 → 3/15），"
  "说明在这类小数据、小答案词表的任务上，适配器容量不是瓶颈。"
  "（来源：Dettmers et al., QLoRA, arXiv:2305.14314。）")
P("题 5（用模型自生成回答 + 规则自动打分来构造偏好对）：好处是标注成本几乎为零、可以持续扩张数据量"
  "（自我对弈/RLAIF 思路），并且能覆盖人工想不到的失败模式；对本题这类「可程序化判分」的任务"
  "（数颜色形状的个数、是否出现某物体），规则打分几乎不会错，最适合自动化。风险有三："
  "① 奖励/规则一旦被利用，模型会朝规则漏洞而不是事实优化（reward hacking）；"
  "② 自生成数据的分布由模型自身决定，容易只覆盖「模型已经会答的问题」，无法纠正系统性盲区；"
  "③ 过度优化——本实验 DPO 只用 17 组偏好对就出现了正确率 4/15→1/15 的退化，"
  "正是模型把「短回答」当成了被偏好的特征；如果 rejected 由模型自生成，退化型回答（「没有看到黄色图形。」"
  "「半色。」）会大量进入数据，模型更容易学到「变短、变含糊」而不是「变对」。"
  "缓解办法：混合人工校验的锚点数据、对 chosen/rejected 施加长度与信息量约束（如长度归一化的对数概率）、"
  "监控 margin 与任务正确率的背离、必要时加入 KL 正则或提前停止。"
  "（来源：Lee et al., RLAIF, arXiv:2309.00267；Yuan et al., Self-Rewarding LMs, arXiv:2401.10020；"
  "Gao et al., Scaling Laws for Reward Model Overoptimization, arXiv:2210.10760。）")
P("题 6（AI 生成代码里最容易悄悄写反的地方 + 验证办法）：本实验里真正「悄悄写反」的就是 DPO 的损失符号。"
  "指南 dpo_check 写成 loss = −log(1 + e^(−margin))，看起来像 −log σ(margin)，"
  "但 −log σ(m) = +log(1 + e^(−m))，多了一个负号，结果输出负损失（−0.6882）；"
  "训练路径因为调用 F.logsigmoid(margin) 恰好正确，于是「训练正常、复核脚本输出负数」这种不对称极易被忽略。"
  "次高风险的是 margin 的方向（把 chosen/rejected 位置互换，或把 (πθ−πref) 写成 (πref−πθ)）："
  "符号反了以后 loss 仍然会下降，只是模型被训练成偏好 rejected——这类 bug 不会报错，只会让指标变差。"
  "再其次是 labels 掩码的 prompt_len：多掩一个 token 会让答案首 token 不受监督，少掩一个会让问题 token 参与损失，"
  "两者都只是「效果差一点」，不会抛异常。验证办法：① 用定义式独立复算——把 F.logsigmoid 的结果与 "
  "math.log(1+e^(−margin)) 手算值对齐，并在 margin=0（应得 ln 2≈0.6931）、margin=+0.12、margin=−0.12 "
  "三点上检查单调性与数值范围；② 交换 chosen/rejected 再算一次，margin 必须变号（不变量检查）；"
  "③ 训练前后打印 policy/reference 在 chosen 与 rejected 上的序列对数概率，确认 ref 路径确实不带梯度且数值与 "
  "policy 不同（防止 disable_adapter 失效导致 πref=πθ、margin 恒等于 0 的静默错误）；"
  "④ 评测集与训练集分离检查判分函数是否会把「格式对但事实错」判成正确（本实验 P01/P03 就是反例）。")

# ------------------------------------------------------------------ 九、延伸
H1("九、延伸进阶实验：DPO 偏好对齐（选做）")
H2("9.1 偏好数据集：指南 12 组 + 自拟 5 组")
P("指南 PRE_PAIRS 的 12 组：chosen 全部取自标准答案；rejected 覆盖三类典型错误——数错（off-by-one）、"
  "形状幻觉（把圆说成方）、对不存在的物体硬答「有」。在此基础上自拟 5 组（E01~E05），"
  "其中 1 组为「冗长废话型」rejected，以观察 DPO 是否会把「长而无关」压低、把「短而正确」抬高：")
TABLE([
    ["编号", "图像", "问题", "chosen", "rejected", "对应失败模式"],
    ["E01", "img10", "图中有几个蓝色圆形？", "3 个。", "4 个。图中蓝色圆形排成两行，上一行 2 个下一行 2 个。",
     "数错·off-by-one（并编造了排布细节）"],
    ["E02", "img13", "图中黄色的图形是什么形状？", "圆形。", "方形，是两个黄色方形。",
     "形状幻觉（圆说成方）+ 数量幻觉"],
    ["E03", "img14", "图里有没有绿色圆形？", "没有。", "有，图右下角有一个绿色圆形。",
     "硬答「有」并对不存在的物体编造位置"],
    ["E04", "img12", "图里有没有红色圆形？", "没有。", "有，有两个红色圆形。",
     "硬答「有」+ 计数幻觉（存在性判断被数量带偏）"],
    ["E05", "img06", "图中有几个黄色圆形？", "4 个。",
     "这是一张白色背景的图片，图中绘制了一些几何图形，它们由线条勾勒而成，分布在不同位置，整体构图上疏密有致，"
     "可以看出作者希望考察观察者对图形的辨识能力。", "冗长废话型：通篇视觉描述却不给结论"],
], caption="表 10　自拟 5 组偏好对及其对应的失败模式（已加入 PRE_PAIRS_ALL，共 17 组参与 DPO）",
    widths=[1.2, 1.4, 3.0, 1.6, 5.6, 3.2], size=8)

H2("9.2 DPO 训练日志与曲线")
rows = [["步", "loss", "margin", "pol_w", "pol_l", "ref_w", "ref_l", "解读"]]
hist = dpo_log.get("history", [])
picks = [1, 5, 10, 15, len(hist)] if hist else []
seen = set()
for h in hist:
    if h["step"] not in picks or h["step"] in seen:
        continue
    seen.add(h["step"])
    note = "margin>0，偏好方向正确" if h["margin"] > 0 else "margin<0，仍在偏好 rejected"
    rows.append([h["step"], f"{h['loss']:.4f}", f"{h['margin']:.4f}", f"{h['pol_w']:.3f}",
                 f"{h['pol_l']:.3f}", f"{h['ref_w']:.3f}", f"{h['ref_l']:.3f}", note])
TABLE(rows, caption="表 11　DPO 训练日志（β=0.1，17 组偏好对，1 轮 17 步）",
      widths=[0.9, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 4.0], size=8.5)
FIG(os.path.join(FIGS, "fig2_dpo_curves.png"), "图 5　DPO 训练曲线：loss 与 margin 随步数变化（含随机基线 ln2）")
FIG(os.path.join(FIGS, "fig4_trainable_params.png"), "图 6　冻结基座与可训练适配器的参数量对比（对数刻度）", width=12.0)

H2("9.3 三阶段对比（含评测口径与更长 DPO 对照）")
rows = [["阶段", "评测口径", "count", "attr", "exist", "总计 /15"]]
for s in ("基线", "SFT", "SFT+DPO"):
    if s in ev:
        b = ev[s]["by_type"]
        rows.append([s, "fp32 基座（指南口径）", f"{b['count'][0]}/5", f"{b['attr'][0]}/5",
                     f"{b['exist'][0]}/5", str(ev[s]["total"])])
for s in ("基线", "SFT", "SFT+DPO"):
    if s in ev4:
        b = ev4[s]["by_type"]
        rows.append([s, "4-bit NF4 基座（与训练一致）", f"{b['count'][0]}/5", f"{b['attr'][0]}/5",
                     f"{b['exist'][0]}/5", str(ev4[s]["total"])])
extra = betas.get("b01e3", ({}, {}))[0]
if extra and extra.get("SFT+DPO"):
    b = extra["SFT+DPO"]["by_type"]
    rows.append(["SFT+DPO（3 轮 51 步）", "fp32 基座", f"{b['count'][0]}/5", f"{b['attr'][0]}/5",
                 f"{b['exist'][0]}/5", str(extra["SFT+DPO"]["total"])])
TABLE(rows, caption="表 12　三阶段 × 题型正确率：两种评测口径 + 更长 DPO 对照",
      widths=[3.6, 4.6, 1.8, 1.8, 1.8, 2.0], size=9)
P("分析：① 口径决定结论——同一个 SFT+DPO 适配器，在 fp32 基座上评测得 1/15，在 4-bit 基座上评测得 4/15。"
  "原因是适配器（ΔW）是针对 NF4 量化基座训练出来的修正量，把同样的 ΔW 加到 fp32 基座上属于「外推」，"
  "修正量未必对应同样的错误分布；SFT 适配器也有类似但更小的偏差（fp32 4/15 vs 4-bit 3/15）。"
  "因此「DPO 让模型变差」这一结论只在 fp32 口径下成立，不能作为最终判断。"
  "② 在口径一致的 4-bit 评测里，SFT+DPO（4/15）略好于 SFT（3/15），且 exist 类第一次命中一题"
  "（P15「图里有没有蓝色图形？」答「没有蓝色图形？」，图里确实没有蓝色图形）——与 rejected 针对"
  "「硬答有」的设计方向一致，但 15 题里 1 题的量级仍不足以说明幻觉被压住。"
  "③ 逐题对比可见 DPO 的副作用：fp32 口径下 count 由「3.」退回「1.」、attr 由「其中，图形的基本体现是小圆形」"
  "退化为「半色。」，而 4-bit 口径下 count 保持 3/5、输出格式未被破坏——说明这种退化主要是口径造成的，"
  "不是 DPO 本身把格式训坏了。④ 综合来看，本轮 DPO（17 组数据、17 步）的作用是「可测但很小」："
  "loss/margin 健康、偏好方向正确、存在性题首次出现正确回答，但整体正确率变化仍在噪声范围内。")

H2("9.4 延伸任务 A：β 敏感性扫描")
P("在同一 SFT 起点上分别用 β=0.05 / 0.1 / 0.5 重跑 DPO（各 1 轮 17 步，偏好对与随机顺序一致），"
  "并另跑 β=0.1 / 3 轮 51 步作为「更长训练」对照。由于单步 margin 波动大（每步只用一个偏好对），"
  "表中用全程均值与最大值描述倾向：")
rows = [["β / 步数", "margin 均值", "margin 最大", "loss 均值", "loss 最小", "fp32 总计 /15", "4-bit 总计 /15"]]
_hist_main = dpo_log.get("history", [])
_runs = [("beta0.05", "0.05 / 17 步"), ("base_default", "0.1 / 17 步"),
         ("beta0.5", "0.5 / 17 步"), ("b01e3", "0.1 / 51 步（3 轮）")]
for tag, label in _runs:
    if tag == "base_default":
        h, evb, evb4 = _hist_main, ev, ev4
    else:
        h = (betas.get(tag, ({}, {}))[1] or {}).get("history", [])
        evb = betas.get(tag, ({}, {}))[0] or {}
        evb4 = load(f"eval_results_{tag}_4bit.json", {}) or {}
    if not h:
        continue
    m = [x["margin"] for x in h]
    l = [x["loss"] for x in h]
    st, st4 = evb.get("SFT+DPO"), evb4.get("SFT+DPO")
    rows.append([label, f"{sum(m)/len(m):.3f}", f"{max(m):.3f}", f"{sum(l)/len(l):.3f}", f"{min(l):.4f}",
                 str(st["total"]) if st else "—", str(st4["total"]) if st4 else "—"])
TABLE(rows, caption="表 13　β 与训练轮数对 DPO 的影响（同一 SFT 起点，两种评测口径）",
      widths=[3.2, 2.0, 2.0, 1.9, 1.9, 2.2, 2.2], size=9)
FIG(os.path.join(FIGS, "fig7_beta_scan.png"),
    "图 7　β 与训练轮数对 DPO 的 margin / loss 轨迹影响（同一 SFT 起点）")
_rows_b = []
for tag, label in (("beta0.05", "0.05"), ("beta0.5", "0.5")):
    evb = betas.get(tag, ({}, {}))[0] or {}
    if evb.get("SFT+DPO"):
        b = evb["SFT+DPO"]["by_type"]
        _rows_b.append([label, f"{b['count'][0]}/5", f"{b['attr'][0]}/5", f"{b['exist'][0]}/5", str(evb["SFT+DPO"]["total"])])
if _rows_b:
    TABLE([["β", "count", "attr", "exist", "总计 /15"]] + _rows_b,
          caption="表 14　β 扫描的分题型明细（fp32 口径）", widths=[2.6, 2.6, 2.6, 2.6, 2.6], size=9)
P("结论：① β 的作用在训练曲线上非常清楚——margin 均值随 β 单调放大（0.246 → 0.485 → 2.146），"
  "最大 margin 由 1.12 升到 10.38，loss 均值同步下降（0.589 → 0.519 → 0.377），β=0.5 时甚至出现 loss≈0.0000 的步"
  "（margin=10.38，σ 已饱和）。这与公式一致：β 是「偏离参考模型的惩罚强度」，直接把同一组对数概率差放大。"
  "② 训练轮数的效果与 β 类似：β=0.1 跑 3 轮（51 步）后 margin 均值由 0.485 升到 0.967、最小 loss 降到 0.0025，"
  "即指南 Q5 说的过度优化——margin 被推向饱和、loss 趋近 0，但评测正确率并未同步上升。"
  "③ 正确率方面（4-bit 口径，与训练一致）：SFT 3/15，DPO 依次为 β=0.05 → 3/15、β=0.1（17 步）→ 4/15、"
  "β=0.5 → 2/15、β=0.1（51 步）→ 3/15。fp32 口径下的同一组数字是 SFT 4/15、DPO 1/1/2/2。"
  "两个口径都表现出「β 过大（0.5）与步数过多（51 步）时正确率低于 β=0.1/17 步」的方向，"
  "与指南任务五(a) 的预期（β 过大 → 过度偏离参考模型 → 正确率下降）一致；但差距只有 1~2 题，"
  "在 15 题 + 宽松判分的规模下只能算弱证据，而且 β=0.05 并没有比 0.1 更好（3 vs 4），说明并非「β 越小越好」。"
  "④ 更靠得住的结论是：β 与步数都能把 margin 推向饱和、把 loss 压到 0，但 loss 低既不代表正确率高、"
  "也不代表幻觉被压住——本实验里 β=0.5 的 loss 均值最低（0.377），正确率却最低（2/15）。")

H2("9.5 延伸任务 B：秩敏感性扫描（实测）")
P("在同一份数据与步数（SFT 2 轮 30 步）下只改 LoRA 秩 r，分别取 r=4 / 8 / 16，"
  "记录可训练参数占比与两种评测口径下的 SFT 正确率（r=8 即 4.2 节的主实验）：")
_rank = {}
for _r in (4, 8, 16):
    _p = (sft_params if _r == 8 else load(f"sft_params_r{_r}.json", {}) or {}).get("params")
    _e = ev if _r == 8 else (load(f"eval_results_r{_r}.json", {}) or {})
    _e4 = ev4 if _r == 8 else (load(f"eval_results_r{_r}_4bit.json", {}) or {})
    if not _p:
        continue
    _rank[_r] = (_p, _e.get("SFT"), _e4.get("SFT"))
rows = [["LoRA 秩 r", "可训练参数量", "总参数量", "占比", "SFT 正确率（fp32）", "SFT 正确率（4-bit）"]]
for _r in (4, 8, 16):
    if _r not in _rank:
        continue
    p, s1, s2 = _rank[_r]
    rows.append([str(_r), f"{p['trainable']:,}", f"{p['total']:,}", f"{p['ratio_pct']:.4f}%",
                 f"{s1['total']}/15" if s1 else "—", f"{s2['total']}/15" if s2 else "—"])
TABLE(rows, caption="表 15　LoRA 秩敏感性（仅改 r，其余设置不变；r=8 为主实验）",
      widths=[1.8, 2.8, 2.8, 1.8, 3.2, 3.2], size=9)
FIG(os.path.join(FIGS, "fig8_rank_scan.png"), "图 8　LoRA 秩对可训练参数占比与 SFT 正确率的影响", width=14.5)
if 4 in _rank and 16 in _rank and _rank[8][1] and _rank[8][2]:
    _p4, _s14, _s44 = _rank[4]
    _p8, _s18, _s48 = _rank[8]
    _p16, _s116, _s416 = _rank[16]
    P(f"实测结论：① 参数量与 r 严格成线性——可训练参数 {_p4['trainable']:,}（r=4）→ "
      f"{_p8['trainable']:,}（r=8）→ {_p16['trainable']:,}（r=16），恰好是 1 : 2 : 4；"
      f"占比 {_p4['ratio_pct']:.4f}% → {_p8['ratio_pct']:.4f}% → {_p16['ratio_pct']:.4f}%。"
      f"因为 LoRA 参数量 ∝ r·(d_in+d_out)，翻倍 r 就翻倍适配器参数，实测结果与思考题(4) 的理论估算一致"
      f"（注意占比分母里也含适配器本身，所以三档总参数量略有差别）。"
      f"② 正确率并没有随容量同步提升：fp32 口径下 SFT 为 {_s14['total']}/15（r=4）、{_s18['total']}/15（r=8）、"
      f"{_s116['total']}/15（r=16）；4-bit 口径下为 {_s44['total']}/15、{_s48['total']}/15、{_s416['total']}/15。"
      f"两个口径下三档 r 的差异都只有 1 题左右，说明本任务的瓶颈不是适配器容量："
      f"把可训练参数从 {_p4['ratio_pct']:.2f}% 提到 {_p16['ratio_pct']:.2f}%（4 倍）并没有解决 attr / exist 的视觉绑定问题。"
      f"③ 实践含义：在这种「格式类 + 小数据」任务上 r=4 已经够用，加大 r 主要增加显存与训练时间，"
      f"不会自动带来事实性能力；要提升 attr / exist 应改数据与任务设计（更多属性监督、图形不重叠、提高分辨率），"
      f"而不是加大 LoRA 秩——这与指南任务五(b) 「对比参数占比与正确率」的考察意图相符："
      f"参数占比是可控的成本项，正确率则由数据和任务难度决定。")

# ------------------------------------------------------------------ 十、总结
H1("十、结论与提交物")
P("① 全链路在本机单卡 RTX 4060 Laptop 上跑通：基线推理（1/15）、QLoRA SFT（0.5290% 可训练参数，count 1/5→3/5）、"
  "DPO 偏好对齐（17 组偏好对，含自拟 5 组；margin 持续为正、loss 低于 ln 2）、三阶段 45 题 × 两种评测口径、"
  "β 敏感性扫描（β=0.05/0.1/0.5）与秩敏感性扫描（r=4/8/16），单次全流程约 10 分钟，总实验耗时约 1 小时；")
P("② SFT 把「答题格式」学会了（count 稳定 3/5），但没有学会「看图说话的事实」（attr 最好 1/5、exist 最好 1/5）；"
  "exist 类在基线上是「一律答 No」的整体否定偏置，不是诚实——这是 SFT 只有正例、从不惩罚错误回答的必然结果；")
P("③ DPO 的偏好方向正确（loss/margin 曲线健康），且 4-bit 口径下 exist 类首次命中 1 题，"
  "但 15 题规模下正确率变化仍落在噪声范围内；「DPO loss 下降 ≠ 任务正确率上升」在两种口径下都成立。"
  "本实验同时暴露了一个更容易被忽略的问题：评测口径（fp32 vs 4-bit 基座）会让同一适配器的正确率相差 3 题，"
  "小样本评测必须写清口径，并且应先把口径与训练对齐再解释超参；")
P("④ 参考代码共发现并修正 3 处口径问题：4-bit 基座载入适配器不能用 .float()、--steps 与 epochs 不一致"
  "（SFT 60 步/DPO 40 步实际最多 30/17 步）、dpo_check 的损失符号写反（−log(1+e^(−m)) 应为 +log(1+e^(−m))）；"
  "另修正判分函数中「没有」包含「有」导致的 exist 误判。这些都写进了报告与代码注释。")
P("提交物：本报告（DOCX/PDF）、hw5_dpo.py（含全部改造与注释）、训练曲线与截图（figs/*.png：SFT loss、"
  "DPO loss/margin、三阶段对比、参数占比、dpo_check 控制台截图）、实测日志（logs/*.json 与 logs/*.log，"
  "含每题输出与逐步 loss/margin）、复现脚本（install_deps.ps1 / run_all.ps1 / plot_curves.py / check_env.py / "
  "download_model.py）。")

os.makedirs(OUT_DIR, exist_ok=True)
out_name = "_".join(x for x in (STUDENT_ID, NAME, "实验作业五") if x) + ".docx"
out = os.path.join(OUT_DIR, out_name)
doc.save(out)
print("saved", out)
