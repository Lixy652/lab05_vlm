# -*- coding: utf-8 -*-
"""环境自检：torch/CUDA、bitsandbytes 4-bit、transformers/peft 版本与模型可加载性。"""
import json
import os
import sys

import torch

print("python      :", sys.version.split()[0], sys.executable)
print("torch       :", torch.__version__)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu         :", torch.cuda.get_device_name(0))
    print("capability  :", torch.cuda.get_device_capability(0))
    print("vram total  : %.2f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
    print("cuda version:", torch.version.cuda)
    print("bf16 support:", torch.cuda.is_bf16_supported())
try:
    import bitsandbytes as bnb
    print("bitsandbytes:", bnb.__version__)
except Exception as e:
    print("bitsandbytes: FAILED", e)
import transformers, peft, accelerate
print("transformers:", transformers.__version__)
print("peft        :", peft.__version__)
print("accelerate  :", accelerate.__version__)

here = os.path.dirname(os.path.abspath(__file__))
local = os.path.join(here, "SmolVLM-256M-Instruct")
model_id = local if os.path.isfile(os.path.join(local, "config.json")) else \
    "HuggingFaceTB/SmolVLM-256M-Instruct"
print("model id    :", model_id)

from transformers import AutoProcessor
from PIL import Image
proc = AutoProcessor.from_pretrained(model_id)
img = Image.new("RGB", (256, 256), "white")
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "图中有几个红色圆形？"}]}]
prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
enc = proc(text=[prompt], images=[img], return_tensors="pt")
print("processor OK, input_ids:", tuple(enc["input_ids"].shape),
      "pixel_values:", tuple(enc["pixel_values"].shape))
print("prompt head:", repr(prompt[:120]))
print("ENV_OK")
