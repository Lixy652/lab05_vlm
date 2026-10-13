# -*- coding: utf-8 -*-
"""下载 SmolVLM-256M-Instruct 权重到项目内目录（等价于 huggingface_hub.snapshot_download）。"""
import os
import sys
import time
import urllib.request

REPO = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SmolVLM-256M-Instruct")
FILES = ["added_tokens.json", "chat_template.json", "config.json", "generation_config.json",
         "merges.txt", "model.safetensors", "preprocessor_config.json", "processor_config.json",
         "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]
BASES = ["https://huggingface.co", "https://hf-mirror.com"]


def fetch(name):
    out = os.path.join(DEST, name)
    if os.path.exists(out) and os.path.getsize(out) > 0 and name != "model.safetensors":
        print(f"[skip] {name} exists ({os.path.getsize(out)} B)", flush=True)
        return True
    for base in BASES:
        url = f"{base}/{REPO}/resolve/main/{name}"
        try:
            t0 = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=60) as r, open(out, "wb") as f:
                total, done = int(r.headers.get("Content-Length") or 0), 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total and done % (50 << 20) < (1 << 20):
                        print(f"  {name}: {done/1e6:.1f}/{total/1e6:.1f} MB", flush=True)
            print(f"[ok] {name} {os.path.getsize(out)} B in {time.time()-t0:.1f}s ({base})", flush=True)
            return True
        except Exception as e:
            print(f"[fail] {base} {name}: {type(e).__name__} {e}", flush=True)
    return False


if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    ok = all(fetch(f) for f in FILES)
    print("ALL_OK" if ok else "SOME_FAILED")
    sys.exit(0 if ok else 1)
