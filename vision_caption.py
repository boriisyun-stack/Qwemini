#!/usr/bin/env python3
"""Small, persistent local image-to-English-description worker.

The web server talks to this program over stdin/stdout so its PyTorch model is
kept out of the MLX Qwen process.  One JSON request per line, one response per
line.  Images never leave the Mac.
"""

from __future__ import annotations

import base64
import io
import json
import sys

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_ID = "microsoft/Florence-2-base"


def load_model():
    # CPU is deliberate: the Qwen runtime owns the GPU/unified-memory pressure.
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32,
        # Florence-2's pinned remote implementation predates Transformers'
        # SDPA capability flag used by newer releases.
        attn_implementation="eager",
    ).eval()
    return processor, model


def describe(processor, model, data_url: str) -> str:
    if not data_url.startswith("data:image/") or "," not in data_url:
        raise ValueError("이미지 데이터 형식이 올바르지 않습니다.")
    raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    if len(raw) > 12 * 1024 * 1024:
        raise ValueError("이미지는 12MB 이하만 올릴 수 있습니다.")
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    task = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=task, images=image, return_tensors="pt")
    with torch.inference_mode():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=768,
            num_beams=3,
            do_sample=False,
        )
    text = processor.batch_decode(generated, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=task, image_size=(image.width, image.height)
    )
    caption = parsed.get(task, text) if isinstance(parsed, dict) else str(parsed)
    return str(caption).strip()


def main():
    processor, model = load_model()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            print(json.dumps({"caption": describe(processor, model, request["data_url"])}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
