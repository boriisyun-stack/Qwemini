"""Experimental Qwen3-Next MLX runner using the split expert directory.

The base/router/shared weights are read from the original MLX checkpoint one
shard at a time. Routed experts are loaded from model_q4_mlx_paged on demand.
This is intentionally a prototype: it favors correctness and observability
over maximum throughput.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock
from mlx_lm.models.switch_layers import _scatter_unsort, _gather_sort, swiglu
from mlx_lm.utils import load_config, load_tokenizer, _get_classes


class LayerExpertCache:
    """One LRU per MoE layer, shared by its three projection linears.

    An expert shard contains gate, up, and down tensors together.  Keeping
    three independent caches meant a routed expert transition opened the same
    safetensors file three times, which caused visible stalls on cold routes.
    """

    def __init__(self, root: Path, layer: int, cache_experts: int):
        self.root = root
        self.layer = layer
        self.cache_experts = cache_experts
        self.cache: OrderedDict[int, dict[str, mx.array]] = OrderedDict()

    def load(self, expert: int) -> dict[str, mx.array]:
        if expert in self.cache:
            value = self.cache.pop(expert)
            self.cache[expert] = value
            return value
        path = self.root / "experts" / f"layer_{self.layer:02d}" / f"expert_{expert:03d}.safetensors"
        # Each file contains the expert-axis slice [1, ...].
        value = {k: v.squeeze(0) for k, v in mx.load(str(path)).items()}
        self.cache[expert] = value
        while len(self.cache) > self.cache_experts:
            self.cache.popitem(last=False)
        return value


class PagedSwitchLinear(nn.Module):
    def __init__(self, expert_cache: LayerExpertCache, projection: str):
        super().__init__()
        self.expert_cache = expert_cache
        self.projection = projection

    def __call__(self, x, indices, sorted_indices=False):
        host_indices = np.asarray(indices).astype(np.int32)
        unique = sorted(set(int(v) for v in host_indices.reshape(-1)))
        mapping = {expert: i for i, expert in enumerate(unique)}
        local = np.asarray([mapping[int(v)] for v in host_indices.reshape(-1)], dtype=np.int32)
        local = mx.array(local.reshape(host_indices.shape))
        loaded = [self.expert_cache.load(expert) for expert in unique]
        weight = mx.stack([item[f"{self.projection}.weight"] for item in loaded])
        scales = mx.stack([item[f"{self.projection}.scales"] for item in loaded])
        biases = mx.stack([item[f"{self.projection}.biases"] for item in loaded])
        return mx.gather_qmm(
            x, weight, scales, biases, rhs_indices=local,
            transpose=True, group_size=64, bits=4, mode="affine",
            sorted_indices=sorted_indices,
        )


class PagedSwitchGLU(nn.Module):
    def __init__(self, root: Path, layer: int, cache_experts: int = 7):
        super().__init__()
        self.expert_cache = LayerExpertCache(root, layer, cache_experts)
        self.gate_proj = PagedSwitchLinear(self.expert_cache, "gate_proj")
        self.up_proj = PagedSwitchLinear(self.expert_cache, "up_proj")
        self.down_proj = PagedSwitchLinear(self.expert_cache, "down_proj")
        self.activation = swiglu

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        x_up = self.up_proj(x, indices)
        x_gate = self.gate_proj(x, indices)
        x = self.down_proj(self.activation(x_up, x_gate), indices)
        return x.squeeze(-2)


def load_paged(base: Path, paged: Path, top_k: int, cache_experts: int):
    config = load_config(base)
    config["num_experts_per_tok"] = top_k
    model_class, args_class = _get_classes(config)
    model = model_class(args_class.from_dict(config))

    # Quantize the model skeleton before loading. MLX quantized embeddings and
    # linears intentionally have packed shapes (e.g. embedding width 256 for
    # a logical width of 2048); loading into an unquantized skeleton is wrong.
    index = json.loads((base / "model.safetensors.index.json").read_text())
    weight_names = set(index["weight_map"])
    quant_config = config.get("quantization", {"group_size": 64, "bits": 4, "mode": "affine"})

    def quant_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if path in quant_config and isinstance(quant_config[path], dict):
            return quant_config[path]
        return f"{path}.scales" in weight_names

    nn.quantize(
        model,
        group_size=quant_config.get("group_size", 64),
        bits=quant_config.get("bits", 4),
        mode=quant_config.get("mode", "affine"),
        class_predicate=quant_predicate,
    )

    # Load only non-routed tensors, shard by shard. This avoids retaining the
    # 42GB checkpoint in the Python dictionary.
    weights = {}
    shard_paths = sorted(base.glob("base-*.safetensors")) or sorted(base.glob("model-*.safetensors"))
    for shard in shard_paths:
        for name, value in mx.load(str(shard)).items():
            if ".mlp.switch_mlp." not in name:
                weights[name] = value
    model.load_weights(list(weights.items()), strict=False)
    del weights

    # Replace the random/in-memory SwitchGLU modules after base weights load.
    for layer_idx, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
            layer.mlp.top_k = top_k
            layer.mlp.switch_mlp = PagedSwitchGLU(paged, layer_idx, cache_experts)
    model.eval()
    tokenizer = load_tokenizer(base)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("model_q4_mlx_base"))
    parser.add_argument("--paged", type=Path, default=Path("model_q4_mlx_paged"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cache-experts", type=int, default=24)
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    model, tokenizer = load_paged(args.base, args.paged, args.top_k, args.cache_experts)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
    )
    from mlx_lm import generate
    print(generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=True))


if __name__ == "__main__":
    main()
