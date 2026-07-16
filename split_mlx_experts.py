"""Split Qwen3-Next MLX quantized switch_mlp tensors by expert."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHT_SUFFIXES = ("weight", "scales", "biases")


def tensor_slice(path: Path, name: str, expert: int) -> torch.Tensor:
    # Torch preserves BF16 metadata used by MLX safetensors; NumPy does not.
    with safe_open(str(path), framework="pt") as handle:
        view = handle.get_slice(name)
        shape = view.get_shape()
        if not shape or shape[0] <= expert:
            raise ValueError(f"{name} has no expert axis 0: shape={shape}")
        return view[expert : expert + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--top-k", type=int, default=7)
    args = parser.parse_args()
    index_path = args.source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    names = sorted(name for name in weight_map if ".switch_mlp." in name)
    if not names:
        raise SystemExit("No switch_mlp tensors found")

    grouped: dict[tuple[int, str, str], str] = {}
    for name in names:
        parts = name.split(".")
        layer = int(parts[2])
        projection = parts[-2]
        kind = parts[-1]
        if kind in WEIGHT_SUFFIXES:
            grouped[(layer, projection, kind)] = name
    layers = sorted({layer for layer, _, _ in grouped})
    projections = ["gate_proj", "up_proj", "down_proj"]
    sample = grouped[(layers[0], projections[0], "weight")]
    with safe_open(str(args.source / weight_map[sample]), framework="pt") as handle:
        experts = handle.get_slice(sample).get_shape()[0]
    if not 1 <= args.top_k <= experts:
        raise SystemExit(f"--top-k must be between 1 and {experts}")

    args.destination.mkdir(parents=True, exist_ok=True)
    for path in args.source.iterdir():
        if path.is_file() and path.name not in {"model.safetensors.index.json"} and not path.name.endswith(".safetensors"):
            shutil.copy2(path, args.destination / path.name)

    manifest = {"source": str(args.source), "experts": experts, "top_k": args.top_k, "layers": {}}
    for layer in layers:
        layer_dir = args.destination / "experts" / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        manifest["layers"][str(layer)] = []
        for expert in range(experts):
            tensors = {}
            for projection in projections:
                for kind in WEIGHT_SUFFIXES:
                    name = grouped.get((layer, projection, kind))
                    if name is None:
                        raise ValueError(f"missing {projection}.{kind} tensor for layer {layer}")
                    tensors[f"{projection}.{kind}"] = tensor_slice(args.source / weight_map[name], name, expert)
            output = layer_dir / f"expert_{expert:03d}.safetensors"
            save_file(tensors, str(output), metadata={"layer": str(layer), "expert": str(expert)})
            manifest["layers"][str(layer)].append(str(output.relative_to(args.destination)))

    (args.destination / "paging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"split {len(layers)} layers x {experts} experts; top_k={args.top_k}")


if __name__ == "__main__":
    main()
