"""Extract non-routed Qwen3-Next tensors so the 42GB source can be removed."""

import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for p in source.iterdir():
        if p.is_file() and not p.name.startswith("model-") and p.name != "model.safetensors.index.json":
            shutil.copy2(p, destination / p.name)

    index = json.loads((source / "model.safetensors.index.json").read_text())
    new_map = {}
    shards = sorted(set(index["weight_map"].values()))
    for number, shard in enumerate(shards, 1):
        tensors = {}
        with safe_open(str(source / shard), framework="pt") as handle:
            for name in handle.keys():
                if ".mlp.switch_mlp." not in name:
                    tensors[name] = handle.get_tensor(name)
        out_name = f"base-{number:05d}-of-{len(shards):05d}.safetensors"
        save_file(tensors, str(destination / out_name))
        for name in tensors:
            new_map[name] = out_name
        print(out_name, len(tensors))

    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum((destination / f).stat().st_size for f in new_map.values())}, "weight_map": new_map}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

