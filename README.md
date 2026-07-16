# Qwemini — local Qwen assistant

Qwemini is a local, LAN-accessible AI assistant built by Galiboole. It combines
a paged Qwen3-Next runtime with a chat UI, image caption hand-off, tools,
CodeGraph function-level repair, and ProgressiveWriter paragraph-level story
generation.

The model weights are intentionally not stored in Git. GitHub Pages can host
the static project information, but it cannot run the local MLX model; use the
local server below for inference.

This is intentionally a weight-free prototype. It tests top-k reduction and a
byte-bounded LRU before downloading the 80B checkpoint.

After the complete MLX Q4 checkpoint is present, split the stacked expert
tensors without loading the whole model into RAM:

```bash
python3 split_mlx_experts.py model_q4_mlx model_q4_mlx_paged --top-k 10
```

Run the experimental paged loader with the local virtual environment:

```bash
.venv/bin/python paged_mlx.py --top-k 10 --cache-experts 10
```

Start the LAN web UI/API:

```bash
.venv/bin/python paged_server.py --host 0.0.0.0 --port 8000
```

The first smoke test has been verified on this Mac: one token generated with
about 2.0 GiB peak reported MLX memory. `model_q4_mlx_base` contains only the
non-routed weights; routed expert tensors come only from the paged directory.

```bash
python3 -m unittest discover -s . -p 'test_*.py'
python3 memory_status.py
```

The model was trained with 10 routed experts plus one shared expert. The
prototype default is `k=10`, with surviving router gates renormalized. It does
not yet execute Qwen3-Next's hybrid DeltaNet/MoE forward pass or convert
weights; those require a larger-memory host and a model-specific backend.
