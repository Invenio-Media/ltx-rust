# Parity fixtures

Each script in this directory generates fixture files for a crate's parity tests.

Run with the reference venv (`LTX-2/.venv/bin/python`):

```sh
python -W ignore tools/parity/ltx-weights.py
python -W ignore tools/parity/ltx-dit.py
```

| Script | Crate | What it generates |
|---|---|---|
| `ltx-weights.py` | `ltx-weights` | BF16 base + LoRA, FP8+scale, transformer KeyMap, IC-LoRA layout |
| `ltx-dit.py` | `ltx-dit` | 2-block tiny video `DiT` (random init, seed 42): weights + inputs + reference output; `atol=1e-4` `rtol=1e-3` |
| `ltx-vae-decoder.py` | `ltx-vae-decoder` | Full decode forward pass: weights + latent + noise + pixels |
| `ltx-vae.py` | `ltx-vae` | `VideoEncoder` forward pass: weights + input `[1,3,9,32,32]` + output `[1,4,3,4,4]`; tiny config with `res_x`, `compress_space_res`, `compress_time_res`, `attn`, `compress_all_res` |
| `ltx-burn.py` | `ltx-burn` | End-to-end alpha-gen pipeline: encoder + conditioning + Euler loop (2 steps, seam keyframe) + decoder; records exact noise for Rust comparison |
