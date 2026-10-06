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
