# Parity fixtures

Each script in this directory generates fixture files for a crate's parity tests.

Run with the reference venv (`LTX-2/.venv/bin/python`):

```sh
python -W ignore tools/parity/ltx-weights.py
```

| Script | Crate | What it generates |
|---|---|---|
| `ltx-weights.py` | `ltx-weights` | BF16 base + LoRA, FP8+scale, transformer KeyMap, IC-LoRA layout |
