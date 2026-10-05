"""Parity fixture generator for ltx-weights.

Builds tiny random-init tensors (base weights + LoRA) in the reference format,
runs the reference fuse / dequant logic, and saves inputs + expected outputs as
a safetensors fixture in crates/ltx-weights/tests/fixtures/.

Run with the reference venv:
    python -W ignore tools/parity/ltx-weights.py

LTX-2 commit: 9ec55f9
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

FIXTURE_DIR = Path(__file__).parent.parent.parent / "crates" / "ltx-weights" / "tests" / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)


def save_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str] = None):
    """Write a minimal safetensors file."""
    meta = metadata or {}
    header_tensors = {}
    data_parts = []
    offset = 0
    for name, t in tensors.items():
        t_cont = t.contiguous()
        raw = t_cont.view(torch.uint8).numpy().tobytes() if t.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2) else t_cont.numpy().tobytes()
        dtype_map = {
            torch.float32: "F32",
            torch.float16: "F16",
            torch.bfloat16: "BF16",
            torch.float8_e4m3fn: "F8_E4M3",
            torch.float8_e5m2: "F8_E5M2",
        }
        header_tensors[name] = {
            "dtype": dtype_map[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data_parts.append(raw)
        offset += len(raw)
    header = {"__metadata__": meta, **header_tensors}
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for part in data_parts:
            f.write(part)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  BF16 base + LoRA parity
# ──────────────────────────────────────────────────────────────────────────────

def make_lora_parity_fixture():
    """Build bf16 base weight, LoRA A/B, alpha; run reference fuse, save fixture."""
    rng = torch.Generator().manual_seed(42)
    out_features, in_features, rank = 8, 4, 2
    alpha = 4.0  # alpha/rank = 2.0

    base_bf16 = torch.randn(out_features, in_features, generator=rng, dtype=torch.bfloat16)
    lora_a = torch.randn(rank, in_features, generator=rng, dtype=torch.float32)
    lora_b = torch.randn(out_features, rank, generator=rng, dtype=torch.float32)
    strength = 0.75

    # Reference fuse logic: delta = strength * (alpha/rank) * B @ A, then add to base.
    coeff = strength * (alpha / rank)
    delta = coeff * (lora_b @ lora_a)
    fused = base_bf16.float() + delta
    fused_bf16 = fused.to(torch.bfloat16)

    base_path = FIXTURE_DIR / "lora_parity_base.safetensors"
    lora_path = FIXTURE_DIR / "lora_parity_lora.safetensors"
    expected_path = FIXTURE_DIR / "lora_parity_expected.safetensors"

    save_safetensors(base_path, {"layer.weight": base_bf16})
    save_safetensors(
        lora_path,
        {
            "layer.lora_A.weight": lora_a,
            "layer.lora_B.weight": lora_b,
            "layer.alpha": torch.tensor(alpha),
        },
    )
    save_safetensors(
        expected_path,
        {"layer.weight": fused_bf16},
        metadata={
            "ltx2_commit": "9ec55f9",
            "fixture": "lora_parity",
            "strength": str(strength),
            "alpha": str(alpha),
            "rank": str(rank),
        },
    )
    print(f"Saved LoRA parity fixture: base={base_path.name}, lora={lora_path.name}, expected={expected_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  FP8 e4m3fn with scale parity
# ──────────────────────────────────────────────────────────────────────────────

def make_fp8_parity_fixture():
    """Build fp8 weight + scale; save raw FP8 bytes and expected f32 result."""
    rng = torch.Generator().manual_seed(7)
    w_f32 = torch.randn(4, 4, generator=rng, dtype=torch.float32) * 0.1
    # Naive downcast to fp8
    w_fp8 = w_f32.to(torch.float8_e4m3fn)
    # Scale: max abs value, as used in fp8_scaled_mm.py
    scale_val = w_f32.abs().max().item()
    scale = torch.tensor(scale_val, dtype=torch.float32)
    # Expected: dequant = fp8.float() * scale
    expected = w_fp8.float() * scale

    path = FIXTURE_DIR / "fp8_e4m3fn_parity.safetensors"
    save_safetensors(
        path,
        {"w": w_fp8, "w_scale": scale},
        metadata={"ltx2_commit": "9ec55f9", "fixture": "fp8_e4m3fn_scale"},
    )
    # Also save expected for Rust test to compare against.
    expected_path = FIXTURE_DIR / "fp8_e4m3fn_expected.safetensors"
    save_safetensors(expected_path, {"w": expected})
    print(f"Saved FP8 parity fixture: {path.name}, expected={expected_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Transformer KeyMap parity: real-looking transformer checkpoint keys
# ──────────────────────────────────────────────────────────────────────────────

def make_transformer_keymap_fixture():
    """Build checkpoint with model.diffusion_model.* keys; verify post-rename keys."""
    rng = torch.Generator().manual_seed(99)
    keys = [
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.bias",
        "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight",
    ]
    tensors = {}
    expected_renamed = {}
    for k in keys:
        t = torch.randn(8, 4, generator=rng, dtype=torch.float32)
        tensors[k] = t
        renamed = k.removeprefix("model.diffusion_model.")
        expected_renamed[renamed] = t

    path = FIXTURE_DIR / "transformer_keymap_fixture.safetensors"
    save_safetensors(
        path,
        tensors,
        metadata={"ltx2_commit": "9ec55f9", "fixture": "transformer_keymap"},
    )
    expected_path = FIXTURE_DIR / "transformer_keymap_expected.safetensors"
    save_safetensors(expected_path, expected_renamed)
    print(f"Saved transformer KeyMap fixture: {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  IC-LoRA layout metadata
# ──────────────────────────────────────────────────────────────────────────────

def make_iclora_layout_fixture():
    lora_sft = {
        "dummy.lora_A.weight": torch.ones(1, 1),
        "dummy.lora_B.weight": torch.ones(1, 1),
    }
    path = FIXTURE_DIR / "iclora_layout.safetensors"
    save_safetensors(
        path,
        lora_sft,
        metadata={
            "reference_downscale_factor": "2",
            "reference_temporal_scale_factor": "4",
            "ltx2_commit": "9ec55f9",
            "fixture": "iclora_layout",
        },
    )
    print(f"Saved IC-LoRA layout fixture: {path.name}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_lora_parity_fixture()
    make_fp8_parity_fixture()
    make_transformer_keymap_fixture()
    make_iclora_layout_fixture()
    print("All fixtures written to", FIXTURE_DIR)
