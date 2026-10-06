#!/usr/bin/env python3
"""Parity fixture generator for ltx-dit.

Generates two fixtures:

1. ``dit_parity.safetensors`` -- 22B-default flags (all off): tests the
   main forward path.
2. ``dit_parity_gated_adaln.safetensors`` -- all non-default flags on:
   ``apply_gated_attention=True``, ``cross_attention_adaln=True``,
   ``use_prompt_adaln_single=False``, ``ff_bias=True``.  Tests the gated
   attention and cross-attention `AdaLN` code paths that the 22B model may
   use (``apply_gated_attention`` / ``cross_attention_adaln`` are read from
   checkpoint metadata in ``model_configurator.py:67,70``).

Context tensor matches what alpha_gen sends:
  ``ctx_p.video_encoding`` from the embeddings processor -- connector already
  applied, no caption projection inside the transformer -- passed directly to
  the denoiser as ``v_context`` with ``context_mask=None``.

Usage:
    python -W ignore tools/parity/ltx-dit.py
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.attention import (
    AttentionFunction,
    AttentionOps,
    MaskedAttentionFunction,
)
from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType

FIXTURE_DIR = (
    Path(__file__).parent.parent.parent / "crates" / "ltx-dit" / "tests" / "fixtures"
)
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
LTX2_COMMIT = "9ec55f9"

# ── Shared tiny dims ───────────────────────────────────────────────────────────
# inner_dim == cross_attention_dim (22B invariant: caption proj is outside the
# transformer, connector already maps to inner_dim).
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = 8
INNER_DIM = NUM_HEADS * HEAD_DIM   # 32
CROSS_ATTN_DIM = INNER_DIM         # 32
IN_CHANNELS = 8
NORM_EPS = 1e-6
THETA = 10000.0
MAX_POS = [4, 8, 8]
TS_SCALE = 1000

# Input dimensions
BATCH = 1
FRAMES = 3
HEIGHT = 2
WIDTH = 2
N_TOKENS = FRAMES * HEIGHT * WIDTH  # 12
CTX_LEN = 6


# ── safetensors serialiser ─────────────────────────────────────────────────────

def save_safetensors(
    path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]
) -> None:
    """Write a safetensors file (float32 only) with __metadata__."""
    tensor_offset = 0
    tensor_entries: dict = {}
    data_parts: list[bytes] = []
    for name, t in tensors.items():
        raw = t.contiguous().float().numpy().astype("float32").tobytes()
        tensor_entries[name] = {
            "dtype": "F32",
            "shape": list(t.shape),
            "data_offsets": [tensor_offset, tensor_offset + len(raw)],
        }
        tensor_offset += len(raw)
        data_parts.append(raw)
    combined: dict = {"__metadata__": metadata}
    combined.update(tensor_entries)
    header_bytes = json.dumps(combined, separators=(",", ":")).encode()
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes = header_bytes + b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for part in data_parts:
            f.write(part)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _cpu_ops() -> AttentionOps:
    """Force CPU math SDPA so fixtures run on any machine."""
    return AttentionOps(
        attention_function=AttentionFunction.SDPA_MATH.to_callable(),
        masked_attention_function=MaskedAttentionFunction.SDPA_MATH.to_callable(),
    )


def _build_inputs(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (latent, timesteps, sigma, positions, context) with a fixed seed."""
    torch.manual_seed(seed)
    latent = torch.randn(BATCH, N_TOKENS, IN_CHANNELS)
    timesteps = torch.full((BATCH, N_TOKENS), 0.8)
    sigma = torch.tensor([0.8])

    t_idx = torch.arange(N_TOKENS) // (HEIGHT * WIDTH)
    h_idx = (torch.arange(N_TOKENS) // WIDTH) % HEIGHT
    w_idx = torch.arange(N_TOKENS) % WIDTH
    positions = torch.stack(
        [
            torch.stack([t_idx.float(), (t_idx + 1).float()], dim=-1),
            torch.stack([h_idx.float(), (h_idx + 1).float()], dim=-1),
            torch.stack([w_idx.float(), (w_idx + 1).float()], dim=-1),
        ],
        dim=0,
    ).unsqueeze(0)
    context = torch.randn(BATCH, CTX_LEN, CROSS_ATTN_DIM)
    return latent, timesteps, sigma, positions, context


def _run_model(
    model: LTXModel,
    latent: torch.Tensor,
    timesteps: torch.Tensor,
    sigma: torch.Tensor,
    positions: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """Run one forward pass and return the video output."""
    video_modality = Modality(
        latent=latent,
        sigma=sigma,
        timesteps=timesteps,
        positions=positions,
        context=context,
        enabled=True,
        context_mask=None,
        attention_mask=None,
        keyframes_mask=None,
    )
    with torch.no_grad():
        perturbations = BatchedPerturbationConfig.empty(
            BATCH, NUM_LAYERS, latent.device, latent.dtype
        )
        output_v, _ = model(video=video_modality, audio=None, perturbations=perturbations)
    assert output_v is not None
    assert output_v.shape == (BATCH, N_TOKENS, IN_CHANNELS), (
        f"unexpected output shape {output_v.shape}"
    )
    assert not output_v.isnan().any(), "NaN in model output"
    return output_v


def _save_fixture(
    path: Path,
    model: LTXModel,
    latent: torch.Tensor,
    timesteps: torch.Tensor,
    positions: torch.Tensor,
    context: torch.Tensor,
    output_v: torch.Tensor,
    config_dict: dict,
    extra_metadata: dict[str, str] | None = None,
) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for key, val in model.state_dict().items():
        tensors[key] = val.float()
    tensors["input.latent"] = latent.float()
    tensors["input.timesteps"] = timesteps.float()
    tensors["input.positions"] = positions.float()
    tensors["input.context"] = context.float()
    tensors["output"] = output_v.detach().float()

    metadata: dict[str, str] = {
        "config": json.dumps(config_dict),
        "ltx2_commit": LTX2_COMMIT,
        "context_mask": "none",
        "attention_mask": "none",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    save_safetensors(path, tensors, metadata)

    size_kb = path.stat().st_size / 1024
    assert size_kb < 2048, f"fixture {size_kb:.0f} KB exceeds 2 MB limit"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved: {path.name}  ({size_kb:.0f} KB, {n_params:,} params)")
    print(f"  output shape {output_v.shape}  range [{output_v.min():.4f}, {output_v.max():.4f}]")


# ── Fixture 1: 22B-default flags ───────────────────────────────────────────────

def make_dit_parity_fixture() -> None:
    """Build the all-defaults fixture (random-init tiny model, seed 42).

    Flags: ``apply_gated_attention=False``, ``cross_attention_adaln=False``,
    ``use_prompt_adaln_single=True`` (inert when cross_attention_adaln=False),
    ``ff_bias=False``.  These match the LTX-2.5 22B production defaults.
    """
    torch.manual_seed(42)
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS,
        num_layers=NUM_LAYERS,
        cross_attention_dim=CROSS_ATTN_DIM,
        norm_eps=NORM_EPS,
        positional_embedding_theta=THETA,
        positional_embedding_max_pos=list(MAX_POS),
        timestep_scale_multiplier=TS_SCALE,
        rope_type=LTXRopeType.SPLIT,
        ff_bias=False,
        apply_gated_attention=False,
        caption_projection=None,
        cross_attention_adaln=False,
        use_prompt_adaln_single=True,
        use_keyframes_abs_pos_embedding=False,
        attention_ops=_cpu_ops(),
    ).eval()

    latent, timesteps, sigma, positions, context = _build_inputs(seed=42)
    output_v = _run_model(model, latent, timesteps, sigma, positions, context)

    config_dict = {
        "num_attention_heads": NUM_HEADS,
        "attention_head_dim": HEAD_DIM,
        "in_channels": IN_CHANNELS,
        "out_channels": IN_CHANNELS,
        "num_layers": NUM_LAYERS,
        "cross_attention_dim": CROSS_ATTN_DIM,
        "norm_eps": NORM_EPS,
        "positional_embedding_theta": THETA,
        "positional_embedding_max_pos": list(MAX_POS),
        "timestep_scale_multiplier": TS_SCALE,
        "use_middle_indices_grid": True,
        "rope_type": "split",
        "apply_gated_attention": False,
        "cross_attention_adaln": False,
        "use_prompt_adaln_single": True,
        "ff_bias": False,
        "use_keyframes_abs_pos_embedding": False,
        "caption_proj_before_connector": True,
    }

    _save_fixture(
        FIXTURE_DIR / "dit_parity.safetensors",
        model, latent, timesteps, positions, context, output_v,
        config_dict,
        extra_metadata={"torch_seed": "42"},
    )


# ── Fixture 2: non-default flags (gated + cross-attention AdaLN) ──────────────

def make_dit_parity_fixture_gated_adaln() -> None:
    """Build the all-non-default-flags fixture (seed 99).

    Flags covered (all read from checkpoint metadata by the reference
    ``LTXVideoOnlyModelConfigurator.from_metadata``):

    - ``apply_gated_attention=True``  (key ``apply_gated_attention``, line 129)
    - ``cross_attention_adaln=True``  (key ``cross_attention_adaln``, line 131)
    - ``use_prompt_adaln_single=False``  (key ``use_prompt_adaln_single``, line 135;
      the ``True`` + ``cross_attention_adaln`` path uses a prompt `AdaLN` MLP that
      is NOT wired in the Rust port; ``False`` = K/V-cacheable, IS wired)
    - ``ff_bias=True``  (key ``ff_bias``, line 137; pre-2.5 default)

    ``BasicAVTransformerBlock`` initialises ``scale_shift_table`` and
    ``prompt_scale_shift_table`` with ``torch.empty`` (uninitialized memory).
    After the model is built, parameters are re-seeded:

    * Linear weights, RmsNorm gammas, and gate-logits weights: ``N(0, 0.02)``
      (small, keeps activations in a reasonable range).
    * ``scale_shift_table`` and ``prompt_scale_shift_table``: ``N(0, 0.5)``
      (larger, so the AdaLN shift/scale/gate terms dominate and any wrong
      index ordering — swapped shift/scale, wrong ``[6:9]`` slice, etc. —
      produces a clearly visible numeric difference in the parity test).
    """
    torch.manual_seed(99)
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS,
        num_layers=NUM_LAYERS,
        cross_attention_dim=CROSS_ATTN_DIM,
        norm_eps=NORM_EPS,
        positional_embedding_theta=THETA,
        positional_embedding_max_pos=list(MAX_POS),
        timestep_scale_multiplier=TS_SCALE,
        rope_type=LTXRopeType.SPLIT,
        ff_bias=True,
        apply_gated_attention=True,
        caption_projection=None,
        cross_attention_adaln=True,
        use_prompt_adaln_single=False,
        use_keyframes_abs_pos_embedding=False,
        attention_ops=_cpu_ops(),
    ).eval()

    # Re-seed and reset all parameters to prevent NaN from torch.empty.
    # scale_shift_table / prompt_scale_shift_table use larger std so wrong
    # coefficient ordering (shift vs scale, wrong [6:9] slice) produces
    # a clearly visible parity failure.
    torch.manual_seed(99)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "scale_shift_table" in name:
                torch.nn.init.normal_(param, std=0.5)
            else:
                torch.nn.init.normal_(param, std=0.02)

    latent, timesteps, sigma, positions, context = _build_inputs(seed=99)
    output_v = _run_model(model, latent, timesteps, sigma, positions, context)

    config_dict = {
        "num_attention_heads": NUM_HEADS,
        "attention_head_dim": HEAD_DIM,
        "in_channels": IN_CHANNELS,
        "out_channels": IN_CHANNELS,
        "num_layers": NUM_LAYERS,
        "cross_attention_dim": CROSS_ATTN_DIM,
        "norm_eps": NORM_EPS,
        "positional_embedding_theta": THETA,
        "positional_embedding_max_pos": list(MAX_POS),
        "timestep_scale_multiplier": TS_SCALE,
        "use_middle_indices_grid": True,
        "rope_type": "split",
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
        "use_prompt_adaln_single": False,
        "ff_bias": True,
        "use_keyframes_abs_pos_embedding": False,
        "caption_proj_before_connector": True,
    }

    _save_fixture(
        FIXTURE_DIR / "dit_parity_gated_adaln.safetensors",
        model, latent, timesteps, positions, context, output_v,
        config_dict,
        extra_metadata={
            "torch_seed": "99",
            "flags": "apply_gated_attention=True,cross_attention_adaln=True,"
                     "use_prompt_adaln_single=False,ff_bias=True",
        },
    )


if __name__ == "__main__":
    make_dit_parity_fixture()
    make_dit_parity_fixture_gated_adaln()
