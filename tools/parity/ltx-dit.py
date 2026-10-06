#!/usr/bin/env python3
"""Parity fixture generator for ltx-dit.

Builds a tiny LTXModel (VideoOnly, random-init, fixed seed) with every
feature that the real LTX-2.5 22B config uses: split `RoPE`, `AdaLN`-single
timestep embedding, cross-attention to text context, per-token timesteps,
and a no-op perturbation config (as alpha_gen passes in production).

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

# ── Tiny config ───────────────────────────────────────────────────────────────
# Every flag mirrors the real LTX-2.5 22B default; only dims are shrunk.
# inner_dim == cross_attention_dim (22B invariant: caption proj is outside the
# transformer, connector already maps to inner_dim).
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = 8
INNER_DIM = NUM_HEADS * HEAD_DIM   # 32
CROSS_ATTN_DIM = INNER_DIM         # 32 (equals inner_dim per 22B convention)
IN_CHANNELS = 8

NORM_EPS = 1e-6
THETA = 10000.0
MAX_POS = [4, 8, 8]
TS_SCALE = 1000
FF_BIAS = False          # 22B default
APPLY_GATED = False      # 22B default
CROSS_ATTN_ADALN = False # 22B default

# Input dimensions
BATCH = 1
FRAMES = 3   # latent frames
HEIGHT = 2   # latent spatial height
WIDTH = 2    # latent spatial width
N_TOKENS = FRAMES * HEIGHT * WIDTH  # 12 patchified tokens
CTX_LEN = 6  # text context sequence length


# ── safetensors serialiser (no external dep on safetensors-python) ────────────

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


# ── Main fixture builder ──────────────────────────────────────────────────────

def make_dit_parity_fixture() -> None:
    """Build the DiT parity fixture (random-init tiny model, fixed seed)."""
    torch.manual_seed(42)

    # Force CPU math SDPA so the fixture runs on any machine without MPS/CUDA.
    cpu_ops = AttentionOps(
        attention_function=AttentionFunction.SDPA_MATH.to_callable(),
        masked_attention_function=MaskedAttentionFunction.SDPA_MATH.to_callable(),
    )

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
        ff_bias=FF_BIAS,
        apply_gated_attention=APPLY_GATED,
        # caption_proj_before_connector=True in 22B -> no caption_projection
        caption_projection=None,
        cross_attention_adaln=CROSS_ATTN_ADALN,
        # use_prompt_adaln_single is True by default but inert when
        # cross_attention_adaln=False (matches 22B checkpoint).
        use_prompt_adaln_single=True,
        use_keyframes_abs_pos_embedding=False,
        attention_ops=cpu_ops,
    ).eval()

    # ── Build random-init inputs ──────────────────────────────────────────────
    # latent: patchified latent BEFORE patchify_proj (B, T, in_channels)
    # TransformerArgsPreprocessor.prepare() applies patchify_proj internally;
    # Rust VideoTransformer::forward also applies it first.
    latent = torch.randn(BATCH, N_TOKENS, IN_CHANNELS)

    # Per-token timesteps (B, T) in [0, 1].
    # In alpha_gen: timesteps_from_mask(denoise_mask, sigma) = denoise_mask * sigma.
    # IC-LoRA clean reference tokens would have timestep=0.0.
    timesteps = torch.full((BATCH, N_TOKENS), 0.8)

    # sigma (B,): per-batch scalar used by prompt_adaln (inert here because
    # cross_attention_adaln=False, so use_prompt_adaln_single is not wired).
    sigma = torch.tensor([0.8])

    # Position bounds (B, 3, T, 2): [start, end) per patch per (t, h, w) dim.
    # In alpha_gen, VideoLatentTools builds these from the latent shape + fps.
    # For the fixture we use integer sequential positions (fps=1, spacing=1).
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
    ).unsqueeze(0)  # (1, 3, T, 2)

    # Text context (B, S, CROSS_ATTN_DIM).
    # Matches EmbeddingsProcessorOutput.video_encoding: connector already
    # applied, no caption projection inside the transformer.
    # alpha_gen passes context_mask=None (no cross-attention mask).
    context = torch.randn(BATCH, CTX_LEN, CROSS_ATTN_DIM)

    # ── Forward pass (matches alpha_gen denoising step) ───────────────────────
    video_modality = Modality(
        latent=latent,
        sigma=sigma,
        timesteps=timesteps,
        positions=positions,
        context=context,
        enabled=True,
        context_mask=None,   # alpha_gen does not pass a context mask
        attention_mask=None, # no IC-LoRA self-attention mask in this fixture
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

    # ── Serialize ─────────────────────────────────────────────────────────────
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
        # 22B flags (all at their default values)
        "apply_gated_attention": APPLY_GATED,
        "cross_attention_adaln": CROSS_ATTN_ADALN,
        "use_prompt_adaln_single": True,
        "ff_bias": FF_BIAS,
        "use_keyframes_abs_pos_embedding": False,
        "caption_proj_before_connector": True,
    }

    tensors: dict[str, torch.Tensor] = {}

    # All model weights by state_dict() names (KeyMap::identity on the fixture)
    for key, val in model.state_dict().items():
        tensors[key] = val.float()

    # Inputs (raw, before any linear projection)
    tensors["input.latent"] = latent.float()
    tensors["input.timesteps"] = timesteps.float()
    tensors["input.positions"] = positions.float()
    tensors["input.context"] = context.float()

    # Reference output
    tensors["output"] = output_v.detach().float()

    metadata = {
        "config": json.dumps(config_dict),
        "ltx2_commit": LTX2_COMMIT,
        "torch_seed": "42",
        "context_mask": "none",
        "attention_mask": "none",
    }

    path = FIXTURE_DIR / "dit_parity.safetensors"
    save_safetensors(path, tensors, metadata)

    size_kb = path.stat().st_size / 1024
    assert size_kb < 2048, f"fixture {size_kb:.0f} KB exceeds 2 MB limit"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved DiT parity fixture: {path.name}  ({size_kb:.0f} KB)")
    print(f"  model params: {n_params:,}")
    print(f"  output shape: {output_v.shape}")
    print(f"  output range: [{output_v.min():.4f}, {output_v.max():.4f}]")


if __name__ == "__main__":
    make_dit_parity_fixture()
