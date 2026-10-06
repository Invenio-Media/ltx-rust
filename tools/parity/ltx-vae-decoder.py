"""Parity fixture generator for ltx-vae-decoder.

Builds a tiny DiffusionVideoDecoder with randomized weights (fixed seed),
runs one full decode with fixed latent and noise, then saves a single
safetensors file containing:
  - all state_dict weights (for Rust to load via WeightStore + KeyMap::identity)
  - input_latent   [1, 8, 1, 4, 4]
  - input_noise    [1, 3, ctx_t, canvas_h, canvas_w]
  - output_pixels  [1, 3, ctx_t, canvas_h, canvas_w]  (cropped to content)

Config JSON is written to safetensors __metadata__["config"].

Run with:
  /Users/keithmanlove/Documents/Projects/ltx-rust-tools/LTX-2/.venv/bin/python \\
      -W ignore tools/parity/ltx-vae-decoder.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import sys

import torch

sys.path.insert(0, "/Users/keithmanlove/Documents/Projects/ltx-rust-tools/LTX-2/packages/ltx-core/src")

from ltx_core.model.video_vae.diffusion_video_decoder import DiffusionVideoDecoder
from ltx_core.model.video_vae import diffusion_tiling
from ltx_core.model.video_vae.transformer import EagerSdpaAttention
from ltx_core.model.video_vae.transformer.apply import _set_attention_function

# ─── Paths ────────────────────────────────────────────────────────────────────

FIXTURE_DIR = (
    Path(__file__).parent.parent.parent
    / "crates"
    / "ltx-vae-decoder"
    / "tests"
    / "fixtures"
)
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
FIXTURE_PATH = FIXTURE_DIR / "parity.safetensors"

# ─── Tiny model config ────────────────────────────────────────────────────────

# All stage_channels are multiples of head_dim=16.
# upsamples length = len(stage_channels) - 1 = 4
TINY_CONFIG = {
    "in_channels": 8,
    "out_channels": 3,
    "patch_size": 2,
    "head_dim": 16,
    "stage_channels": [64, 32, 32, 32, 16],
    "stage_depths": [1, 1, 1, 1, 2],
    "stage_kernels": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    "upsamples": [
        {"stride": [1, 2, 2], "out_channels_reduction_factor": 2},
        {"stride": [2, 1, 1], "out_channels_reduction_factor": 1},
        {"stride": [2, 2, 2], "out_channels_reduction_factor": 1},
        {"stride": [2, 2, 2], "out_channels_reduction_factor": 2},
    ],
    "stage5_kernel": [3, 3, 3],
    "t_emb_dim": 32,
    "default_num_inference_steps": 2,
    "model_output_type": "v",
    "timestep_scale_multiplier": 1.0,
}


# ─── Safetensors writer ───────────────────────────────────────────────────────

_DTYPE_MAP = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
}


def save_safetensors(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a safetensors file with optional __metadata__."""
    meta = metadata or {}
    header_tensors: dict[str, object] = {}
    data_parts: list[bytes] = []
    offset = 0
    for name, t in tensors.items():
        t_cont = t.contiguous().float()  # save everything as F32
        raw = t_cont.numpy().tobytes()
        header_tensors[name] = {
            "dtype": "F32",
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


# ─── Build model ──────────────────────────────────────────────────────────────

def build_model() -> DiffusionVideoDecoder:
    cfg = TINY_CONFIG
    torch.manual_seed(42)
    model = DiffusionVideoDecoder(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        patch_size=cfg["patch_size"],
        head_dim=cfg["head_dim"],
        stage_channels=tuple(cfg["stage_channels"]),
        stage_depths=tuple(cfg["stage_depths"]),
        stage_kernels=tuple(tuple(k) for k in cfg["stage_kernels"]),  # type: ignore[arg-type]
        upsamples=tuple(
            (tuple(u["stride"]), u["out_channels_reduction_factor"])
            for u in cfg["upsamples"]
        ),
        stage5_kernel=tuple(cfg["stage5_kernel"]),  # type: ignore[arg-type]
        t_emb_dim=cfg["t_emb_dim"],
        default_num_inference_steps=cfg["default_num_inference_steps"],
        model_output_type=cfg["model_output_type"],
        timestep_scale_multiplier=cfg["timestep_scale_multiplier"],
    )
    # Use the eager SDPA backend (no NATTEN required) — equivalent to Rust na3d.
    _set_attention_function(model, EagerSdpaAttention())
    model.eval()
    return model


# ─── Generate fixture ─────────────────────────────────────────────────────────

def main() -> None:
    print("Building model…")
    model = build_model()

    # Fixed latent: [1, C_in, 1, H, W] — H,W >= 3 for kernel (3,3,3).
    torch.manual_seed(7)
    latent = torch.randn(1, TINY_CONFIG["in_channels"], 1, 4, 4)
    print(f"latent shape: {tuple(latent.shape)}")

    with torch.no_grad():
        # ── Stage 1-3 ──────────────────────────────────────────────────────
        latent_padded = diffusion_tiling.pad_trailing_latent_for_natten_border(
            latent, model._natten_trailing_pad_latent_frames
        )
        print(f"latent_padded shape: {tuple(latent_padded.shape)}")
        feat_s4 = model.forward_stages_1_to_3(latent_padded, drop_leading_frame=True)
        print(f"feat_s4 shape: {tuple(feat_s4.shape)}")

        # ── Determine noise shape from context ─────────────────────────────
        context_for_shape = model.forward_stage_4(
            feat_s4.clone(), drop_leading_frame=True, pad_trailing=True
        )
        _, ctx_t, ctx_h, ctx_w, _ = context_for_shape.shape
        canvas_h = ctx_h * model.patch_size
        canvas_w = ctx_w * model.patch_size
        print(f"context shape: {tuple(context_for_shape.shape)}")
        print(f"canvas: T={ctx_t}  H={canvas_h}  W={canvas_w}")

        # ── Fixed noise ────────────────────────────────────────────────────
        torch.manual_seed(77)
        noise = torch.randn(1, TINY_CONFIG["out_channels"], ctx_t, canvas_h, canvas_w)
        print(f"noise shape: {tuple(noise.shape)}")

        # ── Full decode via _decode_one_tile ───────────────────────────────
        timestep = model.default_inference_timesteps  # shape [n_steps]
        timestep_batched = timestep.unsqueeze(0).expand(1, -1)  # [1, n_steps]
        pixels = model._decode_one_tile(
            feat_s4,
            noise.clone(),
            is_origin=True,
            timestep=timestep_batched,
            pad_trailing=True,
        )
        print(f"pixels shape: {tuple(pixels.shape)}")
        print(
            f"pixels stats: min={pixels.min():.4f}  max={pixels.max():.4f}"
            f"  mean={pixels.mean():.4f}"
        )

        # ── Crop to match Rust decode() content crop ───────────────────────
        # Rust decode() crops pixels to [pixel_f, pixel_h, pixel_w] where:
        #   pixel_time_scale = time_scale * up4_stride_t
        #   time_scale = prod(upsamples[0:3].stride_t) = 1*2*2 = 4
        #   up4_stride_t = 2  → pixel_time_scale = 8
        #   pixel_time_drop = cumulative_temporal_drop = 7
        #   pixel_f = max(0, latent_t * pixel_time_scale - pixel_time_drop)
        latent_t = latent.shape[2]
        pixel_time_scale = 8   # matches Rust: time_scale=4, up4_stride[0]=2
        pixel_time_drop = 7    # cumulative_temporal_drop(upsamples, 4)
        pixel_f = max(0, latent_t * pixel_time_scale - pixel_time_drop)
        pixel_h = latent.shape[3] * 32  # h_l * 32 (stage strides: 2*2*2*2=16? no)
        pixel_w = latent.shape[4] * 32
        # Rust spatial crop: h_l * 32, w_l * 32 (because canvas_h=ctx_h*patch=32*2=64
        # and h_l=4, 4*32=128 >= 64, so h_keep = min(64, 128) = 64 -- no crop)
        f_out, h_out, w_out = pixels.shape[2], pixels.shape[3], pixels.shape[4]
        f_keep = min(f_out, max(0, pixel_f))
        h_keep = min(h_out, pixel_h)
        w_keep = min(w_out, pixel_w)
        pixels_cropped = pixels[:, :, :f_keep, :h_keep, :w_keep]
        print(f"pixels_cropped shape: {tuple(pixels_cropped.shape)}")

    # ── Save fixture ───────────────────────────────────────────────────────
    print("Saving fixture…")
    tensors: dict[str, torch.Tensor] = {}

    # State dict weights (flat keys; Rust loads via KeyMap::identity()).
    sd = model.state_dict()
    for key, val in sd.items():
        tensors[key] = val.float()

    # I/O tensors.
    tensors["input_latent"] = latent.float()
    tensors["input_noise"] = noise.float()
    tensors["output_pixels"] = pixels_cropped.float()

    # Metadata: config JSON.
    meta = {"config": json.dumps(TINY_CONFIG)}
    save_safetensors(FIXTURE_PATH, tensors, meta)
    print(f"Saved {len(tensors)} tensors to {FIXTURE_PATH}")
    print(f"  state_dict keys: {len(sd)}")
    print(f"  fixture file: {FIXTURE_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
