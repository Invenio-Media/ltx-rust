"""Parity fixtures for ltx-vae-decoder O(N·k) NA3D and tiling paths.

Generates TWO safetensors fixtures:

1. ``parity_scale_edge_interior.safetensors``
   Canvas at stage-1 large enough that NA kernel [3,5,5] has both
   clamped-edge positions AND unclamped interior positions on every axis:
     T=5 with kT=3: ti=2 is interior (ws=1, 0 < ws < max_start=2).
     H=7 with kH=5: hi=3 is interior (ws=1, 0 < ws < max_start=2).
     W=7 with kW=5: wi=3 is interior.
   Small spatial upsampling in the config keeps the fixture under 2 MB.

2. ``parity_scale_tiled.safetensors``
   Same tiny config; a 2×2 spatial tile schedule (no temporal split to
   avoid drop_leading_frame complications in the fixture script).  Each
   tile is decoded independently and the Rust ``decode_with_tiling``
   must agree with the blended reference at atol=5e-3 / rtol=5e-2
   (slightly looser than the untiled test because blending introduces
   additional floating-point differences between the independent tile
   decode passes).

Both fixtures stay well under 2 MB using the tiny-channel config below.

Run with:
  /Users/keithmanlove/Documents/Projects/ltx-rust-tools/LTX-2/.venv/bin/python \\
      -W ignore tools/parity/ltx-vae-decoder-scale.py
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

FIXTURE_EDGE_INTERIOR = FIXTURE_DIR / "parity_scale_edge_interior.safetensors"
FIXTURE_TILED        = FIXTURE_DIR / "parity_scale_tiled.safetensors"

# ─── Tiny model config ────────────────────────────────────────────────────────
#
# Upsamples give small spatial growth: up0 does H,W ×2 only; up1-3 are temporal.
# Total spatial: ×2 (from up0).  Total temporal: ×2×2×2 = ×8 (from up1,2,3)
#   minus drops.  With patch_size=1, canvas_H = latent_H × 2.
#
# Channel compatibility: stage_channels[i+1] = stage_channels[i] / factor[i]
#   up0: 32/2=16   stage_channels[1]=16 ✓
#   up1: 16/1=16   stage_channels[2]=16 ✓
#   up2: 16/1=16   stage_channels[3]=16 ✓
#   up3: 16/2=8    stage_channels[4]=8  ✓

SCALE_CONFIG = {
    "in_channels": 4,
    "out_channels": 3,
    "patch_size": 1,
    "head_dim": 16,
    # All channels multiples of head_dim=16 (default_rope_dim_split requires head_dim>=16).
    # Channel compatibility: stage_channels[i+1] = stage_channels[i] / factor[i]
    #   up0: 32/2=16  ✓  up1: 16/1=16  ✓  up2: 16/1=16  ✓  up3: 16/1=16  ✓
    "stage_channels": [32, 16, 16, 16, 16],
    "stage_depths": [1, 1, 1, 1, 1],
    # Stage-1 kernel [3,5,5]: T=5 → interior T positions; H=W=7 → interior spatial.
    "stage_kernels": [[3, 5, 5], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    "upsamples": [
        {"stride": [1, 2, 2], "out_channels_reduction_factor": 2},  # H,W ×2
        {"stride": [2, 1, 1], "out_channels_reduction_factor": 1},  # T ×2
        {"stride": [2, 1, 1], "out_channels_reduction_factor": 1},  # T ×2
        {"stride": [2, 1, 1], "out_channels_reduction_factor": 1},  # T ×2
    ],
    "stage5_kernel": [3, 3, 3],
    "t_emb_dim": 32,
    "default_num_inference_steps": 2,
    "model_output_type": "v",
    "timestep_scale_multiplier": 1.0,
}

# Temporal geometry for SCALE_CONFIG with latent_t:
#   natten_trailing_pad = (stage_kernels[0][0] // 2) * 2 = (3//2)*2 = 2
#   time_scale = up0_t * up1_t * up2_t = 1*2*2 = 4  (first 3 upsamples T strides)
#   pixel_time_scale = time_scale * up3_t = 4*2 = 8
#   cumulative_drop3 = up1 adds 1, up2 adds 1 (then ×1) = 0 + 1*2 = 2? 
#     Rust: up0 T=1 (skip), up1 T=2 (drop=0*2+1=1), up2 T=2 (drop=1*2+1=3). Wait:
#     The strides are up0=[1,2,2], up1=[2,1,1], up2=[2,1,1], up3=[2,1,1].
#     cumulative_temporal_drop(upsamples, 3) iterates first 3:
#       up0 T=1: skip → drop=0
#       up1 T=2: drop=0*2+1=1
#       up2 T=2: drop=1*2+1=3
#     So stage4_time_drop=3, time_scale=1*2*2=4
#   feat_s4_T = latent_t_padded * 4 - 3
#   ctx_T_before_crop = feat_s4_T * 2 - 1  (up3 T=2 with drop_leading)
#   ghost = natten_pad * pixel_time_scale = 2 * 8 = 16
#   pixel_time_drop = cumulative_drop(ups, 4): up3 T=2 → 3*2+1=7
#   pixel_f = latent_t * pixel_time_scale - pixel_time_drop = latent_t*8 - 7


# ─── Safetensors writer ───────────────────────────────────────────────────────

def save_safetensors(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str] | None = None,
) -> None:
    meta = metadata or {}
    header_tensors: dict[str, object] = {}
    data_parts: list[bytes] = []
    offset = 0
    for name, t in tensors.items():
        t_cont = t.contiguous().float()
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
    cfg = SCALE_CONFIG
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
    _set_attention_function(model, EagerSdpaAttention())
    model.eval()
    return model


def _decode_ref(
    model: DiffusionVideoDecoder,
    latent: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Run one full untiled decode via model._decode_one_tile."""
    with torch.no_grad():
        latent_padded = diffusion_tiling.pad_trailing_latent_for_natten_border(
            latent, model._natten_trailing_pad_latent_frames
        )
        feat_s4 = model.forward_stages_1_to_3(latent_padded, drop_leading_frame=True)
        timestep_batched = model.default_inference_timesteps.unsqueeze(0).expand(latent.shape[0], -1)
        pixels = model._decode_one_tile(
            feat_s4,
            noise.clone(),
            is_origin=True,
            timestep=timestep_batched,
            pad_trailing=True,
        )
    return pixels


def _content_crop(pixels: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """Crop pixels to content dimensions matching Rust decode() logic."""
    latent_t = latent.shape[2]
    # pixel_time_scale = up0_t * up1_t * up2_t * up3_t = 1*2*2*2 = 8
    # pixel_time_drop  = cumulative_temporal_drop for 4 upsamples = 7
    pixel_f = max(0, latent_t * 8 - 7)
    # canvas_H = latent_H * up0_H * patch_size = latent_H * 2 * 1 = latent_H * 2
    canvas_h = latent.shape[3] * 2
    canvas_w = latent.shape[4] * 2
    # Rust decode() computes pixel_h = latent_H * 32, but actual canvas_h = latent_H * 2.
    # Since pixel_h >> canvas_h, no spatial crop → keep all canvas pixels.
    f_keep = min(pixels.shape[2], pixel_f)
    h_keep = min(pixels.shape[3], canvas_h)
    w_keep = min(pixels.shape[4], canvas_w)
    return pixels[:, :, :f_keep, :h_keep, :w_keep]


# ─── Fixture 1: edge + interior NA windows ────────────────────────────────────

def gen_edge_interior(model: DiffusionVideoDecoder) -> None:
    """Latent [1, 4, 5, 7, 7] → stage-1 sees T=5, H=7, W=7 with kernel [3,5,5].

    T=5, kT=3: half=1, max_start=2.
      - ti=0,1:    ts=0 (left boundary)
      - ti=2:      ts=1 (interior: 0 < ts < max_start=2)
      - ti=3,4:    ts=2 (right boundary)

    H=7, kH=5: half=2, max_start=2.
      - hi=0,1,2:  hs=0 (left boundary)
      - hi=3:      hs=1 (interior: 0 < hs < max_start=2)
      - hi=4,5,6:  hs=2 (right boundary)

    W=7: same as H.
    """
    torch.manual_seed(7)
    latent = torch.randn(1, SCALE_CONFIG["in_channels"], 5, 7, 7)
    print(f"[edge_interior] latent: {tuple(latent.shape)}")

    with torch.no_grad():
        latent_padded = diffusion_tiling.pad_trailing_latent_for_natten_border(
            latent, model._natten_trailing_pad_latent_frames
        )
        feat_s4 = model.forward_stages_1_to_3(latent_padded, drop_leading_frame=True)
        ctx = model.forward_stage_4(feat_s4.clone(), drop_leading_frame=True, pad_trailing=True)
        _, ctx_t, ctx_h, ctx_w, _ = ctx.shape
        canvas_h = ctx_h * model.patch_size
        canvas_w = ctx_w * model.patch_size
        print(f"[edge_interior] context: T={ctx_t}  H={ctx_h}  W={ctx_w}")

        torch.manual_seed(77)
        noise = torch.randn(1, SCALE_CONFIG["out_channels"], ctx_t, canvas_h, canvas_w)

        timestep_batched = model.default_inference_timesteps.unsqueeze(0).expand(1, -1)
        pixels = model._decode_one_tile(
            feat_s4,
            noise.clone(),
            is_origin=True,
            timestep=timestep_batched,
            pad_trailing=True,
        )
        pixels_cropped = _content_crop(pixels, latent)
        print(f"[edge_interior] pixels_cropped: {tuple(pixels_cropped.shape)}")

    tensors: dict[str, torch.Tensor] = {}
    for key, val in model.state_dict().items():
        tensors[key] = val.float()
    tensors["input_latent"] = latent.float()
    tensors["input_noise"] = noise.float()
    tensors["output_pixels"] = pixels_cropped.float()

    meta = {"config": json.dumps(SCALE_CONFIG)}
    save_safetensors(FIXTURE_EDGE_INTERIOR, tensors, meta)
    sz = FIXTURE_EDGE_INTERIOR.stat().st_size / 1024
    print(f"[edge_interior] saved → {FIXTURE_EDGE_INTERIOR.name}  ({sz:.1f} KB)")
    assert sz < 2048, f"Fixture too large: {sz:.1f} KB > 2048 KB"


# ─── Fixture 2: 2×2 spatial tiling ───────────────────────────────────────────

def gen_tiled(model: DiffusionVideoDecoder) -> None:
    """Latent [1, 4, 2, 8, 8] with 2×2 spatial tile schedule.

    canvas_H = canvas_W = 8*2 = 16 (patch_size=1, up0 ×2).
    Split each spatial axis into 2 non-overlapping tiles:
      - H tiles: [0, 8) and [8, 16)
      - W tiles: [0, 8) and [8, 16)

    Four tiles total; no temporal tiling (single T slice).
    Tiles are non-overlapping so blend masks are all 1.0; the blended
    result should be identical to assembling the four tile outputs by
    concatenation.

    The Rust decode_with_tiling is run with:
      DecodeTileConfig {
          frames: TileDim { tile_size: 0, overlap: 0 },    # no temporal tiling
          height: TileDim { tile_size: 8, overlap: 0 },
          width:  TileDim { tile_size: 8, overlap: 0 },
      }
    """
    torch.manual_seed(13)
    latent = torch.randn(1, SCALE_CONFIG["in_channels"], 2, 8, 8)
    print(f"[tiled] latent: {tuple(latent.shape)}")

    with torch.no_grad():
        latent_padded = diffusion_tiling.pad_trailing_latent_for_natten_border(
            latent, model._natten_trailing_pad_latent_frames
        )
        feat_s4 = model.forward_stages_1_to_3(latent_padded, drop_leading_frame=True)
        full_ctx = model.forward_stage_4(feat_s4.clone(), drop_leading_frame=True, pad_trailing=True)
        _, ctx_t, ctx_h, ctx_w, _ = full_ctx.shape
        canvas_h = ctx_h * model.patch_size
        canvas_w = ctx_w * model.patch_size
        print(f"[tiled] context: T={ctx_t}  H={ctx_h}  W={ctx_w}")
        print(f"[tiled] canvas:  H={canvas_h}  W={canvas_w}")

        torch.manual_seed(99)
        noise_full = torch.randn(1, SCALE_CONFIG["out_channels"], ctx_t, canvas_h, canvas_w)

        # Stage-4 spatial dims.
        s4_t, s4_h, s4_w = feat_s4.shape[1], feat_s4.shape[2], feat_s4.shape[3]
        half_s4_h = s4_h // 2
        half_s4_w = s4_w // 2
        half_ctx_h = ctx_h // 2
        half_ctx_w = ctx_w // 2
        half_canvas_h = canvas_h // 2
        half_canvas_w = canvas_w // 2

        print(f"[tiled] feat_s4 shape: {tuple(feat_s4.shape)}")
        print(f"[tiled] half_s4: H={half_s4_h}  W={half_s4_w}")

        timestep_batched = model.default_inference_timesteps.unsqueeze(0).expand(1, -1)

        # Four tiles: [H_lo,W_lo], [H_lo,W_hi], [H_hi,W_lo], [H_hi,W_hi].
        tile_specs = [
            # (s4_h_slice, s4_w_slice, out_h_slice, out_w_slice)
            (slice(0, half_s4_h), slice(0, half_s4_w),
             slice(0, half_canvas_h), slice(0, half_canvas_w)),
            (slice(0, half_s4_h), slice(half_s4_w, s4_w),
             slice(0, half_canvas_h), slice(half_canvas_w, canvas_w)),
            (slice(half_s4_h, s4_h), slice(0, half_s4_w),
             slice(half_canvas_h, canvas_h), slice(0, half_canvas_w)),
            (slice(half_s4_h, s4_h), slice(half_s4_w, s4_w),
             slice(half_canvas_h, canvas_h), slice(half_canvas_w, canvas_w)),
        ]

        pixels_blended = torch.zeros(1, SCALE_CONFIG["out_channels"], ctx_t, canvas_h, canvas_w)

        for tile_hs4, tile_ws4, tile_hpx, tile_wpx in tile_specs:
            feat_tile = feat_s4[:, :, tile_hs4, tile_ws4, :]
            tile_ctx = model.forward_stage_4(feat_tile.clone(), drop_leading_frame=True, pad_trailing=True)
            _, t_t, t_h, t_w, _ = tile_ctx.shape
            t_canvas_h = t_h * model.patch_size
            t_canvas_w = t_w * model.patch_size

            noise_tile = noise_full[:, :, :t_t, tile_hpx, tile_wpx]
            tile_pixels = model._decode_one_tile(
                feat_tile,
                noise_tile.clone(),
                is_origin=True,
                timestep=timestep_batched,
                pad_trailing=True,
            )
            print(f"[tiled] tile {tile_hs4},{tile_ws4}: pixels={tuple(tile_pixels.shape)}")
            pixels_blended[:, :, :min(t_t, ctx_t), tile_hpx, tile_wpx] = \
                tile_pixels[:, :, :min(t_t, ctx_t)]

        pixels_tiled_cropped = _content_crop(pixels_blended, latent)
        print(f"[tiled] blended+cropped: {tuple(pixels_tiled_cropped.shape)}")

    tensors: dict[str, torch.Tensor] = {}
    for key, val in model.state_dict().items():
        tensors[key] = val.float()
    tensors["input_latent"] = latent.float()
    tensors["input_noise"] = noise_full.float()
    tensors["output_pixels_tiled"] = pixels_tiled_cropped.float()
    # Store tile dims so the Rust test can reproduce the schedule.
    tensors["tile_half_ctx_h"] = torch.tensor([half_canvas_h], dtype=torch.float32)
    tensors["tile_half_ctx_w"] = torch.tensor([half_canvas_w], dtype=torch.float32)

    meta = {"config": json.dumps(SCALE_CONFIG)}
    save_safetensors(FIXTURE_TILED, tensors, meta)
    sz = FIXTURE_TILED.stat().st_size / 1024
    print(f"[tiled] saved → {FIXTURE_TILED.name}  ({sz:.1f} KB)")
    assert sz < 2048, f"Fixture too large: {sz:.1f} KB > 2048 KB"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Building model (seed=42)…")
    model = build_model()
    print(f"  state_dict keys: {len(model.state_dict())}")

    print("\n=== Fixture 1: edge + interior NA windows ===")
    gen_edge_interior(model)

    print("\n=== Fixture 2: 2×2 spatial tiling ===")
    gen_tiled(model)

    print("\nDone.")


if __name__ == "__main__":
    main()
