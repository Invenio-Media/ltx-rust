"""Parity fixture generator for ltx-vae (video VAE encoder).

Builds the reference VideoEncoder with a tiny random-init config, runs it on a
small input video, and saves weights + input + output as a safetensors fixture
in crates/ltx-vae/tests/fixtures/.

Config exercises all encoder block kinds used by the real LTX-2.5 encoder:
  - res_x          (UNetMidBlock3D / ResnetBlock3D)
  - compress_space_res  (SpaceToDepthDownsample, spatial stride)
  - compress_time_res   (SpaceToDepthDownsample, temporal stride)
  - attn                (AttnBlock3D)
  - compress_all_res    (SpaceToDepthDownsample, all dimensions)

Fixture is saved with module.state_dict() key names so the Rust test can load
it with WeightStore::open(..., &KeyMap::identity()) and a root scope.

Run with the reference venv:
    python -W ignore tools/parity/ltx-vae.py

LTX-2 commit: 9ec55f9
"""

import json
import struct
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import the reference encoder and configurator
# ---------------------------------------------------------------------------
from ltx_core.model.video_vae.video_vae import VideoEncoder
from ltx_core.model.video_vae.model_configurator import VideoEncoderConfigurator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "crates" / "ltx-vae" / "tests" / "fixtures"
FIXTURE_PATH = FIXTURE_DIR / "ltx_vae_encoder_parity.safetensors"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Tiny config (covers all block types used by the real LTX-2.5 encoder)
# ---------------------------------------------------------------------------
# Stored in checkpoint-style nested layout (config.vae.encoder) so the Rust
# VaeEncoderConfig::from_vae_json can parse it directly from the metadata.
VAE_CONFIG = {
    "vae": {
        "_class_name": "CausalDiffusionVAE",
        "encoder": {
            "in_channels": 3,
            "out_channels": 4,
            "dims": 3,
            "patch_size": 2,
            "norm_layer": "pixel_norm",
            "latent_log_var": "uniform",
            "spatial_padding_mode": "zeros",
            "blocks": [
                ["res_x",              {"num_layers": 1}],
                ["compress_space_res", {"multiplier": 2}],
                ["compress_time_res",  {"multiplier": 2}],
                ["attn",               {}],
                ["compress_all_res",   {"multiplier": 2}],
            ],
        },
    }
}

LTX2_COMMIT = "9ec55f9"


# ---------------------------------------------------------------------------
# safetensors writer (no external library required)
# ---------------------------------------------------------------------------
def save_safetensors(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a minimal safetensors file (host byte order = little-endian)."""
    meta = metadata or {}
    header_tensors: dict = {}
    data_parts: list[bytes] = []
    offset = 0
    for name, t in tensors.items():
        t_cont = t.contiguous().cpu()
        # Convert to bytes
        if t.dtype in (torch.bfloat16,):
            raw = t_cont.view(torch.uint8).numpy().tobytes()
        else:
            raw = t_cont.numpy().tobytes()
        dtype_map = {
            torch.float32: "F32",
            torch.float16: "F16",
            torch.bfloat16: "BF16",
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
    print(f"  saved {path.relative_to(REPO_ROOT)}  ({path.stat().st_size // 1024} KiB)")


# ---------------------------------------------------------------------------
# Build the encoder
# ---------------------------------------------------------------------------
def build_encoder() -> VideoEncoder:
    torch.manual_seed(0)
    enc_cfg = VAE_CONFIG["vae"]
    encoder = VideoEncoderConfigurator.from_metadata({"config": {"vae": enc_cfg}})
    encoder.eval()
    # Set per-channel statistics to non-identity values so the test exercises
    # the normalize path with real numbers.
    with torch.no_grad():
        torch.manual_seed(1)
        encoder.per_channel_statistics.get_buffer("std-of-means").copy_(
            torch.abs(torch.randn(4)) + 0.5
        )
        encoder.per_channel_statistics.get_buffer("mean-of-means").copy_(
            torch.randn(4) * 0.1
        )
    return encoder


# ---------------------------------------------------------------------------
# Run encode and collect fixture
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Building encoder …")
    encoder = build_encoder()

    # Input: (B=1, C=3, F=9, H=32, W=32) — 9 = 1 + 8*1, satisfies frame rule
    torch.manual_seed(42)
    video = torch.randn(1, 3, 9, 32, 32) * 0.5  # in (-1, 1) range approx

    print(f"Running encode …")
    with torch.no_grad():
        latent = encoder(video)

    print(f"Input shape:  {tuple(video.shape)}")
    print(f"Output shape: {tuple(latent.shape)}")

    # ── Collect state_dict + input/output ────────────────────────────────
    state = encoder.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    for key, val in state.items():
        tensors[key] = val.float()  # save as f32 for clean parity
    tensors["input"] = video.float()
    tensors["output"] = latent.float()

    # Metadata
    config_json = json.dumps(VAE_CONFIG, separators=(",", ":"))
    metadata = {
        "ltx2_commit": LTX2_COMMIT,
        "config": config_json,
        "input_shape": json.dumps(list(video.shape)),
        "output_shape": json.dumps(list(latent.shape)),
    }

    print(f"Saving fixture …")
    save_safetensors(FIXTURE_PATH, tensors, metadata)

    size_bytes = FIXTURE_PATH.stat().st_size
    assert size_bytes < 2 * 1024 * 1024, f"fixture too large: {size_bytes} bytes"
    print(f"  fixture size: {size_bytes // 1024} KiB  (limit 2048 KiB)  ✓")
    print("Done.")


if __name__ == "__main__":
    main()
