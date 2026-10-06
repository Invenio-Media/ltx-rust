"""Create production-format checkpoint files from the parity fixture for smoke testing.

Run with:
    python tools/make_smoke_checkpoints.py

Outputs to /tmp/ltx-smoke/:
  - transformer.safetensors  (model.diffusion_model.* prefix + config metadata)
  - vae.safetensors          (encoder.* / decoder.* / per_channel_statistics.* + config)
  - lora.safetensors         (bare patchify_proj.* + IC-LoRA metadata)
  - prompt_context.safetensors (positive/negative.* + format metadata)
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
FIXTURE = REPO_ROOT / "crates" / "ltx-burn" / "tests" / "fixtures" / "parity.safetensors"
OUT_DIR = Path("/tmp/ltx-smoke")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_safetensors(
    path: Path,
    tensors: dict[str, Any],
    metadata: dict[str, str] | None = None,
) -> None:
    import torch

    meta: dict[str, str] = metadata or {}
    hdr: dict[str, Any] = {}
    data_parts: list[bytes] = []
    offset = 0
    for name, t in tensors.items():
        t_cont = t.detach().contiguous().cpu()
        if t.dtype == torch.bfloat16:
            raw = t_cont.view(torch.uint8).numpy().tobytes()
            dtype_str = "BF16"
        elif t.dtype == torch.float16:
            raw = t_cont.numpy().tobytes()
            dtype_str = "F16"
        else:
            raw = t_cont.float().numpy().tobytes()
            dtype_str = "F32"
        hdr[name] = {
            "dtype": dtype_str,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data_parts.append(raw)
        offset += len(raw)
    header = {"__metadata__": meta, **hdr}
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for part in data_parts:
            f.write(part)
    kb = path.stat().st_size // 1024
    print(f"  {path.name}  ({kb} KiB)")


def main() -> None:
    import torch
    from safetensors import safe_open

    print("Loading parity fixture …")
    raw_tensors: dict[str, Any] = {}
    meta: dict[str, str] = {}
    with safe_open(str(FIXTURE), framework="pt") as f:
        meta = dict(f.metadata())
        for k in f.keys():
            raw_tensors[k] = f.get_tensor(k)

    enc_cfg = json.loads(meta["encoder_config"])
    dec_cfg = json.loads(meta["decoder_config"])
    tfm_cfg = json.loads(meta["transformer_config"])

    # ── 1. Transformer: model.diffusion_model.* prefix ──────────────────────
    tfm_tensors: dict[str, Any] = {}
    for k, v in raw_tensors.items():
        if k.startswith("tfm."):
            rest = k[4:]
            tfm_tensors[f"model.diffusion_model.{rest}"] = v

    save_safetensors(
        OUT_DIR / "transformer.safetensors",
        tfm_tensors,
        {"config": json.dumps(tfm_cfg, separators=(",", ":"))},
    )

    # ── 2. VAE: encoder.* / per_channel_statistics.* / decoder.* ────────────
    # KeyMap::video_encoder() accepts encoder.* and per_channel_statistics.*
    # KeyMap::video_decoder() accepts decoder.* and per_channel_statistics.*
    # Fixture already has post-transform keys (t_embedder renamed, QKV split,
    # gates folded) — prefix with decoder.* so the KeyMap strips it, then the
    # further transforms are no-ops.
    vae_tensors: dict[str, Any] = {}
    for k, v in raw_tensors.items():
        if k.startswith("enc."):
            rest = k[4:]
            if rest.startswith("per_channel_statistics."):
                # Encoder per-channel stats: no extra prefix for encoder keymap
                vae_tensors[rest] = v
            else:
                vae_tensors[f"encoder.{rest}"] = v
        elif k.startswith("dec."):
            rest = k[4:]
            if rest.startswith("per_channel_statistics."):
                # Decoder also needs per_channel_statistics at root (same tensor)
                vae_tensors[rest] = v
            else:
                vae_tensors[f"decoder.{rest}"] = v

    # Combined vae config: KeyMap reads __metadata__["config"] → vae sub-key
    vae_config = {"vae": {**enc_cfg, "decoder": dec_cfg}}
    save_safetensors(
        OUT_DIR / "vae.safetensors",
        vae_tensors,
        {"config": json.dumps(vae_config, separators=(",", ":"))},
    )

    # ── 3. LoRA: bare keys (WeightStore matches against stripped tfm keys) ───
    # After KeyMap::transformer() strips "model.diffusion_model.", the base key
    # is e.g. "patchify_proj.weight".  merge_lora looks for "patchify_proj.lora_A.weight"
    # in the LoRA file (no prefix needed).
    lora_tensors: dict[str, Any] = {}
    for k, v in raw_tensors.items():
        if k.startswith("lora."):
            rest = k[5:]
            lora_tensors[rest] = v

    ds = meta.get("lora_reference_downscale_factor", "2")
    ts = meta.get("lora_reference_temporal_scale_factor", "2")
    save_safetensors(
        OUT_DIR / "lora.safetensors",
        lora_tensors,
        {
            "reference_downscale_factor": ds,
            "reference_temporal_scale_factor": ts,
        },
    )

    # ── 4. Prompt context ─────────────────────────────────────────────────────
    ctx_tensors: dict[str, Any] = {}
    for k, v in raw_tensors.items():
        if k.startswith("ctx."):
            rest = k[4:]  # e.g. "positive.video_encoding"
            ctx_tensors[rest] = v

    save_safetensors(
        OUT_DIR / "prompt_context.safetensors",
        ctx_tensors,
        {"format": "ltx-prompt-context/1"},
    )

    print("Done. Files in /tmp/ltx-smoke/:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KiB)")

    # ── 5. Create a 1-second test video with ffmpeg ───────────────────────────
    import subprocess
    import shutil

    video_path = OUT_DIR / "test_input.mp4"
    # Get pixel dims from fixture metadata
    ph = meta.get("pixel_h", "8")
    pw = meta.get("pixel_w", "8")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration=1:size={pw}x{ph}:rate=24",
        "-pix_fmt", "yuv420p",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print("WARNING: ffmpeg failed; trying 8x8 fallback")
        cmd[5] = "testsrc=duration=1:size=8x8:rate=24"
        subprocess.run(cmd, check=True, capture_output=True)
    print(f"  test_input.mp4  ({video_path.stat().st_size} bytes)")

    print("\nRun the smoke test with:")
    print(
        f"""  cargo build --release -p ltx-cli --features metal 2>&1 | tail -3
  /usr/bin/time -l target/release/ltx \\
    {video_path} /tmp/ltx-smoke/out \\
    --transformer /tmp/ltx-smoke/transformer.safetensors \\
    --video-vae /tmp/ltx-smoke/vae.safetensors \\
    --lora /tmp/ltx-smoke/lora.safetensors:{1.0} \\
    --prompt-context /tmp/ltx-smoke/prompt_context.safetensors \\
    --num-inference-steps 2 \\
    --chunk-len 3 \\
    --overlap 0"""
    )


if __name__ == "__main__":
    main()
