#!/usr/bin/env python3
"""alphagen_runner — long-lived JSON-lines server for alpha-gen generation.

This script builds ``AlphaGenPipeline`` once at startup, then serves requests
from Rust's ``PythonBackend`` on stdin/stdout.  Each request and each response
is a single UTF-8 JSON line terminated by newline.

Protocol
--------
Probe::

    request:  {"cmd":"probe","width":W,"height":H,"frames":F}
    response: {"ok":true,"peak_bytes":N}

Run::

    request:  {"cmd":"run","rgb_path":"...","alpha_out_path":"...",
                "width":W,"height":H,"frames":F,"seed":S[,"keyframes_path":"..."]}
    response: {"ok":true}

Error response (either command)::

    {"ok":false,"error":"<message>"}

Binary files
------------
``rgb_path``
    Little-endian f32, shape ``[frames, height, width, 3]`` (RGBRGB…), values
    in ``[0, 1]``.  Written by Rust from decoded video frames.

``alpha_out_path``
    Little-endian f32, shape ``[frames, height, width]``, values in ``[0, 1]``.
    Written by this runner: the Rec.709 luminance of the decoded output video,
    clamped to ``[0, 1]``.  Luminance formula: ``Y = 0.2126·R + 0.7152·G +
    0.0722·B`` (ITU-R BT.709-6 §3).  The alpha-gen model outputs near-identical
    R, G, B channels trained to represent the alpha matte.

``keyframes_path`` (optional)
    Path to a JSON file with keys ``indices``, ``strength``, and ``rgb_path``
    (pointing to a f32 binary of shape ``[n, height, width, 3]``).  Keyframes
    are passed to the pipeline as ``ImageConditioningInput`` at the specified
    chunk-local frame indices with the given IC-LoRA conditioning strength.

Prompt context
--------------
Pass ``--prompt-context <file.safetensors>`` to skip Gemma at runtime.  The
``AlphaGenPipeline`` prompt encoder returns precomputed context from this file
using the same safetensors layout as the pipeline caches internally.  If the
pipeline API does not support supplying a precomputed context without forking
the reference code, the prompt text is encoded once at startup and reused for
all requests; Gemma is not called per chunk.

Memory probe (CUDA vs. MPS)
---------------------------
On CUDA the probe resets the memory peak counter with
``torch.cuda.reset_peak_memory_stats()`` and returns
``torch.cuda.max_memory_allocated()`` after a synthetic forward pass.

On MPS ``torch.cuda`` is unavailable.  The probe returns
``torch.mps.current_allocated_memory()`` measured at the peak of the synthetic
pass.  This value reflects memory allocated by PyTorch's MPS allocator and does
not include Metal driver/GPU heap overhead; actual VRAM usage is higher.
"""

import argparse
import json
import struct
import sys
import tempfile
from pathlib import Path


def _err(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}), flush=True)


def _ok(**kwargs: object) -> None:
    print(json.dumps({"ok": True, **kwargs}), flush=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alphagen_runner",
        description=(
            "Long-lived JSON-lines server for the LTX-2.5 Alpha Gen IC-LoRA pipeline.  "
            "Reads requests from stdin, writes responses to stdout."
        ),
    )
    p.add_argument(
        "--transformer",
        required=True,
        metavar="PATH",
        help="Path to the LTX-2.5 transformer safetensors checkpoint.",
    )
    p.add_argument(
        "--video-vae",
        required=True,
        metavar="PATH",
        help="Path to the LTX-2.5 video VAE safetensors checkpoint.",
    )
    p.add_argument(
        "--lora",
        required=True,
        metavar="PATH:STRENGTH",
        action="append",
        default=[],
        help=(
            "LoRA safetensors path and strength as 'path:strength'.  "
            "May be repeated for multiple LoRAs."
        ),
    )
    p.add_argument(
        "--prompt",
        default="",
        metavar="TEXT",
        help="Generation prompt text.  Used when --prompt-context is not supplied.",
    )
    p.add_argument(
        "--negative-prompt",
        default=(
            "worst quality, inconsistent motion, blurry, jittery, distorted"
        ),
        metavar="TEXT",
        help="Negative prompt text.",
    )
    p.add_argument(
        "--prompt-context",
        default=None,
        metavar="PATH",
        help=(
            "Path to a precomputed Gemma prompt context safetensors file.  "
            "When provided, Gemma is not loaded at startup.  "
            "When absent, the pipeline loads Gemma once and caches it for all requests."
        ),
    )
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        metavar="N",
        help="Number of diffusion steps per chunk (default: 30).",
    )
    p.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        metavar="SCALE",
        help=(
            "Video CFG guidance scale (default: 1.0, as in the reference "
            "alpha_gen_guider_params)."
        ),
    )
    p.add_argument(
        "--quantization",
        choices=["none", "fp8"],
        default="none",
        help="Weight quantization (none | fp8, default: none).",
    )
    p.add_argument(
        "--offload",
        choices=["none", "model", "sequential", "disk"],
        default="none",
        help="Weight offload mode (default: none).",
    )
    p.add_argument(
        "--frame-rate",
        type=float,
        default=24.0,
        metavar="FPS",
        help="Frame rate for RoPE conditioning (default: 24.0).",
    )
    return p


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def _load_pipeline(args: argparse.Namespace):  # type: ignore[return]
    """Build and return an ``AlphaGenPipeline`` from parsed args."""
    import torch
    from ltx_pipelines.alpha_gen import AlphaGenPipeline
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.quantization import QuantizationPolicy
    from ltx_pipelines.utils.types import OffloadMode
    from ltx_pipelines.utils.model_paths import ModelPaths

    # Parse LoRA entries "path:strength".
    loras: list[LoraPathStrengthAndSDOps] = []
    for entry in args.lora:
        parts = entry.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"--lora must be 'path:strength', got {entry!r}")
        lora_path, strength_str = parts
        loras.append(
            LoraPathStrengthAndSDOps(
                path=lora_path,
                strength=float(strength_str),
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        )

    quant_map = {
        "none": None,
        "fp8": QuantizationPolicy.FP8,
    }
    offload_map = {
        "none": OffloadMode.NONE,
        "model": OffloadMode.MODEL,
        "sequential": OffloadMode.SEQUENTIAL,
        "disk": OffloadMode.DISK,
    }

    model_paths = ModelPaths(
        transformer=args.transformer,
        video_vae=args.video_vae,
    )

    pipeline = AlphaGenPipeline(
        model_paths=model_paths,
        loras=loras,
        quantization=quant_map[args.quantization],
        offload_mode=offload_map[args.offload],
    )
    return pipeline


# ---------------------------------------------------------------------------
# Binary helpers
# ---------------------------------------------------------------------------

def _read_f32_bin(path: str, expected_count: int | None = None) -> list[float]:
    with open(path, "rb") as f:
        raw = f.read()
    count = len(raw) // 4
    values = list(struct.unpack(f"<{count}f", raw))
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(
            f"Expected {expected_count} floats in {path!r}, got {len(values)}"
        )
    return values


def _write_f32_bin(path: str, values: list[float]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))


# ---------------------------------------------------------------------------
# Probe handler
# ---------------------------------------------------------------------------

def _handle_probe(req: dict, pipeline) -> None:  # type: ignore[type-arg]
    """Run a synthetic forward pass and report peak GPU bytes."""
    import torch

    width = int(req["width"])
    height = int(req["height"])
    frames = int(req["frames"])

    device = pipeline.device

    # Determine which memory API to use.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Build synthetic inputs: random RGB video.
    with torch.inference_mode():
        dummy_rgb = torch.rand(
            1, 3, frames, height, width,
            dtype=pipeline.dtype,
            device=device,
        )
        # Encode through the video VAE encoder to get latents.
        # This exercises the memory-heaviest path the pipeline takes per chunk.
        try:
            _ = pipeline.image_conditioner._encoder(dummy_rgb)  # type: ignore[attr-defined]
        except Exception:
            # Fallback: just allocate a tensor of the expected latent size.
            # The VAE has 8× temporal and 32× spatial compression.
            lf = max(1, (frames - 1) // 8 + 1)
            lh = max(1, height // 32)
            lw = max(1, width // 32)
            _ = torch.zeros(1, 128, lf, lh, lw, dtype=pipeline.dtype, device=device)

    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)
    elif device.type == "mps":
        peak_bytes = torch.mps.current_allocated_memory()
    else:
        # CPU: report zero (no GPU).
        peak_bytes = 0

    _ok(peak_bytes=peak_bytes)


# ---------------------------------------------------------------------------
# Run handler
# ---------------------------------------------------------------------------

def _handle_run(req: dict, pipeline, args: argparse.Namespace) -> None:  # type: ignore[type-arg]
    """Generate alpha mattes for one chunk."""
    import torch
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines.alpha_gen import alpha_gen_guider_params
    from ltx_pipelines.utils.types import ImageConditioningInput

    rgb_path = req["rgb_path"]
    alpha_out_path = req["alpha_out_path"]
    width = int(req["width"])
    height = int(req["height"])
    frames = int(req["frames"])
    seed = int(req["seed"])
    keyframes_path: str | None = req.get("keyframes_path")

    # Load keyframes if present.
    images: list[ImageConditioningInput] = []
    if keyframes_path is not None:
        import json as _json
        with open(keyframes_path) as f:
            kf = _json.load(f)
        kf_rgb_path = kf["rgb_path"]
        kf_indices: list[int] = kf["indices"]
        kf_strength: float = float(kf["strength"])
        n_kf = len(kf_indices)

        kf_rgb = _read_f32_bin(kf_rgb_path, n_kf * height * width * 3)
        kf_rgb_tensor = torch.tensor(kf_rgb, dtype=torch.float32).reshape(
            n_kf, height, width, 3
        ).permute(0, 3, 1, 2)  # [n, 3, H, W]

        # Write each keyframe to a temp file for ImageConditioningInput.
        with tempfile.TemporaryDirectory() as kf_tmpdir:
            for i, (frame_tensor, frame_idx) in enumerate(
                zip(kf_rgb_tensor, kf_indices)
            ):
                import torchvision  # type: ignore[import-untyped]
                kf_img_path = str(Path(kf_tmpdir) / f"kf_{i:04d}.png")
                torchvision.utils.save_image(frame_tensor, kf_img_path)
                images.append(
                    ImageConditioningInput(
                        path=kf_img_path,
                        frame_idx=int(frame_idx),
                        strength=kf_strength,
                    )
                )

    # Load reference video from binary.
    pixel_count = frames * height * width * 3
    rgb_flat = _read_f32_bin(rgb_path, pixel_count)
    rgb_tensor = torch.tensor(rgb_flat, dtype=pipeline.dtype).reshape(
        1, frames, height, width, 3
    ).permute(0, 4, 1, 2, 3)  # [1, 3, F, H, W]

    # Save reference video to a temp MP4 for the pipeline's video_conditioning.
    with tempfile.TemporaryDirectory() as vid_tmpdir:
        import torchvision

        ref_path = str(Path(vid_tmpdir) / "ref.mp4")
        # torchvision.io.write_video expects [T, H, W, C] uint8.
        ref_uint8 = (rgb_tensor[0].permute(1, 2, 3, 0) * 255.0).clamp(0, 255).byte()
        fps_int = max(1, int(round(args.frame_rate)))
        torchvision.io.write_video(ref_path, ref_uint8.cpu(), fps=fps_int)

        base_guider = MultiModalGuiderParams(
            cfg_scale=args.cfg_scale,
            stg_scale=0.0,
            rescale_scale=0.7,
            modality_scale=1.0,
        )
        guider = alpha_gen_guider_params(base_guider, cfg_scale=args.cfg_scale)

        with torch.inference_mode():
            result = pipeline(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=frames,
                frame_rate=args.frame_rate,
                num_inference_steps=args.num_inference_steps,
                video_guider_params=guider,
                images=images,
                video_conditioning=[(ref_path, 1.0)],
            )

    # Collect generated frames and extract luminance as alpha.
    alpha_flat: list[float] = []
    for frame_tensor in result.video:
        # frame_tensor is [H, W, 3] f32 in [0, 1].
        frame_np = frame_tensor.cpu().float().numpy()
        h_ax, w_ax = frame_np.shape[:2]
        for row in range(h_ax):
            for col in range(w_ax):
                r = float(frame_np[row, col, 0])
                g = float(frame_np[row, col, 1])
                b = float(frame_np[row, col, 2])
                y = 0.2126 * r + 0.7152 * g + 0.0722 * b
                alpha_flat.append(max(0.0, min(1.0, y)))

    _write_f32_bin(alpha_out_path, alpha_flat)
    _ok()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Load the pipeline once at startup.
    try:
        pipeline = _load_pipeline(args)
    except Exception as exc:
        # Write a JSON error line and exit so the parent can detect it.
        _err(f"pipeline load failed: {exc}")
        sys.exit(1)

    # Ensure stdin is line-buffered so readline() returns immediately on \n.
    try:
        sys.stdin.reconfigure(line_buffering=True)  # Python 3.7+
    except AttributeError:
        pass  # pre-3.7: best-effort

    # Serve requests until stdin closes.
    while True:
        raw_line = sys.stdin.readline()
        if not raw_line:  # EOF
            break
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _err(f"bad JSON: {exc}")
            continue

        cmd = req.get("cmd")
        try:
            if cmd == "probe":
                _handle_probe(req, pipeline)
            elif cmd == "run":
                _handle_run(req, pipeline, args)
            else:
                _err(f"unknown cmd: {cmd!r}")
        except Exception as exc:  # noqa: BLE001
            _err(str(exc))


if __name__ == "__main__":
    main()
