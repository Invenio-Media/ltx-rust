"""Parity fixture for `ltx-burn` (end-to-end `BurnBackend`).

Builds tiny random-init model components, runs a simplified alpha-gen
pipeline step by step (mirroring ``BurnBackend::run_chunk_impl``), and
saves all weights and noise tensors so the Rust test can reproduce the
result deterministically.

Run with the reference venv::

    /Users/keithmanlove/Documents/Projects/ltx-rust-tools/LTX-2/.venv/bin/python \\
        -W ignore tools/parity/ltx-burn.py

LTX-2 commit: 9ec55f9
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LTX2_CORE = (
    "/Users/keithmanlove/Documents/Projects/ltx-rust-tools/LTX-2"
    "/packages/ltx-core/src"
)
LTX2_PIPES = (
    "/Users/keithmanlove/Documents/Projects/ltx-rust-tools/LTX-2"
    "/packages/ltx-pipelines/src"
)
for p in (LTX2_CORE, LTX2_PIPES):
    if p not in sys.path:
        sys.path.insert(0, p)

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "crates" / "ltx-burn" / "tests" / "fixtures"
FIXTURE_PATH = FIXTURE_DIR / "parity.safetensors"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

LTX2_COMMIT = "9ec55f9"

# ---------------------------------------------------------------------------
# Tiny model dimensions
#
# Encoder: patch_size=1, compress_time × 1, compress_space × 1
#   → temporal_factor = 2, spatial_factor = 2, latent_channels = 4
#
# Target video: 3 frames, 4×4 pixels → latent [1, 4, 2, 2, 2] = 8 tokens
#
# LoRA: reference_temporal_scale_factor = 2, reference_downscale_factor = 1
#   Ref video: temporal subsample → 2 pixel frames → 1 latent frame
#   → ref_latent [1, 4, 1, 2, 2] = 4 ref tokens
#   → total = 12 tokens
# ---------------------------------------------------------------------------
PIXEL_F = 3
PIXEL_H = 4
PIXEL_W = 4
LAT_C = 4
LAT_F = 2        # (3-1)//2 + 1
LAT_H = 2        # 4 // 2
LAT_W = 2        # 4 // 2
TARGET_TOKENS = LAT_F * LAT_H * LAT_W   # 8
REF_TS = 2       # reference_temporal_scale_factor (LoRA metadata)
REF_DS = 1       # reference_downscale_factor
REF_LAT_F = 1   # ref latent frames after temporal subsample + VAE encode
REF_TOKENS = REF_LAT_F * LAT_H * LAT_W  # 4
TOTAL_TOKENS = TARGET_TOKENS + REF_TOKENS  # 12
KF_STRENGTH = 0.95
FPS = 24.0
NUM_STEPS = 2
CONTEXT_S = 4         # prompt context tokens
INNER_DIM = 32        # 4 heads × 8 head_dim
# cross_attention_dim == inner_dim (LTX-2.5 22B invariant: connector maps to inner_dim)
CONTEXT_D = INNER_DIM

# ---------------------------------------------------------------------------
# Tiny configs
# ---------------------------------------------------------------------------
ENCODER_CFG = {
    "_class_name": "CausalDiffusionVAE",
    "latent_channels": LAT_C,
    "encoder": {
        "in_channels": 3,
        "out_channels": LAT_C,
        "dims": 3,
        "patch_size": 1,
        "norm_layer": "pixel_norm",
        "latent_log_var": "uniform",
        "spatial_padding_mode": "zeros",
        "blocks": [
            ["compress_time", {}],
            ["compress_space", {}],
        ],
    },
}

DECODER_CFG = {
    "in_channels": LAT_C,
    "out_channels": 3,
    "patch_size": 2,
    "head_dim": 16,                       # minimum for default_rope_dim_split
    "stage_channels": [16, 16, 16, 16, 16],  # multiples of head_dim=16
    "stage_depths": [1, 1, 1, 1, 1],
    "stage_kernels": [[1, 1, 1]] * 5,     # 1×1×1: works with tiny 2×2 spatial
    "upsamples": [
        {"stride": [1, 1, 1], "out_channels_reduction_factor": 1},
        {"stride": [1, 1, 1], "out_channels_reduction_factor": 1},
        {"stride": [1, 1, 1], "out_channels_reduction_factor": 1},
        {"stride": [1, 1, 1], "out_channels_reduction_factor": 1},
    ],
    "stage5_kernel": [1, 1, 1],
    "t_emb_dim": 16,
    "default_num_inference_steps": NUM_STEPS,
    "model_output_type": "v",
    "timestep_scale_multiplier": 1.0,
}

TRANSFORMER_CFG = {
    "num_attention_heads": 4,
    "attention_head_dim": 8,
    "in_channels": LAT_C,
    "out_channels": LAT_C,
    "num_layers": 2,
    "cross_attention_dim": CONTEXT_D,
    "norm_eps": 1e-6,
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "timestep_scale_multiplier": 1000,
    "use_middle_indices_grid": True,
    "rope_type": "split",
    "ff_bias": False,
    "apply_gated_attention": False,
    "cross_attention_adaln": False,
    "use_prompt_adaln_single": True,
    "use_keyframes_abs_pos_embedding": False,
}

# ---------------------------------------------------------------------------
# safetensors writer
# ---------------------------------------------------------------------------
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
    meta: dict[str, str] = metadata or {}
    hdr: dict[str, Any] = {}
    data_parts: list[bytes] = []
    offset = 0
    for name, t in tensors.items():
        t_cont = t.detach().contiguous().cpu()
        if t.dtype == torch.bfloat16:
            raw = t_cont.view(torch.uint8).numpy().tobytes()
        else:
            raw = t_cont.numpy().tobytes()
        hdr[name] = {
            "dtype": _DTYPE_MAP[t.dtype],
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
    print(f"  saved {path.relative_to(REPO_ROOT)}  ({kb} KiB)")


# ---------------------------------------------------------------------------
# Build models
# ---------------------------------------------------------------------------
def build_encoder():
    from ltx_core.model.video_vae.model_configurator import VideoEncoderConfigurator
    torch.manual_seed(10)
    enc = VideoEncoderConfigurator.from_metadata(
        {"config": {"vae": ENCODER_CFG}}
    )
    enc.eval()
    with torch.no_grad():
        torch.manual_seed(11)
        enc.per_channel_statistics.get_buffer("std-of-means").copy_(
            torch.abs(torch.randn(LAT_C)) + 0.5
        )
        enc.per_channel_statistics.get_buffer("mean-of-means").copy_(
            torch.randn(LAT_C) * 0.1
        )
    return enc


def build_decoder():
    from ltx_core.model.video_vae.diffusion_video_decoder import DiffusionVideoDecoder
    from ltx_core.model.video_vae.transformer import EagerSdpaAttention
    from ltx_core.model.video_vae.transformer.apply import _set_attention_function
    torch.manual_seed(20)
    cfg = DECODER_CFG
    dec = DiffusionVideoDecoder(
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
    _set_attention_function(dec, EagerSdpaAttention())
    dec.eval()
    return dec


def build_transformer():
    from ltx_core.model.transformer.model import LTXModel, LTXModelType
    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.model.transformer.attention import (
        AttentionFunction,
        AttentionOps,
        MaskedAttentionFunction,
    )
    cpu_ops = AttentionOps(
        attention_function=AttentionFunction.SDPA_MATH.to_callable(),
        masked_attention_function=MaskedAttentionFunction.SDPA_MATH.to_callable(),
    )
    torch.manual_seed(30)
    cfg = TRANSFORMER_CFG
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=cfg["num_attention_heads"],
        attention_head_dim=cfg["attention_head_dim"],
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        num_layers=cfg["num_layers"],
        cross_attention_dim=cfg["cross_attention_dim"],
        norm_eps=cfg["norm_eps"],
        positional_embedding_theta=cfg["positional_embedding_theta"],
        positional_embedding_max_pos=cfg["positional_embedding_max_pos"],
        timestep_scale_multiplier=cfg["timestep_scale_multiplier"],
        rope_type=LTXRopeType.SPLIT,
        ff_bias=cfg["ff_bias"],
        apply_gated_attention=cfg["apply_gated_attention"],
        caption_projection=None,
        cross_attention_adaln=cfg["cross_attention_adaln"],
        use_prompt_adaln_single=cfg["use_prompt_adaln_single"],
        use_keyframes_abs_pos_embedding=cfg["use_keyframes_abs_pos_embedding"],
        attention_ops=cpu_ops,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Position grid (matches make_target_positions / make_reference_positions in Rust)
# ---------------------------------------------------------------------------
def make_positions(lat_f, lat_h, lat_w, time_scale, hw_scale, fps):
    """[1, 3, T, 2] position tensor matching Rust `make_target_positions`."""
    T = lat_f * lat_h * lat_w
    td, hd, wd = [], [], []
    for t_idx in range(T):
        fi = t_idx // (lat_h * lat_w)
        rem = t_idx - fi * lat_h * lat_w
        hi = rem // lat_w
        wi = rem % lat_w
        # causal-fix: t_start = max(0, fi * ts + 1 - ts)
        ts_raw_s = max(0, fi * time_scale + 1 - time_scale)
        ts_raw_e = max(0, (fi + 1) * time_scale + 1 - time_scale)
        td += [ts_raw_s / fps, ts_raw_e / fps]
        hd += [hi * hw_scale, (hi + 1) * hw_scale]
        wd += [wi * hw_scale, (wi + 1) * hw_scale]
    t_t = torch.tensor(td).reshape(T, 2)
    h_t = torch.tensor(hd).reshape(T, 2)
    w_t = torch.tensor(wd).reshape(T, 2)
    return torch.stack([t_t, h_t, w_t], 0).unsqueeze(0)  # [1, 3, T, 2]


def make_ref_positions(ref_lat_f, lat_h, lat_w, time_scale, hw_scale,
                       fps, temporal_scale_factor, downscale_factor):
    """[1, 3, T_ref, 2] matching Rust `make_reference_positions`."""
    T = ref_lat_f * lat_h * lat_w
    ref_fps = fps / temporal_scale_factor
    shift_s = (temporal_scale_factor - 1) / fps
    td, hd, wd = [], [], []
    for t_idx in range(T):
        fi = t_idx // (lat_h * lat_w)
        rem = t_idx - fi * lat_h * lat_w
        hi = rem // lat_w
        wi = rem % lat_w
        ts_raw_s = max(0, fi * time_scale + 1 - time_scale)
        ts_raw_e = max(0, (fi + 1) * time_scale + 1 - time_scale)
        t_s = max(0.0, ts_raw_s / ref_fps - shift_s)
        t_e = max(0.0, ts_raw_e / ref_fps - shift_s)
        h_s = hi * hw_scale * downscale_factor
        h_e = (hi + 1) * hw_scale * downscale_factor
        w_s = wi * hw_scale * downscale_factor
        w_e = (wi + 1) * hw_scale * downscale_factor
        td += [t_s, t_e]
        hd += [h_s, h_e]
        wd += [w_s, w_e]
    t_t = torch.tensor(td).reshape(T, 2)
    h_t = torch.tensor(hd).reshape(T, 2)
    w_t = torch.tensor(wd).reshape(T, 2)
    return torch.stack([t_t, h_t, w_t], 0).unsqueeze(0)  # [1, 3, T_ref, 2]


# ---------------------------------------------------------------------------
# Pipeline (mirrors BurnBackend::run_chunk_impl step by step)
# ---------------------------------------------------------------------------
def run_pipeline(enc, dec, transformer):
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.video_vae import diffusion_tiling

    torch.manual_seed(1)
    rgb = torch.rand(1, 3, PIXEL_F, PIXEL_H, PIXEL_W) * 2.0 - 1.0  # [-1,1]

    torch.manual_seed(2)
    kf_rgb = torch.rand(1, 3, 1, PIXEL_H, PIXEL_W) * 2.0 - 1.0

    with torch.no_grad():
        # ── step 1: reference VAE encode ─────────────────────────────────────
        ref_indices = [0] + list(range(1, PIXEL_F, REF_TS))  # [0, 1]
        ref_rgb = rgb[:, :, ref_indices]      # [1, 3, 2, H, W]
        ref_latent = enc(ref_rgb)              # [1, 4, 1, 2, 2]
        assert ref_latent.shape == (1, LAT_C, REF_LAT_F, LAT_H, LAT_W), (
            f"unexpected ref_latent shape {ref_latent.shape}"
        )

        # ── step 2: seam keyframe VAE encode ─────────────────────────────────
        kf_latent = enc(kf_rgb)               # [1, 4, 1, 2, 2]

        # ── step 3: patchify (b c f h w → b (fhw) c) ─────────────────────────
        def patchify5d(t):  # [B,C,F,H,W] → [B, F*H*W, C]
            b, c, f, h, w = t.shape
            return t.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, c)

        target_tokens = patchify5d(torch.zeros(1, LAT_C, LAT_F, LAT_H, LAT_W))
        ref_tokens = patchify5d(ref_latent)
        kf_tokens = patchify5d(kf_latent)   # [1, 4, 4] (1 frame × 2×2)
        TPS = LAT_H * LAT_W  # tokens per frame = 4

        # ── step 4: positions ─────────────────────────────────────────────────
        time_scale = 2  # temporal_factor
        hw_scale = 2    # spatial_factor
        target_pos = make_positions(LAT_F, LAT_H, LAT_W, time_scale, hw_scale, FPS)
        ref_pos = make_ref_positions(REF_LAT_F, LAT_H, LAT_W,
                                     time_scale, hw_scale, FPS, REF_TS, REF_DS)

        # ── step 5: initial LatentState ───────────────────────────────────────
        denoise_mask = torch.ones(1, TARGET_TOKENS, 1)
        clean_latent = target_tokens.clone()

        # ── step 6: VideoReferenceCondition (strength=1.0) ────────────────────
        # ref_mask = 1 - strength = 0 (all frozen/clean)
        ref_mask = torch.zeros(1, REF_TOKENS, 1)
        ref_latent_zeros = torch.zeros(1, REF_TOKENS, LAT_C)

        all_lat = torch.cat([target_tokens, ref_latent_zeros], 1)  # [1,12,4]
        all_mask = torch.cat([denoise_mask, ref_mask], 1)           # [1,12,1]
        all_pos = torch.cat([target_pos, ref_pos], 2)               # [1,3,12,2]
        all_clean = torch.cat([clean_latent, ref_tokens], 1)        # [1,12,4]

        # ── step 7: ImageKeyframeCondition (frame 0, strength=0.95) ──────────
        # PR #17: mask = state.denoise_mask * keep + frame_mask * (1 - strength)
        # where keep = 1 - frame_mask, frame_mask = 1 at the keyframe tokens.
        # We set clean_latent at frame 0 to kf_tokens;
        # denoise_mask at those tokens becomes 1 - strength = 0.05.
        kf_frame_idx = 0  # latent frame index
        tok_start = kf_frame_idx * TPS    # 0
        tok_end = tok_start + TPS          # 4

        # Build a mask that is 1 only at [tok_start:tok_end] (over TOTAL_TOKENS).
        frame_mask = torch.zeros(1, TOTAL_TOKENS, 1)
        frame_mask[:, tok_start:tok_end, :] = 1.0

        # img_clean: zeros everywhere except the keyframe tokens.
        img_clean = torch.zeros(1, TOTAL_TOKENS, LAT_C)
        img_clean[:, tok_start:tok_end, :] = kf_tokens

        keep = 1.0 - frame_mask
        all_clean = all_clean * keep + img_clean * frame_mask
        all_mask = all_mask * keep + frame_mask * (1.0 - KF_STRENGTH)

        # ── step 8: sigma schedule ────────────────────────────────────────────
        sigmas_list = LTX2Scheduler().execute(
            steps=NUM_STEPS, num_tokens=TOTAL_TOKENS
        )
        sigmas = [float(s) for s in sigmas_list]
        sigma_max = sigmas[0]

        # ── step 9: initial noise (GaussianNoiser.apply_with_noise) ──────────
        torch.manual_seed(100)
        initial_noise = torch.randn(1, TOTAL_TOKENS, LAT_C)
        # noised = clean*(1-mask) + (latent*(1-σ) + noise*σ) * mask
        noised = all_lat * (1.0 - sigma_max) + initial_noise * sigma_max
        all_lat_noised = all_clean * (1.0 - all_mask) + noised * all_mask

        # ── step 10: prompt context ───────────────────────────────────────────
        torch.manual_seed(200)
        context = torch.randn(1, CONTEXT_S, CONTEXT_D)

        # ── step 11: Euler denoising loop ─────────────────────────────────────
        cur_lat = all_lat_noised
        n_layers = TRANSFORMER_CFG["num_layers"]
        for step_i in range(NUM_STEPS):
            sigma = sigmas[step_i]
            sigma_next = sigmas[step_i + 1]

            timesteps = (all_mask * sigma).squeeze(-1)  # [1, 12]
            sigma_t = torch.tensor([sigma])

            video_mod = Modality(
                latent=cur_lat,
                sigma=sigma_t,
                timesteps=timesteps,
                positions=all_pos,
                context=context,
                enabled=True,
                context_mask=None,
                attention_mask=None,
                keyframes_mask=None,
            )
            perturb = BatchedPerturbationConfig.empty(
                1, n_layers, cur_lat.device, cur_lat.dtype
            )
            x0_v, _ = transformer(video=video_mod, audio=None, perturbations=perturb)

            # post_process: x0_adj = x0 * mask + clean * (1 - mask)
            x0_adj = x0_v * all_mask + all_clean * (1.0 - all_mask)

            # Euler step: v = (cur - x0) / σ; next = cur + v * (σ_next - σ)
            if abs(sigma) > 1e-6:
                velocity = (cur_lat - x0_adj) / sigma
            else:
                velocity = torch.zeros_like(cur_lat)
            cur_lat = cur_lat + velocity * (sigma_next - sigma)

        # ── step 12: unpatchify (keep target tokens) ──────────────────────────
        target_tok = cur_lat[:, :TARGET_TOKENS, :]    # [1, 8, 4]
        # [1, T, C] → [1, F, H, W, C] → [1, C, F, H, W]
        denoised_lat = target_tok.reshape(
            1, LAT_F, LAT_H, LAT_W, LAT_C
        ).permute(0, 4, 1, 2, 3)

        # ── step 13: VAE decode ───────────────────────────────────────────────
        # Compute canvas shape from stages 1-3 output.
        latent_padded = diffusion_tiling.pad_trailing_latent_for_natten_border(
            denoised_lat, dec._natten_trailing_pad_latent_frames
        )
        feat_s4 = dec.forward_stages_1_to_3(latent_padded, drop_leading_frame=True)
        # Get canvas size from stage-4 context.
        ctx_test = dec.forward_stage_4(feat_s4.clone(), drop_leading_frame=True, pad_trailing=True)
        _, ctx_t, ctx_h, ctx_w, _ = ctx_test.shape
        canvas_h = ctx_h * dec.patch_size
        canvas_w = ctx_w * dec.patch_size
        print(f"  canvas: T={ctx_t}  H={canvas_h}  W={canvas_w}")

        torch.manual_seed(300)
        decoder_noise = torch.randn(1, dec.out_channels, ctx_t, canvas_h, canvas_w)

        # Timestep schedule for decoder.
        dec_ts = dec.default_inference_timesteps  # [n_steps]
        dec_ts_b = dec_ts.unsqueeze(0).expand(1, -1)  # [1, n_steps]

        pixels_raw = dec._decode_one_tile(
            feat_s4,
            decoder_noise.clone(),
            is_origin=True,
            timestep=dec_ts_b,
            pad_trailing=True,
        )
        print(f"  pixels_raw shape: {tuple(pixels_raw.shape)}")

        # Rust crops to content: pixel_f, pixel_h=H_lat*32, pixel_w=W_lat*32.
        # With all-1 upsamples in this tiny config, pixel dims match canvas dims.
        pix_f = pixels_raw.shape[2]
        pix_h = pixels_raw.shape[3]
        pix_w = pixels_raw.shape[4]
        pixels = pixels_raw  # no crop needed for all-1-stride config

        # ── step 14: RGB → alpha via Rec.709 luminance ────────────────────────
        # pixels is [1, 3, F, H, W] in [-1, 1].
        pix_01 = (pixels + 1.0) / 2.0
        alpha = (
            0.2126 * pix_01[:, 0]
            + 0.7152 * pix_01[:, 1]
            + 0.0722 * pix_01[:, 2]
        ).clamp(0.0, 1.0)  # [1, F, H, W]
        alpha_flat = alpha.reshape(-1)

    return {
        "rgb": rgb,
        "kf_rgb": kf_rgb,
        "initial_noise": initial_noise,
        "decoder_noise": decoder_noise,
        "output_alpha": alpha_flat,
        "pix_f": pix_f,
        "pix_h": pix_h,
        "pix_w": pix_w,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Building tiny encoder …")
    enc = build_encoder()
    print("Building tiny decoder …")
    dec = build_decoder()
    print("Building tiny transformer …")
    transformer = build_transformer()

    print("Running reference pipeline …")
    out = run_pipeline(enc, dec, transformer)

    print(
        f"  output_alpha: shape {tuple(out['output_alpha'].shape)}"
        f"  range [{out['output_alpha'].min():.4f}, {out['output_alpha'].max():.4f}]"
    )

    # Collect all tensors for the fixture.
    tensors: dict[str, torch.Tensor] = {}

    for k, v in enc.state_dict().items():
        tensors[f"enc.{k}"] = v.float()
    for k, v in dec.state_dict().items():
        tensors[f"dec.{k}"] = v.float()
    for k, v in transformer.state_dict().items():
        tensors[f"tfm.{k}"] = v.float()

    # Tiny LoRA (rank-2 targeting patchify_proj).
    torch.manual_seed(40)
    tensors["lora.patchify_proj.lora_A.weight"] = (
        torch.randn(2, LAT_C) * 0.01
    ).float()
    tensors["lora.patchify_proj.lora_B.weight"] = (
        torch.randn(INNER_DIM, 2) * 0.01
    ).float()

    # Prompt context (matches the context used in run_pipeline).
    torch.manual_seed(200)
    ctx = torch.randn(1, CONTEXT_S, CONTEXT_D).to(torch.bfloat16)
    tensors["ctx.positive.video_encoding"] = ctx
    tensors["ctx.positive.attention_mask"] = torch.ones(1, CONTEXT_S)
    tensors["ctx.negative.video_encoding"] = torch.zeros(
        1, CONTEXT_S, CONTEXT_D, dtype=torch.bfloat16
    )
    tensors["ctx.negative.attention_mask"] = torch.ones(1, CONTEXT_S)

    # Test inputs and outputs.
    tensors["input_rgb"] = out["rgb"]
    tensors["input_kf_rgb"] = out["kf_rgb"]
    tensors["initial_noise"] = out["initial_noise"].float()
    tensors["decoder_noise"] = out["decoder_noise"].float()
    tensors["output_alpha"] = out["output_alpha"].float()

    metadata = {
        "ltx2_commit": LTX2_COMMIT,
        "encoder_config": json.dumps(ENCODER_CFG, separators=(",", ":")),
        "decoder_config": json.dumps(DECODER_CFG, separators=(",", ":")),
        "transformer_config": json.dumps(
            {"transformer": TRANSFORMER_CFG}, separators=(",", ":")
        ),
        "lora_reference_downscale_factor": str(REF_DS),
        "lora_reference_temporal_scale_factor": str(REF_TS),
        "kf_strength": str(KF_STRENGTH),
        "num_inference_steps": str(NUM_STEPS),
        "cfg_scale": "1.0",
        "frame_rate": str(FPS),
        "pixel_f": str(PIXEL_F),
        "pixel_h": str(PIXEL_H),
        "pixel_w": str(PIXEL_W),
    }

    print("Saving fixture …")
    save_safetensors(FIXTURE_PATH, tensors, metadata)
    mb = FIXTURE_PATH.stat().st_size / 1024 / 1024
    print(f"  total: {mb:.2f} MB (limit 2 MB)")
    if mb > 2.0:
        print("  WARNING: fixture exceeds 2 MB limit — reduce config dimensions")
    print("Done.")


if __name__ == "__main__":
    main()
