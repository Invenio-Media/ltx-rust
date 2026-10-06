# ltx-rust

A Rust port of the LTX-2.5 Alpha Gen IC-LoRA pipeline
([`Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen`](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen)),
built on [Burn](https://burn.dev). The reference implementation is
`ltx_pipelines.alpha_gen` in [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2).

The tool finds how many frames the GPU can process in one pass, splits a long
clip into overlapping chunks of that length, and blends the chunks into one
alpha matte.

## Crates

| Crate | Job |
| --- | --- |
| `ltx-shape` | Pixel and latent shape rules, IC-LoRA token counts |
| `ltx-budget` | GPU memory budget estimation and solver (device query, quadratic fit, cache, spatial fallback) |
| `ltx-weights` | Memory-mapped safetensors store; FP8 dequant; LoRA merging |
| `ltx-chunk` | Temporal/spatial chunk planning, reflection padding, smoothstep blend, seam conditioning |
| `ltx-io` | ffprobe metadata, ffmpeg frame decode, EXR matte write, preview encode |
| `ltx-backend` | `AlphaBackend` trait and Python backend (JSON-lines over stdio) |
| `ltx-sampler` | Euler denoising schedule and classifier-free guidance protocol |
| `ltx-vae` | Diffusion-video VAE encoder modules |
| `ltx-dit` | LTX video transformer modules |
| `ltx-vae-decoder` | Diffusion-video VAE decoder modules |
| `ltx-burn` | Pure Burn backend: loads transformer, VAE, LoRA, prompt context; runs `euler_denoising_loop`; outputs alpha mattes |
| `ltx-pipeline` | End-to-end chunk orchestration from video input to EXR matte output |
| `ltx-cli` | `ltx` command: `--transformer`, `--video-vae`, `--lora`, `--prompt-context` |

## Usage

### Prompt context (run once per prompt)

```sh
python -W ignore tools/prompt_context.py \
  --transformer ltx-2.5-transformer.safetensors \
  --text-encoder /path/to/gemma \
  --prompt "generate alpha matte" \
  --out prompt_context.safetensors
```

### Inference (CPU / NdArray)

```sh
cargo run --release -p ltx-cli -- input.mp4 out_mattes \
  --transformer ltx-2.5-transformer.safetensors \
  --video-vae ltx-2.5-video-vae.safetensors \
  --lora alpha-gen-ic-lora.safetensors:1.0 \
  --prompt-context prompt_context.safetensors
```

### Inference (Apple Metal, f32)

Burn 0.21 on Metal cannot run bf16 matmul, so the `metal` feature computes in
f32. The video transformer's f32 weights must fit in the GPU working set
(about 107 GB on a 128 GB M4 Max).

```sh
cargo run --release -p ltx-cli --features metal -- input.mp4 out_mattes \
  --transformer ltx-2.5-transformer.safetensors \
  --video-vae ltx-2.5-video-vae.safetensors \
  --lora alpha-gen-ic-lora.safetensors:1.0 \
  --prompt-context prompt_context.safetensors
```

`--chunk-len` (8k+1), `--overlap` (multiple of 8), `--num-inference-steps`,
`--cfg-scale`, and `--frame-rate` are the key tuning flags.
Run `ltx --help` for all options.
## Checks

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked
cargo test --workspace --locked
```
