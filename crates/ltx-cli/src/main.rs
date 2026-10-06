use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use ltx_burn::{BurnBackend, GenerationSettings, ModelFiles};
use ltx_pipeline::{PipelineConfig, run_video};

// ── Backend type selection ─────────────────────────────────────────────────────

/// CPU backend (default, no GPU feature selected).
#[cfg(not(any(feature = "metal", feature = "cuda")))]
type ActiveBackend = burn::backend::NdArray<f32>;
/// Device for the CPU backend.
#[cfg(not(any(feature = "metal", feature = "cuda")))]
type ActiveDevice = burn::backend::ndarray::NdArrayDevice;

/// Apple Metal GPU backend.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
type ActiveBackend = burn::backend::Metal<half::bf16>;
/// Device for the Metal backend.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
type ActiveDevice = burn::backend::metal::MetalDevice;

/// NVIDIA CUDA GPU backend.
#[cfg(feature = "cuda")]
type ActiveBackend = burn::backend::Cuda<half::bf16>;
/// Device for the CUDA backend.
#[cfg(feature = "cuda")]
type ActiveDevice = burn::backend::cuda::CudaDevice;
// ── Argument parser ────────────────────────────────────────────────────────────

/// A `LoRA` file with its merge strength.
#[derive(Debug, Clone, PartialEq)]
struct LoraSpec {
    path: PathBuf,
    strength: f32,
}

/// Parse `PATH:STRENGTH`, splitting on the last `:` so drive-letter paths work.
fn parse_lora(value: &str) -> Result<LoraSpec, String> {
    let (path, strength) = value
        .rsplit_once(':')
        .ok_or_else(|| format!("expected PATH:STRENGTH, got {value:?}"))?;
    if path.is_empty() {
        return Err(format!("missing LoRA path in {value:?}"));
    }
    let strength: f32 = strength
        .parse()
        .map_err(|_| format!("LoRA strength {strength:?} is not a number"))?;
    if !strength.is_finite() {
        return Err(format!("LoRA strength {strength} must be finite"));
    }
    Ok(LoraSpec {
        path: PathBuf::from(path),
        strength,
    })
}

fn parse_positive_f32(value: &str) -> Result<f32, String> {
    let parsed: f32 = value
        .parse()
        .map_err(|_| format!("{value:?} is not a number"))?;
    if parsed.is_finite() && parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(format!("{value} must be a finite number above 0"))
    }
}

fn parse_finite_f32(value: &str) -> Result<f32, String> {
    let parsed: f32 = value
        .parse()
        .map_err(|_| format!("{value:?} is not a number"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(format!("{value} must be finite"))
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "ltx",
    about = "Generate LTX Alpha Gen mattes with the pure Burn backend"
)]
struct Cli {
    /// Input video file.
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Directory for alpha EXR frames.
    #[arg(value_name = "OUTPUT_DIR")]
    output_dir: PathBuf,

    /// LTX-2.5 transformer safetensors checkpoint.
    #[arg(long, value_name = "PATH")]
    transformer: PathBuf,

    /// LTX-2.5 video VAE safetensors checkpoint.
    #[arg(long = "video-vae", value_name = "PATH")]
    video_vae: PathBuf,

    /// IC-LoRA safetensors path and strength as PATH:STRENGTH. May be repeated.
    #[arg(long, value_name = "PATH:STRENGTH", required = true, value_parser = parse_lora)]
    lora: Vec<LoraSpec>,

    /// Precomputed Gemma prompt context safetensors file (from `tools/prompt_context.py`).
    ///
    /// Generate with:
    ///   `python -W ignore tools/prompt_context.py`
    ///     `--transformer <path> --text-encoder <gemma-dir> --prompt "..." --out ctx.safetensors`
    #[arg(long, value_name = "PATH", required = true)]
    prompt_context: PathBuf,

    /// Euler denoising steps per chunk.
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u32).range(1..))]
    num_inference_steps: u32,

    /// Video CFG guidance scale (1.0 = conditioned only; no unconditioned pass).
    #[arg(long, default_value_t = 1.0, value_parser = parse_finite_f32)]
    cfg_scale: f32,

    /// Frame rate for `RoPE` conditioning.
    #[arg(long, default_value_t = 24.0, value_parser = parse_positive_f32)]
    frame_rate: f32,

    /// Frames per backend chunk. Must be 8k+1.
    #[arg(long, default_value_t = 49)]
    chunk_len: u32,

    /// Overlap frames between chunks. Must be a multiple of 8.
    #[arg(long, default_value_t = 8)]
    overlap: u32,

    /// Base seed. Chunk index is added to this value.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// EXR filename prefix.
    #[arg(long, default_value = "alpha")]
    matte_prefix: String,

    /// Disable seam keyframe conditioning between chunks.
    #[arg(long)]
    no_seam_keyframes: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Validate files exist before loading weights.
    check_inputs_exist(&cli)?;

    let pipeline_config = PipelineConfig {
        chunk_len: cli.chunk_len,
        overlap: cli.overlap,
        seed: cli.seed,
        matte_prefix: cli.matte_prefix.clone(),
        seam_keyframes: !cli.no_seam_keyframes,
    };
    pipeline_config
        .validate()
        .context("invalid --chunk-len / --overlap")?;

    let model_files = ModelFiles {
        transformer: cli.transformer.clone(),
        video_vae: cli.video_vae.clone(),
        loras: cli
            .lora
            .iter()
            .map(|spec| (spec.path.clone(), spec.strength))
            .collect(),
        prompt_context: cli.prompt_context.clone(),
    };

    let settings = GenerationSettings {
        num_inference_steps: cli.num_inference_steps,
        cfg_scale: cli.cfg_scale,
        frame_rate: cli.frame_rate,
    };

    let device = ActiveDevice::default();
    let backend = BurnBackend::<ActiveBackend>::load(&model_files, &settings, &device)
        .context("failed to load model weights")?;

    let report = run_video(&backend, &cli.input, &cli.output_dir, &pipeline_config)?;
    println!(
        "wrote {} alpha frames from {} input frames ({} padded, {} chunks)",
        report.written_frames, report.input_frames, report.padded_frames, report.chunk_count
    );
    Ok(())
}

/// Fail before loading any weights when a required file is missing.
fn check_inputs_exist(cli: &Cli) -> Result<()> {
    let mut files: Vec<(&str, &std::path::Path)> = vec![
        ("input video", cli.input.as_path()),
        ("--transformer", cli.transformer.as_path()),
        ("--video-vae", cli.video_vae.as_path()),
        ("--prompt-context", cli.prompt_context.as_path()),
    ];
    files.extend(cli.lora.iter().map(|s| ("--lora", s.path.as_path())));
    for (label, path) in files {
        if !path.is_file() {
            bail!("{label} file not found: {}", path.display());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_lora_accepts_valid_spec() {
        let spec = parse_lora("/path/to/lora.safetensors:0.8").unwrap();
        assert_eq!(spec.path, PathBuf::from("/path/to/lora.safetensors"));
        assert!((spec.strength - 0.8).abs() < 1e-6);
    }

    #[test]
    fn parse_lora_handles_windows_path() {
        // Last colon is the separator.
        let spec = parse_lora("C:\\models\\lora.safetensors:1.0").unwrap();
        assert_eq!(spec.path, PathBuf::from("C:\\models\\lora.safetensors"));
    }

    #[test]
    fn parse_lora_rejects_non_finite() {
        assert!(parse_lora("/path:inf").is_err());
        assert!(parse_lora("/path:nan").is_err());
    }

    #[test]
    fn parse_lora_rejects_missing_strength() {
        assert!(parse_lora("/path/to/lora.safetensors").is_err());
    }
}
