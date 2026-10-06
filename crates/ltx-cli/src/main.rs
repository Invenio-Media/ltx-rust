use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use ltx_backend::PythonBackend;
use ltx_pipeline::{PipelineConfig, run_video};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum QuantizationArg {
    None,
    Fp8,
}

impl QuantizationArg {
    const fn as_runner_value(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Fp8 => "fp8",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OffloadArg {
    None,
    Model,
    Sequential,
    Disk,
}

impl OffloadArg {
    const fn as_runner_value(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Model => "model",
            Self::Sequential => "sequential",
            Self::Disk => "disk",
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "ltx", about = "Generate LTX Alpha Gen mattes")]
struct Cli {
    /// Input video file.
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Directory for alpha EXR frames.
    #[arg(value_name = "OUTPUT_DIR")]
    output_dir: PathBuf,

    /// Python interpreter with the LTX-2 environment.
    #[arg(long, value_name = "PATH", default_value = "python3")]
    python: PathBuf,

    /// Path to `python/alphagen_runner.py`. Defaults to this checkout's runner.
    #[arg(long, value_name = "PATH")]
    runner: Option<PathBuf>,

    /// LTX-2.5 transformer safetensors checkpoint.
    #[arg(long, value_name = "PATH")]
    transformer: PathBuf,

    /// LTX-2.5 video VAE safetensors checkpoint.
    #[arg(long = "video-vae", value_name = "PATH")]
    video_vae: PathBuf,

    /// `LoRA` safetensors path and strength as PATH:STRENGTH. May be repeated.
    #[arg(long, value_name = "PATH:STRENGTH", required = true)]
    lora: Vec<String>,

    /// Prompt text. Used when --prompt-context is absent.
    #[arg(long, default_value = "")]
    prompt: String,

    /// Negative prompt text.
    #[arg(
        long,
        default_value = "worst quality, inconsistent motion, blurry, jittery, distorted"
    )]
    negative_prompt: String,

    /// Precomputed Gemma prompt context safetensors file.
    #[arg(long, value_name = "PATH")]
    prompt_context: Option<PathBuf>,

    /// Diffusion steps per chunk.
    #[arg(long, default_value_t = 30)]
    num_inference_steps: u32,

    /// Video CFG guidance scale.
    #[arg(long, default_value_t = 1.0)]
    cfg_scale: f32,

    /// Runner weight quantization.
    #[arg(long, value_enum, default_value_t = QuantizationArg::None)]
    quantization: QuantizationArg,

    /// Runner offload mode.
    #[arg(long, value_enum, default_value_t = OffloadArg::None)]
    offload: OffloadArg,

    /// Frame rate for `RoPE` conditioning.
    #[arg(long, default_value_t = 24.0)]
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
    let runner = cli.runner.clone().unwrap_or_else(default_runner_path);
    let runner_args = build_runner_args(&cli)?;
    let runner_arg_refs: Vec<&str> = runner_args.iter().map(String::as_str).collect();
    let backend = PythonBackend::spawn(&cli.python, &runner, &runner_arg_refs)
        .with_context(|| format!("failed to spawn runner {}", runner.display()))?;
    let config = PipelineConfig {
        chunk_len: cli.chunk_len,
        overlap: cli.overlap,
        seed: cli.seed,
        matte_prefix: cli.matte_prefix,
        seam_keyframes: !cli.no_seam_keyframes,
    };
    let report = run_video(&backend, &cli.input, &cli.output_dir, &config)?;
    println!(
        "wrote {} alpha frames from {} input frames ({} padded frames, {} chunks)",
        report.written_frames, report.input_frames, report.padded_frames, report.chunk_count
    );
    Ok(())
}

fn default_runner_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("python")
        .join("alphagen_runner.py")
}

fn build_runner_args(cli: &Cli) -> Result<Vec<String>> {
    let mut args = vec![
        "--transformer".to_owned(),
        path_to_string(&cli.transformer)?,
        "--video-vae".to_owned(),
        path_to_string(&cli.video_vae)?,
    ];
    for lora in &cli.lora {
        args.push("--lora".to_owned());
        args.push(lora.clone());
    }
    args.push("--prompt".to_owned());
    args.push(cli.prompt.clone());
    args.push("--negative-prompt".to_owned());
    args.push(cli.negative_prompt.clone());
    if let Some(prompt_context) = &cli.prompt_context {
        args.push("--prompt-context".to_owned());
        args.push(path_to_string(prompt_context)?);
    }
    args.push("--num-inference-steps".to_owned());
    args.push(cli.num_inference_steps.to_string());
    args.push("--cfg-scale".to_owned());
    args.push(cli.cfg_scale.to_string());
    args.push("--quantization".to_owned());
    args.push(cli.quantization.as_runner_value().to_owned());
    args.push("--offload".to_owned());
    args.push(cli.offload.as_runner_value().to_owned());
    args.push("--frame-rate".to_owned());
    args.push(cli.frame_rate.to_string());
    Ok(args)
}

fn path_to_string(path: &Path) -> Result<String> {
    path.to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("path contains non-UTF-8 bytes: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn clap_definition_is_valid() {
        Cli::command().debug_assert();
    }
}
