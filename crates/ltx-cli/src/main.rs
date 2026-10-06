use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use ltx_backend::PythonBackend;
use ltx_pipeline::{PipelineConfig, run_video};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum QuantizationArg {
    None,
    Fp8,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OffloadArg {
    None,
    Model,
    Sequential,
    Disk,
}

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

    /// Path to `python/alphagen_runner.py`. Defaults to `python/alphagen_runner.py`
    /// next to the `ltx` binary, then to the one in this source checkout.
    #[arg(long, value_name = "PATH", env = "LTX_RUNNER")]
    runner: Option<PathBuf>,

    /// LTX-2.5 transformer safetensors checkpoint.
    #[arg(long, value_name = "PATH")]
    transformer: PathBuf,

    /// LTX-2.5 video VAE safetensors checkpoint.
    #[arg(long = "video-vae", value_name = "PATH")]
    video_vae: PathBuf,

    /// `LoRA` safetensors path and strength as PATH:STRENGTH. May be repeated.
    #[arg(long, value_name = "PATH:STRENGTH", required = true, value_parser = parse_lora)]
    lora: Vec<LoraSpec>,

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
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u32).range(1..))]
    num_inference_steps: u32,

    /// Video CFG guidance scale.
    #[arg(long, default_value_t = 1.0, value_parser = parse_finite_f32)]
    cfg_scale: f32,

    /// Runner weight quantization.
    #[arg(long, value_enum, default_value_t = QuantizationArg::None)]
    quantization: QuantizationArg,

    /// Runner offload mode.
    #[arg(long, value_enum, default_value_t = OffloadArg::None)]
    offload: OffloadArg,

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
    let config = PipelineConfig {
        chunk_len: cli.chunk_len,
        overlap: cli.overlap,
        seed: cli.seed,
        matte_prefix: cli.matte_prefix.clone(),
        seam_keyframes: !cli.no_seam_keyframes,
    };
    config
        .validate()
        .context("invalid --chunk-len / --overlap")?;
    check_inputs_exist(&cli)?;
    let runner = resolve_runner(cli.runner.as_deref())?;

    let runner_args = build_runner_args(&cli)?;
    let runner_arg_refs: Vec<&str> = runner_args.iter().map(String::as_str).collect();
    let backend = PythonBackend::spawn(&cli.python, &runner, &runner_arg_refs)
        .with_context(|| format!("failed to spawn runner {}", runner.display()))?;
    let report = run_video(&backend, &cli.input, &cli.output_dir, &config)?;
    println!(
        "wrote {} alpha frames from {} input frames ({} padded frames, {} chunks)",
        report.written_frames, report.input_frames, report.padded_frames, report.chunk_count
    );
    Ok(())
}

/// Fail before the runner loads models when a required file is missing.
fn check_inputs_exist(cli: &Cli) -> Result<()> {
    let mut files = vec![
        ("input video", cli.input.as_path()),
        ("--transformer", cli.transformer.as_path()),
        ("--video-vae", cli.video_vae.as_path()),
    ];
    files.extend(cli.lora.iter().map(|lora| ("--lora", lora.path.as_path())));
    if let Some(prompt_context) = &cli.prompt_context {
        files.push(("--prompt-context", prompt_context.as_path()));
    }
    for (label, path) in files {
        if !path.is_file() {
            bail!("{label} file not found: {}", path.display());
        }
    }
    Ok(())
}

/// Pick the runner script: `--runner` / `LTX_RUNNER`, then
/// `python/alphagen_runner.py` beside the binary, then the source checkout.
fn resolve_runner(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if path.is_file() {
            return Ok(path.to_path_buf());
        }
        bail!("--runner file not found: {}", path.display());
    }
    let beside_exe = std::env::current_exe().ok().and_then(|exe| {
        exe.parent()
            .map(|dir| dir.join("python").join("alphagen_runner.py"))
    });
    let in_checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("python")
        .join("alphagen_runner.py");
    beside_exe
        .into_iter()
        .chain(std::iter::once(in_checkout))
        .find(|path| path.is_file())
        .ok_or_else(|| anyhow!("alphagen_runner.py not found; pass --runner or set LTX_RUNNER"))
}

fn value_name<T: ValueEnum>(value: &T) -> Result<String> {
    value
        .to_possible_value()
        .map(|v| v.get_name().to_owned())
        .ok_or_else(|| anyhow!("value enum has no name"))
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
        args.push(format!("{}:{}", path_to_string(&lora.path)?, lora.strength));
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
    args.push(value_name(&cli.quantization)?);
    args.push("--offload".to_owned());
    args.push(value_name(&cli.offload)?);
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

    #[test]
    fn lora_spec_splits_on_last_colon() {
        assert_eq!(
            parse_lora(r"C:\models\alpha.safetensors:0.75").unwrap(),
            LoraSpec {
                path: PathBuf::from(r"C:\models\alpha.safetensors"),
                strength: 0.75,
            }
        );
    }

    #[test]
    fn lora_spec_rejects_missing_or_bad_strength() {
        assert!(parse_lora("alpha.safetensors").is_err());
        assert!(parse_lora("alpha.safetensors:").is_err());
        assert!(parse_lora("alpha.safetensors:strong").is_err());
        assert!(parse_lora("alpha.safetensors:NaN").is_err());
        assert!(parse_lora(":1.0").is_err());
    }

    #[test]
    fn numeric_options_reject_out_of_range_values() {
        assert!(parse_positive_f32("0").is_err());
        assert!(parse_positive_f32("-24").is_err());
        assert!(parse_positive_f32("inf").is_err());
        assert!(parse_finite_f32("NaN").is_err());
        assert_eq!(
            parse_finite_f32("-1.5").unwrap().to_bits(),
            (-1.5_f32).to_bits()
        );
        let zero_steps = Cli::try_parse_from([
            "ltx",
            "in.mp4",
            "out",
            "--transformer",
            "t",
            "--video-vae",
            "v",
            "--lora",
            "a:1",
            "--num-inference-steps",
            "0",
        ]);
        assert!(zero_steps.is_err());
    }
}
