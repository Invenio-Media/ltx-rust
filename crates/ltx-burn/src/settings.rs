//! Model file paths and generation hyper-parameters.

use std::path::PathBuf;

/// Paths to all model checkpoint files for one `BurnBackend` instance.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    /// LTX-2.5 transformer safetensors checkpoint.
    pub transformer: PathBuf,
    /// LTX-2.5 video VAE safetensors checkpoint (encoder + decoder).
    pub video_vae: PathBuf,
    /// `IC-LoRA` safetensors files paired with their merge strengths.
    ///
    /// At least one `LoRA` is required: `alpha_gen` always uses `IC-LoRA` conditioning.
    /// `LoRA` reference scale metadata must agree across files (values of 1 are "unset"
    /// and ignored; any two non-1 values must match).
    pub loras: Vec<(PathBuf, f32)>,
    /// Precomputed Gemma prompt context (output of `tools/prompt_context.py`).
    ///
    /// The file must carry `format = "ltx-prompt-context/1"` in its safetensors
    /// metadata and must have been generated from the same transformer checkpoint.
    pub prompt_context: PathBuf,
}

/// Diffusion generation hyper-parameters.
#[derive(Debug, Clone)]
pub struct GenerationSettings {
    /// Euler denoising steps per chunk.
    ///
    /// Reference default for LTX-2.5 `alpha_gen`: **30**.
    pub num_inference_steps: u32,
    /// Classifier-free guidance scale.
    ///
    /// Reference default: **1.0** (conditioned-only; unconditioned pass skipped).
    /// Set to a value other than 1.0 to enable CFG; the negative context in the
    /// prompt-context file is then used.
    pub cfg_scale: f32,
    /// Video frame rate fed to the `RoPE` position encoding.
    pub frame_rate: f32,
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            num_inference_steps: 30,
            cfg_scale: 1.0,
            frame_rate: 24.0,
        }
    }
}
