# Cross-crate contracts (wave 1)

Branches are built in parallel. These interfaces are fixed so the crates fit together. You may add
items. To change a listed signature, first message the agent that consumes it.

## ltx-weights (branch feat/ltx-weights) — consumed by ltx-vae, ltx-vae-decoder, ltx-dit

```rust
pub struct WeightStore { /* memory-mapped safetensors files */ }
impl WeightStore {
    /// Opens one or more safetensors files. Keys are rewritten by `map` (port of the reference
    /// SDOps: prefix strip + renames), so lookups use the reference module's state_dict names.
    pub fn open(paths: &[impl AsRef<Path>], map: &KeyMap) -> Result<Self, WeightError>;
    pub fn metadata(&self, key: &str) -> Option<&str>;               // merged __metadata__
    pub fn config(&self) -> Result<serde_json::Value, WeightError>;   // parsed __metadata__["config"]
    pub fn keys(&self) -> impl Iterator<Item = &str>;
    pub fn contains(&self, key: &str) -> bool;
    pub fn shape(&self, key: &str) -> Result<&[usize], WeightError>;
    /// f32 host copy: bf16/f16/f32 and FP8 (with the reference's scale convention) dequantized,
    /// merged LoRA deltas applied.
    pub fn read(&self, key: &str) -> Result<HostTensor, WeightError>;
    /// W += strength * (B @ A) * (alpha / rank), key matching as in ltx_core.loader.fuse_loras.
    pub fn merge_lora(&mut self, lora: &LoraFile, strength: f32) -> Result<MergeReport, WeightError>;
    pub fn scope(&self, prefix: &str) -> Scope<'_>;
}
pub struct KeyMap { /* ordered rules */ }
impl KeyMap {
    pub fn identity() -> Self;            // fixtures saved with module.state_dict() names
    pub fn transformer() -> Self;         // alpha_gen transformer SDOps
    pub fn video_encoder() -> Self;       // video VAE encoder SDOps (diffusion-VAE checkpoint layout)
    pub fn video_decoder() -> Self;       // diffusion video VAE decoder SDOps
}
pub struct LoraFile { /* opened LoRA safetensors */ }
impl LoraFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WeightError>;
    pub fn metadata(&self, key: &str) -> Option<&str>;
    /// reference_downscale_factor / reference_temporal_scale_factor (default 1 each).
    pub fn ic_layout(&self) -> Result<ltx_shape::IcLoraLayout, WeightError>;
}
pub struct HostTensor { pub shape: Vec<usize>, pub data: Vec<f32> }
impl HostTensor {
    pub fn into_tensor<B: Backend, const D: usize>(self, device: &B::Device) -> Result<Tensor<B, D>, WeightError>;
}
pub struct Scope<'a> { /* store + prefix */ }
impl Scope<'_> {
    pub fn scope(&self, child: &str) -> Scope<'_>;   // joins with "."
    pub fn contains(&self, name: &str) -> bool;
    pub fn tensor<B: Backend, const D: usize>(&self, name: &str, device: &B::Device) -> Result<Tensor<B, D>, WeightError>;
    pub fn optional<B: Backend, const D: usize>(&self, name: &str, device: &B::Device) -> Result<Option<Tensor<B, D>>, WeightError>;
}
```
Until `feat/ltx-weights` merges, consumers can `git fetch origin feat/ltx-weights` and stack on it
(`git rebase origin/feat/ltx-weights`), then rebase onto `main` after it merges.

## Model crates (ltx-vae, ltx-vae-decoder, ltx-dit)
- Each model has `fn load(scope: &Scope, config: &<Crate>Config, device) -> Result<Self, Error>` and a
  `Config` that deserializes (serde) from the same JSON the reference configurator reads.
- Tensor layouts follow the reference: video pixels `[b, 3, f, h, w]` in [-1, 1]; latents
  `[b, 128, f', h', w']`, normalized as the reference encoder returns them.
- ltx-dit mirrors the reference `Modality` inputs for the video stream (patchified latent tokens,
  per-token timesteps, positions, text context + mask) and returns what the reference transformer
  returns for video. Name fields after the reference.

## ltx-sampler (branch feat/ltx-sampler)
- Defines the model-side trait the sampler calls. It mirrors the reference denoiser protocol
  (`ltx_pipelines.utils.denoisers`, `ltx_core.components.*`), so a later adapter wraps ltx-dit with no
  change to the math. Generic over `B: Backend`. No dependency on ltx-dit.

## ltx-budget / ltx-chunk / ltx-backend / ltx-io
- ltx-budget never depends on a backend crate. Calibration takes a closure
  `FnMut(PixelShape) -> Result<u64 /* peak bytes */, E>`.
- ltx-chunk is pure index and pixel math over `f32` frame planes; no I/O, no Burn.
- ltx-backend defines `AlphaBackend` and the Python backend. It may depend on ltx-shape, ltx-io.
