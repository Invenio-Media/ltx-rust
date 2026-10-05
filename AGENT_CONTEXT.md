# ltx-rust shared context

## Goal
Port the LTX-2.5 Alpha Gen IC-LoRA pipeline (`ltx_pipelines.alpha_gen` in Lightricks/LTX-2)
to Rust on Burn 0.21. The finished tool:
1. measures how many frames the GPU can process in one pass (`ltx-budget`),
2. splits a long clip into overlapping chunks of that length and blends them (`ltx-chunk`),
3. runs each chunk through a backend: first a Python backend that calls the reference, later a
   pure Burn backend (VAE encoder, DiT, sampler, diffusion VAE decoder ported to Burn).

## Places
- Repo: `/Users/keithmanlove/Documents/Projects/ltx-rust` (GitHub `Invenio-Media/ltx-rust`, base `main`).
  Do not work in this checkout. Make your own worktree:
  `git -C /Users/keithmanlove/Documents/Projects/ltx-rust fetch origin && git -C /Users/keithmanlove/Documents/Projects/ltx-rust worktree add /Users/keithmanlove/Documents/Projects/ltx-rust-worktrees/<branch> -b <branch> origin/main`
- Reference source (read it, do not edit): `/Users/keithmanlove/Documents/Projects/ltx-rust-tools/LTX-2`
  at commit `9ec55f9` (LTX-2 v1.4.2). Key packages: `packages/ltx-core/src/ltx_core`,
  `packages/ltx-pipelines/src/ltx_pipelines` (`alpha_gen.py`, `iclora_utils.py`, `utils/blocks.py`,
  `utils/helpers.py`, `utils/args.py`, `hdr_ic_lora.py` for seam keyframes).
- Reference Python env (torch 2.14, MPS works): `/Users/keithmanlove/Documents/Projects/ltx-rust-tools/LTX-2/.venv/bin/python`.
  `import ltx_core, ltx_pipelines` works. Use `python -W ignore` to hide a colour-science warning.
- PR helpers: `/Users/keithmanlove/Documents/Projects/ltx-rust-tools/pr-wait.sh <N>` (blocks until all
  checks finish, then prints feedback) and `pr-feedback.sh <N>` (prints feedback now).
- No real Lightricks weights are on this machine (the model repos are gated). Parity tests use tiny
  random-init configs of the reference modules.

## Facts already established
- alpha_gen: one stage, full (not distilled) model, Euler with `LTX2Scheduler`, `GaussianNoiser`,
  video CFG scale 1.0 by default, STG off, rescale 0.7, video only (`a_context=None`, no audio branch).
  Reference video enters as `VideoConditionByReferenceLatent` (clean tokens appended to the target).
- LTX-2.5 ships a split pack: transformer, video VAE, audio VAE, text encoder as separate files. The
  LTX-2.5 video VAE is a diffusion VAE (`config.vae._class_name != CausalVideoAutoencoder`): conv
  encoder, diffusion (transformer, neighborhood attention) decoder. `is_diffusion_video_vae()` in
  `ltx_core/model/video_vae/model_configurator.py` decides from checkpoint metadata.
- Model configs live in safetensors `__metadata__["config"]` (JSON). Rust loaders read the config from
  there, as the Python configurators do.
- The text encoder (Gemma) is NOT ported. The prompt context is computed once in Python and shipped
  as a `.safetensors` file.
- Shape rules are in `crates/ltx-shape` (merged): `PixelShape` (8k+1 frames, 32x grid, keeps its
  `ScaleFactors`), `LatentShape`, `IcLoraLayout` (reference downscale/temporal factors from LoRA
  metadata), `SequenceTokens`, `floor_frames`, `ceil_frames`, `ceil_spatial`. Reuse it; do not
  re-derive shape math.

## Code rules (CI enforces; the reviewer checks)
- Workspace lints in root `Cargo.toml` deny clippy pedantic + nursery, `unwrap_used`, `expect_used`,
  `indexing_slicing`, `arithmetic_side_effects`, `as_conversions`, `panic`, `todo`, `unimplemented`,
  `unreachable`, `string_slice`, `exit`. Tests may unwrap/expect/panic/index (`clippy.toml`).
  Use checked/saturating integer ops, `get()`, `try_from`, `NonZero*`. No crate-level `allow`.
  A narrow `#[expect(lint, reason = "...")]` on one item is allowed only when it is provably safe and
  the reason says why.
- `clippy.toml` exempts `burn::tensor::Tensor` operators from `arithmetic_side_effects`. If clippy still
  flags tensor `+ - * /`, find the path that works (clippy resolves def paths) and fix `clippy.toml`
  in your PR.
- Every crate: `[lints] workspace = true`, `version.workspace = true` etc. (copy `crates/ltx-shape/Cargo.toml`).
  Third-party deps go in root `[workspace.dependencies]`; crates use `dep.workspace = true`.
  Internal deps: `ltx-shape = { path = "../ltx-shape" }`.
- Burn: `burn = { workspace = true, features = [...] }`, version 0.21. Model code is generic over
  `B: burn::tensor::backend::Backend`. Tests run on the CPU backend (`burn` feature `ndarray`) so CI
  (Linux, no GPU) runs them. Offer cargo features `metal` (`burn/metal`) and `cuda` (`burn/cuda`)
  where a crate runs models.
- Errors: `thiserror` enums per crate. No `TODO`, stubs, placeholders, or dead code.
- Doc comments in plain, short English. Do not narrate.
- Tests must catch real bugs: behavior, boundaries, error cases, parity. No tests of wiring or of
  incidental defaults.

## Parity fixtures (model crates)
- Script per crate: `tools/parity/<crate>.py`, run with the reference venv. It builds the reference
  module with a tiny config and a fixed seed (random init, no Lightricks weights), runs it, and
  saves weights + inputs + outputs as one safetensors file in `crates/<crate>/tests/fixtures/`.
  Store the module config and the LTX-2 commit in the safetensors metadata. Keep each fixture under
  2 MB. Exercise every code path that the real config uses (causal padding, RoPE, AdaLN, masks...).
- The Rust test loads the fixture, builds the Burn module from the stored config and weights, and
  compares outputs with an absolute + relative tolerance stated in the test (f32 on CPU).
- `tools/parity/README.md` says how to regenerate (shared file; add one line for your script).

## Branch and PR protocol
1. Work only in your worktree on your branch. Commit in logical steps.
2. Before opening the PR: `cargo fmt --all`, `cargo clippy --workspace --all-targets`,
   `cargo test --workspace` all clean. Commit `Cargo.lock`.
3. All `gh` calls: `env -u GITHUB_TOKEN gh ...` (the env token cannot create PRs; the keyring login can).
   `git push -u origin <branch>`, then
   `env -u GITHUB_TOKEN gh pr create --base main --title "..." --body "..."`. The body has
   `## Changes` and `## Checks` (only checks you ran, with real results).
4. Run `pr-wait.sh <N>`. Two automated things report: GitHub Actions (`Test`, `Clippy`, `Rustfmt Check`,
   `MSRV (1.92)`) and the Claude reviewer (a `claude[bot]` conversation comment, about 1-3 minutes).
5. For each reviewer point: fix it if valid; if not, decline with evidence. Push the fixes (this
   re-runs the reviewer), then post one `gh pr comment` that answers every point (fixed / declined + why).
   Repeat until the newest review lists no blocking issue and CI is green. Nits may be declined.
   Stop after 6 rounds and report what is left.
6. Never merge, never push to `main`, never force-push another agent's branch. The orchestrator merges.
7. If `main` moves and your PR conflicts, `git fetch && git rebase origin/main`, resolve, re-run checks,
   `git push --force-with-lease`.
8. Shared files that several branches touch: root `Cargo.toml` (add only your deps), `README.md` crate
   table (add only your row), `Cargo.lock`, `tools/parity/README.md`. Keep those edits minimal.

## Final report (your last message)
Branch, PR number and URL, CI state, the newest reviewer verdict, what you declined and why, the
public API of your crate (signatures), and anything a later crate must know.
