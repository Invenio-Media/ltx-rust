//! LTX-2 sigma schedule ([`LTX2Scheduler`]).
//!
//! Ports `ltx_core.components.schedulers.LTX2Scheduler.execute` exactly.
//! Returns a `Vec<f32>` of length `steps + 1` in descending order (1.0 → 0.0
//! after shifting and optional stretching).
//!
//! **Default step counts** (from `ltx_pipelines.utils.constants`):
//! - LTX-2.0 models: **40 steps**
//! - LTX-2.3+ models (including LTX-2.5 alpha\_gen): **30 steps**
//!
//! Use [`LTX2Scheduler::execute`] with [`SchedulerConfig::default`] to match
//! the reference.

use crate::SamplerError;

/// Anchors for the token-count-dependent shift interpolation.
const BASE_SHIFT_ANCHOR: f32 = 1024.0;
const MAX_SHIFT_ANCHOR: f32 = 4096.0;

/// Configuration for [`LTX2Scheduler::execute`].
///
/// Mirrors the keyword arguments of `LTX2Scheduler.execute` in the reference Python.
#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    /// Maximum shift (for token count = `MAX_SHIFT_ANCHOR`). Reference default: 2.05.
    pub max_shift: f32,
    /// Base shift (for token count = `BASE_SHIFT_ANCHOR`). Reference default: 0.95.
    pub base_shift: f32,
    /// When true, stretch the schedule so the last non-zero sigma equals `terminal`.
    pub stretch: bool,
    /// Target value for the last non-zero sigma after stretching. Reference default: 0.1.
    pub terminal: f32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_shift: 2.05,
            base_shift: 0.95,
            stretch: true,
            terminal: 0.1,
        }
    }
}

/// LTX-2 default sigma schedule with token-count-dependent shifting.
///
/// Identical to `LTX2Scheduler.execute` in `ltx_core.components.schedulers`.
pub struct LTX2Scheduler;

impl LTX2Scheduler {
    /// Build the sigma schedule.
    ///
    /// # Parameters
    /// - `steps`: Number of denoising steps. The returned `Vec` has `steps + 1` elements.
    /// - `num_tokens`: Latent token count (= frames × height × width after patchification).
    ///   Pass `4096` (`MAX_SHIFT_ANCHOR`) when no latent is available yet.
    /// - `config`: Schedule hyper-parameters.
    ///
    /// # Errors
    /// Returns [`SamplerError::Overflow`] if `steps + 1` overflows `usize`.
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "scheduler math intentionally uses f32 to match the reference implementation"
    )]
    pub fn execute(
        steps: u32,
        num_tokens: u64,
        config: SchedulerConfig,
    ) -> Result<Vec<f32>, SamplerError> {
        let n_steps = usize::try_from(steps).map_err(|_| SamplerError::Overflow)?;
        let len = n_steps.checked_add(1).ok_or(SamplerError::Overflow)?;
        let denom = n_steps as f32; // safe: usize→f32 via #[expect] below

        // linspace(1.0, 0.0, steps + 1)
        #[expect(
            clippy::as_conversions,
            reason = "i: usize ≤ steps ≤ u32::MAX; the value fits in f32 with \
                      acceptable rounding above 2^24"
        )]
        let mut sigmas: Vec<f32> = (0..len)
            .map(|i| {
                if denom == 0.0 {
                    1.0_f32
                } else {
                    1.0_f32 - i as f32 / denom
                }
            })
            .collect();

        // Token-count-dependent flux shift.
        let tokens_f32 = num_tokens as f32; // large token counts: acceptable f32 rounding
        let mm = (config.max_shift - config.base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR);
        let b_off = mm.mul_add(-BASE_SHIFT_ANCHOR, config.base_shift);
        let sigma_shift = mm.mul_add(tokens_f32, b_off);
        let exp_shift = sigma_shift.exp();

        for s in &mut sigmas {
            if *s != 0.0 {
                *s = exp_shift / exp_shift.mul_add(1.0, 1.0 / *s - 1.0);
            }
        }

        // Stretch so the last non-zero sigma equals `terminal`.
        if config.stretch {
            let non_zero_indices: Vec<usize> = sigmas
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v == 0.0 { None } else { Some(i) })
                .collect();

            if let Some(&last_idx) = non_zero_indices.last() {
                let last_s = *sigmas.get(last_idx).ok_or(SamplerError::Overflow)?;
                let scale_factor = (1.0 - last_s) / (1.0 - config.terminal);

                for &idx in &non_zero_indices {
                    let s = sigmas.get_mut(idx).ok_or(SamplerError::Overflow)?;
                    *s = 1.0 - (1.0 - *s) / scale_factor;
                }
            }
        }

        Ok(sigmas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_length() {
        let s = LTX2Scheduler::execute(5, 1024, SchedulerConfig::default()).unwrap();
        assert_eq!(s.len(), 6);
    }

    #[test]
    fn schedule_ends_at_zero() {
        let s = LTX2Scheduler::execute(5, 1024, SchedulerConfig::default()).unwrap();
        assert!(s.last().copied().unwrap().abs() < 1e-6);
    }

    #[test]
    fn schedule_descending() {
        let s = LTX2Scheduler::execute(10, 4096, SchedulerConfig::default()).unwrap();
        for pair in s.windows(2) {
            let a = pair.first().copied().unwrap();
            let b = pair.get(1).copied().unwrap();
            assert!(a >= b, "sigmas must be descending: {a} < {b}");
        }
    }

    #[test]
    fn known_values_5steps_1024tokens() {
        // Verified against `LTX2Scheduler().execute(steps=5, default_number_of_tokens=1024)`.
        let got = LTX2Scheduler::execute(5, 1024, SchedulerConfig::default()).unwrap();
        let expected = [1.0_f32, 0.8694, 0.6963, 0.4560, 0.1000, 0.0000];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-3,
                "sigma mismatch: got {g}, expected {e}"
            );
        }
    }

    #[test]
    fn no_stretch() {
        let cfg = SchedulerConfig {
            stretch: false,
            ..SchedulerConfig::default()
        };
        let s = LTX2Scheduler::execute(3, 4096, cfg).unwrap();
        assert!(s.last().copied().unwrap().abs() < 1e-6);
        assert_eq!(s.len(), 4);
    }
}
