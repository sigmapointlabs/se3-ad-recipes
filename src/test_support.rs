//! Shared test/bench scaffolding: a zero-dependency deterministic RNG and a
//! small statistics helper.
//!
//! Consumed by the `#[cfg(test)]` modules (`graph`, `isserlis`) and the
//! `bench-support` examples (`nees_consistency`, `posegraph_consistency`).
//! These had each carried a hand-copied SplitMix64 + Box–Muller generator, and
//! the copies had silently drifted — some cached the second Box–Muller deviate,
//! some discarded it — so "same seed" no longer meant "same stream" across
//! consumers.  This single source keeps the generator honest: `normal()` caches
//! the spare, so two uniforms yield two normals and one seed drives one
//! reproducible stream everywhere.
//!
//! Gated behind `#[cfg(any(test, feature = "bench-support"))]`; not part of the
//! stable public API.

/// SplitMix64 state advance plus two-uniform Box–Muller normals: deterministic,
/// dependency-free, and reproducible from its seed.
pub struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    /// Seed the generator.  Any `u64` is a valid seed.
    pub fn new(seed: u64) -> Self {
        Rng {
            state: seed,
            spare: None,
        }
    }

    /// Next 64-bit value (SplitMix64).
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[lo, hi)`.
    pub fn uniform_in(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard normal via Box–Muller.  Caches the second deviate, so two
    /// uniforms produce two normals — the property callers rely on for a
    /// reproducible stream across consumers.
    pub fn normal(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
        self.spare = Some(r * s);
        r * c
    }
}

/// Mean and unbiased (sample) standard deviation of a small slice.
pub fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = if xs.len() < 2 {
        0.0
    } else {
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    };
    (mean, var.sqrt())
}
