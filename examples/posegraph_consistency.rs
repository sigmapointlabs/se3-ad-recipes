//! NEES covariance-consistency study, pose-graph edition: robust
//! Gauss–Newton/FIM information vs the exact observed information (seeded-AD
//! factor Hessians, `graph::exact_hessian`) on a K = 6 SE(3) pose graph with
//! loop closures and spurious-closure contamination.
//!
//! Monte-Carlo protocol (mirrors `posegraph_reference.py`, the JAX prototype):
//!   1. Sample a truth chain X_{k+1} = X_k · Exp(u + w), u the nominal step,
//!      w ~ N(0, Σ_step); X_0 = X_anc · Exp(ξ), ξ ~ N(0, Σ_anchor).
//!   2. Generate Gaussian odometry measurements along the chain and robust
//!      (pseudo-Huber) loop-closure measurements; each closure is spurious
//!      with probability γ (offset σ = [0.3 rot, 0.5 trans]).
//!   3. Initialize by dead reckoning, solve the MAP problem by damped
//!      robust-GN with re-basing, so the converged linearization point has
//!      δ = 0 across all nodes.
//!   4. Evaluate two information matrices at the MAP:
//!        I_GN = Σ_f w_f·J_fᵀ W J_f    (what robust solvers report)
//!        H    = exact NLL Hessian via one seeded `adfn<12>` pass per factor
//!   5. NEES = ξ_errᵀ · I · ξ_err with ξ_err = stacked Log(X̂_k⁻¹ X_k^true);
//!      averaged over trials → ANEES vs the χ² consistency band, for BOTH the
//!      full 36-dim state and the marginal last pose (6-dim, via the trailing
//!      block of I⁻¹).
//!
//! Run:  cargo run --release --features bench-support --example posegraph_consistency

// Same rationale as the crate-level allow in lib.rs: numerical / matrix code
// uses index-based loops pervasively.  The doc allow keeps the aligned
// formula layout in the protocol list above.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_overindented_list_items)]

use se3_ad_recipes::Vec6;
use se3_ad_recipes::graph::{BetweenFactor, GraphProblem, PriorFactor, diagonal_sqrt_info};
use se3_ad_recipes::linalg::{Chol, cholesky_n};
use se3_ad_recipes::se3_unsafe::Pose;

// ─── Experiment constants ───────────────────────────────────────────────

const K: usize = 6;
const DIM: usize = 6 * K;
const CLOSURES: [(usize, usize); 4] = [(0, 3), (1, 4), (2, 5), (0, 5)];
const SIG_ANCHOR: Vec6 = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02];
const SIG_ODO: Vec6 = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02];
const SIG_CLO: Vec6 = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02];
const SIG_STEP: Vec6 = [0.05, 0.05, 0.05, 0.10, 0.10, 0.10];
const SPUR_SIG: Vec6 = [0.3, 0.3, 0.3, 0.5, 0.5, 0.5];
const MEAN_STEP: Vec6 = [0.0, 0.0, 0.35, 1.0, 0.0, 0.0];
const KAPPA: f64 = 3.0; // pseudo-Huber threshold on closures, whitened units
const M_TRIALS: usize = 1000;
const GAMMA_SWEEP: [f64; 3] = [0.0, 0.25, 0.5];
/// Three independent RNG seeds per sweep point; final ANEES is the mean
/// across seeds, and the between-seed standard deviation quantifies the
/// Monte-Carlo uncertainty of the reported number.
const SEEDS: [u64; 3] = [1, 2, 3];
/// CSV output path, relative to the crate root (where `cargo run` sets cwd).
const CSV_PATH: &str = "../experiments/data/posegraph.csv";

// ─── Minimal deterministic RNG (SplitMix64 + Box–Muller), zero deps ─────

struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng {
            state: seed,
            spare: None,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    /// Standard normal via Box–Muller (caches the spare deviate).
    fn normal(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
        self.spare = Some(r * s);
        r * c
    }
    /// Draw ξ ~ N(0, diag(sig²)).
    fn draw6(&mut self, sig: &Vec6) -> Vec6 {
        std::array::from_fn(|a| sig[a] * self.normal())
    }
}

fn inv6(sig: &Vec6) -> Vec6 {
    std::array::from_fn(|a| 1.0 / sig[a])
}

// ─── Problem sampling ────────────────────────────────────────────────────

/// One Monte-Carlo instance: truth chain, contaminated measurements, and a
/// `GraphProblem` initialized by dead reckoning.
fn sample_trial(rng: &mut Rng, gamma: f64) -> (Vec<Pose>, GraphProblem) {
    let x_anc = Pose::identity();

    let mut truth = Vec::with_capacity(K);
    truth.push(x_anc.compose(&Pose::exp(&rng.draw6(&SIG_ANCHOR))));
    for k in 0..K - 1 {
        let w = rng.draw6(&SIG_STEP);
        let u: Vec6 = std::array::from_fn(|a| MEAN_STEP[a] + w[a]);
        truth.push(truth[k].compose(&Pose::exp(&u)));
    }

    let mut factors = Vec::with_capacity(K - 1 + CLOSURES.len());
    let mut z_odo = Vec::with_capacity(K - 1);
    for k in 0..K - 1 {
        let rel = truth[k].inverse().compose(&truth[k + 1]);
        let z = rel.compose(&Pose::exp(&rng.draw6(&SIG_ODO)));
        z_odo.push(z);
        factors.push(BetweenFactor {
            i: k,
            j: k + 1,
            z,
            sqrt_info: diagonal_sqrt_info(&inv6(&SIG_ODO)),
            kappa: None,
        });
    }
    for &(i, j) in CLOSURES.iter() {
        let rel = truth[i].inverse().compose(&truth[j]);
        let sig = if rng.uniform() < gamma {
            &SPUR_SIG
        } else {
            &SIG_CLO
        };
        let z = rel.compose(&Pose::exp(&rng.draw6(sig)));
        factors.push(BetweenFactor {
            i,
            j,
            z,
            sqrt_info: diagonal_sqrt_info(&inv6(&SIG_CLO)),
            kappa: Some(KAPPA),
        });
    }

    // Dead-reckoning init along the odometry chain.
    let mut poses = Vec::with_capacity(K);
    poses.push(x_anc);
    for k in 0..K - 1 {
        poses.push(poses[k].compose(&z_odo[k]));
    }

    let problem = GraphProblem {
        poses,
        priors: vec![PriorFactor {
            node: 0,
            x_ref: x_anc,
            sqrt_info: diagonal_sqrt_info(&inv6(&SIG_ANCHOR)),
        }],
        factors,
        linear: vec![],
    };
    (truth, problem)
}

// ─── MAP solve: damped robust-GN with re-basing ─────────────────────────

/// Iterates X_k ← X_k·Exp(step_k) so the converged problem has δ = 0 at
/// every node.
fn solve_map(p: &mut GraphProblem, iters: usize) {
    let lambda = 1e-6;
    for _ in 0..iters {
        let g = p.gradient();
        let gnorm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        if gnorm < 1e-10 {
            break;
        }
        let mut h = p.gn_information();
        for (i, row) in h.iter_mut().enumerate() {
            row[i] += lambda;
        }
        let Some(chol) = cholesky_n(&h) else { break };
        let neg_g: Vec<f64> = g.iter().map(|x| -x).collect();
        let step = chol.solve(&neg_g);
        p.re_base(&step);
    }
}

// ─── NEES evaluation helpers ─────────────────────────────────────────────

/// xᵀ·M·x for a dense matrix.
fn quad_form(m: &[Vec<f64>], x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (row, &xi) in m.iter().zip(x) {
        let mut s = 0.0;
        for (v, &xj) in row.iter().zip(x) {
            s += v * xj;
        }
        acc += xi * s;
    }
    acc
}

/// Marginal NEES of the last pose: x_lᵀ·B⁻¹·x_l with B the trailing 6×6
/// block of I⁻¹ (I factored as `chol`).
fn last_pose_nees(chol: &Chol, xl: &[f64; 6]) -> Option<f64> {
    let dim = chol.dim();
    // Trailing 6 columns of I⁻¹, then the trailing 6×6 block B.
    let mut b = vec![vec![0.0f64; 6]; 6];
    for (col, b_col) in (dim - 6..dim).enumerate() {
        let mut e = vec![0.0f64; dim];
        e[b_col] = 1.0;
        let s = chol.solve(&e);
        for row in 0..6 {
            b[row][col] = s[dim - 6 + row];
        }
    }
    let chol_b = cholesky_n(&b)?;
    let y = chol_b.solve(xl);
    Some(xl.iter().zip(&y).map(|(a, b)| a * b).sum())
}

// ─── Monte Carlo ─────────────────────────────────────────────────────────

struct SweepRow {
    n: usize,
    skipped: usize,
    full_gn: f64,
    full_h: f64,
    last_gn: f64,
    last_h: f64,
}

fn run_sweep(gamma: f64, seed: u64) -> SweepRow {
    let mut rng = Rng::new(seed);
    let (mut s_fg, mut s_fh, mut s_lg, mut s_lh) = (0.0, 0.0, 0.0, 0.0);
    let (mut n, mut skipped) = (0usize, 0usize);

    for _ in 0..M_TRIALS {
        let (truth, mut p) = sample_trial(&mut rng, gamma);
        solve_map(&mut p, 25);

        // Stacked error in the right tangent at the estimate.
        let mut xi_err = vec![0.0f64; DIM];
        for k in 0..K {
            let e = p.poses[k].relative(&truth[k]).log();
            xi_err[6 * k..6 * k + 6].copy_from_slice(&e);
        }
        let xl: [f64; 6] = std::array::from_fn(|a| xi_err[DIM - 6 + a]);

        // Exact observed information; PD guard (the exact Hessian can be
        // indefinite under extreme spurious closures) — skip and count.
        let h_exact = p.exact_hessian();
        let Some(chol_h) = cholesky_n(&h_exact) else {
            skipped += 1;
            continue;
        };
        let i_gn = p.gn_information();
        let Some(chol_gn) = cholesky_n(&i_gn) else {
            skipped += 1;
            continue;
        };
        let (Some(lh), Some(lg)) = (last_pose_nees(&chol_h, &xl), last_pose_nees(&chol_gn, &xl))
        else {
            skipped += 1;
            continue;
        };

        s_fg += quad_form(&i_gn, &xi_err);
        s_fh += quad_form(&h_exact, &xi_err);
        s_lg += lg;
        s_lh += lh;
        n += 1;
    }
    SweepRow {
        n,
        skipped,
        full_gn: s_fg / n as f64,
        full_h: s_fh / n as f64,
        last_gn: s_lg / n as f64,
        last_h: s_lh / n as f64,
    }
}

/// Mean and unbiased (sample) standard deviation of a small slice.
fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = if xs.len() < 2 {
        0.0
    } else {
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    };
    (mean, var.sqrt())
}

fn main() {
    use std::io::Write;

    // χ² consistency bands are a property of the per-seed sample size;
    // report them as diagnostics on individual sweeps, one per dimension.
    let band_full = 1.96 * (2.0 / (DIM as f64 * M_TRIALS as f64)).sqrt() * DIM as f64;
    let band_last = 1.96 * (2.0 / (6.0 * M_TRIALS as f64)).sqrt() * 6.0;

    println!(
        "Pose-graph NEES consistency: K={} nodes, {} closures, kappa={}, \
         M={} trials/seed, {} seeds/point",
        K,
        CLOSURES.len(),
        KAPPA,
        M_TRIALS,
        SEEDS.len()
    );
    println!(
        "{:>5} {:>21} {:>21} {:>19} {:>19} {:>5}",
        "gamma",
        "full36 GN mean±std",
        "full36 H mean±std",
        "last6 GN mean±std",
        "last6 H mean±std",
        "skip"
    );

    let mut csv = {
        // Create the parent directory if it doesn't exist yet, so a fresh
        // checkout doesn't require the user to mkdir experiments/data by hand.
        if let Some(parent) = std::path::Path::new(CSV_PATH).parent() {
            std::fs::create_dir_all(parent).expect("create CSV parent directory");
        }
        std::fs::File::create(CSV_PATH).expect("open CSV for writing")
    };
    // Column semantics documented in the header row so a downstream plot
    // script (or a reviewer) can consume the file without re-reading the
    // paper. `_std` columns are between-seed standard deviations of the
    // per-seed ANEES estimates and quantify Monte-Carlo uncertainty at
    // the reported N = M_TRIALS × #seeds effective sample size.
    writeln!(
        csv,
        "gamma,n_total,skipped_total,anees_full_gn_mean,anees_full_gn_std,\
         anees_full_h_mean,anees_full_h_std,band_full_lo,band_full_hi,\
         anees_last_gn_mean,anees_last_gn_std,anees_last_h_mean,anees_last_h_std,\
         band_last_lo,band_last_hi,m_trials,n_seeds"
    )
    .unwrap();

    for (k, &gamma) in GAMMA_SWEEP.iter().enumerate() {
        let mut fg = Vec::with_capacity(SEEDS.len());
        let mut fh = Vec::with_capacity(SEEDS.len());
        let mut lg = Vec::with_capacity(SEEDS.len());
        let mut lh = Vec::with_capacity(SEEDS.len());
        let mut n_total = 0usize;
        let mut skipped_total = 0usize;
        for (j, &seed) in SEEDS.iter().enumerate() {
            // Seed streams: distinct per (gamma, seed) pair so no two
            // sweep points share a Monte-Carlo trajectory.
            let stream = 1 + (k as u64) * 100 + seed + (j as u64);
            let row = run_sweep(gamma, stream);
            fg.push(row.full_gn);
            fh.push(row.full_h);
            lg.push(row.last_gn);
            lh.push(row.last_h);
            n_total += row.n;
            skipped_total += row.skipped;
        }
        let (fg_m, fg_s) = mean_std(&fg);
        let (fh_m, fh_s) = mean_std(&fh);
        let (lg_m, lg_s) = mean_std(&lg);
        let (lh_m, lh_s) = mean_std(&lh);

        println!(
            "{:>5.2} {:>13.2} ± {:>5.2} {:>13.2} ± {:>5.2} {:>11.3} ± {:>5.3} {:>11.3} ± {:>5.3} {:>5}",
            gamma, fg_m, fg_s, fh_m, fh_s, lg_m, lg_s, lh_m, lh_s, skipped_total
        );

        writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            gamma,
            n_total,
            skipped_total,
            fg_m,
            fg_s,
            fh_m,
            fh_s,
            DIM as f64 - band_full,
            DIM as f64 + band_full,
            lg_m,
            lg_s,
            lh_m,
            lh_s,
            6.0 - band_last,
            6.0 + band_last,
            M_TRIALS,
            SEEDS.len()
        )
        .unwrap();
    }

    println!(
        "\nper-seed 95% bands: full36 [{:.1}, {:.1}]   last6 [{:.2}, {:.2}]",
        DIM as f64 - band_full,
        DIM as f64 + band_full,
        6.0 - band_last,
        6.0 + band_last
    );
    println!("Wrote {}", CSV_PATH);
}
