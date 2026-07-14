//! NEES covariance-consistency study: Gauss–Newton/FIM information vs the
//! exact observed information (nested-dual Hessian, `hessian_d2`) on a
//! robust SE(3) pose-with-prior problem.
//!
//! Monte-Carlo protocol (mirrors `nees_consistency.py`, the JAX prototype;
//! kept out of the public tree — see `.gitignore`):
//!   1. Sample T_true = T_prior · Exp(ξ),  ξ ~ N(0, Σ_prior).
//!   2. Generate landmark observations with Gaussian inlier noise σ_z and a
//!      fraction ε of gross uniform outliers.
//!   3. Solve the MAP problem by damped robust-GN with re-basing, so the
//!      converged linearization point has δ = 0 (the Algorithm-1 setting).
//!   4. Evaluate two information matrices at the MAP:
//!        I_GN = Σ_i 2ρ'(s_i)·J_iᵀJ_i + J_pᵀ Σ_p⁻¹ J_p    (what solvers report)
//!        H    = exact NLL Hessian via nested duals (`hessian_d2`)
//!   5. NEES = ξ_errᵀ · I · ξ_err with ξ_err = Log(T̂⁻¹ T_true); average over
//!      trials → ANEES, compared against the χ²₆ 95% consistency band.
//!
//! Run:  cargo run --release --features bench-support --example nees_consistency

// Same rationale as the crate-level allow in lib.rs: numerical / matrix code
// uses index-based loops pervasively.  The doc allow keeps the aligned
// formula layout in the protocol list above.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_overindented_list_items)]

use se3_ad_recipes::nll_bench::{FixedBasis, Problem, hessian_d2, nll_gradient};
use se3_ad_recipes::projective::{j_cross, project, project_jacobian, transform_point};
use se3_ad_recipes::se3_adsafe::{adjoint_g, se3_jr_inv_g};
use se3_ad_recipes::se3_unsafe::{Pose, right_update};
use se3_ad_recipes::test_support::{Rng, mean_std};
use se3_ad_recipes::{Mat6, Vec6, mm, mv, norm, transpose};

// ─── Experiment constants ───────────────────────────────────────────────

const N_LM: usize = 12;
const SIG_Z: f64 = 0.003; // inlier noise, normalized image coords
const SIG_PRIOR: [f64; 6] = [0.05, 0.05, 0.05, 0.10, 0.10, 0.10];
const KAPPA: f64 = 3.0; // pseudo-Huber threshold, whitened units
const XI_PRIOR_MEAN: Vec6 = [0.30, -0.20, 0.40, 0.50, -0.30, 0.70];
const OUTLIER_HALF_WIDTH: f64 = 0.15; // uniform gross-outlier offset
const M_TRIALS: usize = 1000;
const EPS_SWEEP: [f64; 4] = [0.0, 0.1, 0.2, 0.3];
/// Three independent RNG seeds per sweep point; final ANEES is the mean
/// across seeds, and the between-seed standard deviation quantifies the
/// Monte-Carlo uncertainty of the reported number.
const SEEDS: [u64; 3] = [1, 2, 3];
/// CSV output path, anchored to the crate root at compile time — `cargo
/// run` inherits the invoker's cwd, so a raw relative path would scatter
/// output wherever the example happened to be launched from.
const CSV_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../experiments/data/nees.csv");

// ─── 6×6 Cholesky solve with positive-definiteness detection ────────────

/// Lower-triangular Cholesky factor; `None` if the matrix is not PD.
fn chol6(a: &Mat6) -> Option<Mat6> {
    let mut l = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let d = a[i][i] - sum;
                // Non-finite pivots rejected too (`NaN <= 0.0` is false),
                // matching linalg::cholesky_n.
                if d <= 0.0 || !d.is_finite() {
                    return None;
                }
                l[i][i] = d.sqrt();
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve (L Lᵀ) x = b given the Cholesky factor L.
fn chol6_solve(l: &Mat6, b: &Vec6) -> Vec6 {
    let mut y = [0.0f64; 6];
    for i in 0..6 {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = [0.0f64; 6];
    for i in (0..6).rev() {
        let mut s = y[i];
        for k in (i + 1)..6 {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    x
}

// ─── Problem sampling ────────────────────────────────────────────────────

/// Fixed landmark field, drawn once (deterministic).
fn make_landmarks(rng: &mut Rng) -> Vec<[f64; 3]> {
    (0..N_LM)
        .map(|_| {
            [
                rng.uniform_in(-1.2, 1.2),
                rng.uniform_in(-0.9, 0.9),
                rng.uniform_in(2.5, 7.5),
            ]
        })
        .collect()
}

/// One Monte-Carlo instance: truth sampled from the prior, contaminated
/// observations, and a `Problem` initialized at the prior mean.
fn sample_trial(rng: &mut Rng, lms: &[[f64; 3]], eps: f64, kappa: f64) -> (Pose, Problem) {
    let t_prior = Pose::exp(&XI_PRIOR_MEAN);
    let xi_true: Vec6 = std::array::from_fn(|i| SIG_PRIOR[i] * rng.normal());
    let t_true = t_prior.compose(&Pose::exp(&xi_true));

    let l_white = [[1.0 / SIG_Z, 0.0], [0.0, 1.0 / SIG_Z]];
    let landmarks = lms
        .iter()
        .map(|x| {
            let pi = project(&t_true.act(x));
            let mut z = [pi[0] + SIG_Z * rng.normal(), pi[1] + SIG_Z * rng.normal()];
            if rng.uniform() < eps {
                z[0] += rng.uniform_in(-OUTLIER_HALF_WIDTH, OUTLIER_HALF_WIDTH);
                z[1] += rng.uniform_in(-OUTLIER_HALF_WIDTH, OUTLIER_HALF_WIDTH);
            }
            (*x, z, l_white)
        })
        .collect();

    let sigma_inv_sq_diag: [f64; 6] = std::array::from_fn(|i| 1.0 / (SIG_PRIOR[i] * SIG_PRIOR[i]));
    let problem = Problem {
        base: t_prior, // initial guess = prior mean; re-based during solve
        t_prior,
        sigma_inv_sq_diag,
        landmarks,
        kappa,
    };
    (t_true, problem)
}

// ─── Robust Gauss–Newton information (the covariance solvers report) ────

/// I_GN = Σ_i 2ρ'(s_i)·J_iᵀJ_i + J_pᵀ Σ_p⁻¹ J_p, assembled at
/// the current `p.base` with δ = 0.
///
/// Convention check: the crate's pseudo-Huber ρ(s) = κ²(√(1+s/κ²) − 1) has
/// ρ'(0) = ½, so the Gaussian limit of the data Hessian is JᵀJ (whitened),
/// i.e. weight w = 2ρ'(s) = 1/√(1+s/κ²) applied to JᵀJ — the standard
/// robust-GN (Triggs) weighting.
fn gn_information(p: &Problem) -> Mat6 {
    let r = &p.base.rot;
    let k2 = p.kappa * p.kappa;
    let mut info = [[0.0f64; 6]; 6];

    // Data blocks: J = L · Jπ(x') · R · J×(x)  (2×6, whitened).
    for (x, z, l) in p.landmarks.iter() {
        let xp = transform_point(r, &p.base.trans, x);
        let jpi = project_jacobian(&xp); // 2×3
        let jx = j_cross(x); // 3×6, action Jacobian is R·J×
        // a = Jπ · R  (2×3)
        let mut a = [[0.0f64; 3]; 2];
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..3 {
                    a[i][j] += jpi[i][k] * r[k][j];
                }
            }
        }
        // jw = L · a · J×  (2×6)
        let mut jr = [[0.0f64; 6]; 2];
        for i in 0..2 {
            for j in 0..6 {
                let mut aj = 0.0;
                for k in 0..3 {
                    aj += a[i][k] * jx[k][j];
                }
                jr[i][j] = aj; // pre-whitening
            }
        }
        let mut jw = [[0.0f64; 6]; 2];
        for i in 0..2 {
            for j in 0..6 {
                jw[i][j] = l[i][0] * jr[0][j] + l[i][1] * jr[1][j];
            }
        }
        // whitened residual and robust weight
        let pi = project(&xp);
        let raw = [pi[0] - z[0], pi[1] - z[1]];
        let rw = [
            l[0][0] * raw[0] + l[0][1] * raw[1],
            l[1][0] * raw[0] + l[1][1] * raw[1],
        ];
        let s = rw[0] * rw[0] + rw[1] * rw[1];
        let w = 1.0 / (1.0 + s / k2).sqrt(); // = 2ρ'(s)
        for i in 0..6 {
            for j in 0..6 {
                info[i][j] += w * (jw[0][i] * jw[0][j] + jw[1][i] * jw[1][j]);
            }
        }
    }

    // Prior block: J_p = −(J_r^{SE(3)}(ξ))⁻¹ · Ad_{G⁻¹},  G = T⁻¹T_prior.
    let g = p.base.relative(&p.t_prior);
    let xi = g.log();
    let jr_inv = se3_jr_inv_g::<f64>(&xi);
    let gi = g.inverse();
    let ad = adjoint_g::<f64>(&gi.rot, &gi.trans);
    let mut jp = mm(&jr_inv, &ad);
    for row in jp.iter_mut() {
        for v in row.iter_mut() {
            *v = -*v;
        }
    }
    // info += J_pᵀ · diag(Σ_p⁻¹) · J_p
    let jpt = transpose(&jp);
    for i in 0..6 {
        for j in 0..6 {
            let mut s = 0.0;
            for k in 0..6 {
                s += jpt[i][k] * p.sigma_inv_sq_diag[k] * jp[k][j];
            }
            info[i][j] += s;
        }
    }
    info
}

// ─── MAP solve: damped robust-GN with re-basing ─────────────────────────

/// Iterates T̄ ← T̄·Exp(step) so the converged problem has δ = 0 at `p.base`.
fn solve_map(p: &mut Problem, iters: usize) {
    let lambda = 1e-6;
    for _ in 0..iters {
        let g = nll_gradient::<FixedBasis>(p, &[0.0; 6]);
        if norm(&g) < 1e-10 {
            break;
        }
        let mut h = gn_information(p);
        for i in 0..6 {
            h[i][i] += lambda;
        }
        let Some(l) = chol6(&h) else { break };
        let neg_g: Vec6 = std::array::from_fn(|i| -g[i]);
        let step = chol6_solve(&l, &neg_g);
        p.base = right_update(&p.base, &step);
    }
}

// ─── Monte Carlo ─────────────────────────────────────────────────────────

struct SweepRow {
    n: usize,
    skipped: usize,
    anees_gn: f64,
    anees_h: f64,
}

fn run_sweep(eps: f64, lms: &[[f64; 3]], seed: u64, kappa: f64) -> SweepRow {
    let mut rng = Rng::new(seed);
    let (mut sum_gn, mut sum_h, mut n, mut skipped) = (0.0, 0.0, 0usize, 0usize);

    for _ in 0..M_TRIALS {
        let (t_true, mut p) = sample_trial(&mut rng, lms, eps, kappa);
        solve_map(&mut p, 30);

        // Error in the right tangent at the estimate.
        let xi_err = p.base.relative(&t_true).log();

        // Exact observed information via nested duals at δ = 0.
        let (h_exact, _) = hessian_d2::<FixedBasis>(&p);
        // Guard: the exact Hessian can be indefinite under extreme outliers.
        if chol6(&h_exact).is_none() {
            skipped += 1;
            continue;
        }
        let i_gn = gn_information(&p);

        sum_gn += se3_ad_recipes::dot(&xi_err, &mv(&i_gn, &xi_err));
        sum_h += se3_ad_recipes::dot(&xi_err, &mv(&h_exact, &xi_err));
        n += 1;
    }
    assert!(
        n > 0,
        "all {M_TRIALS} trials skipped at eps={eps} — cannot form an ANEES mean"
    );
    SweepRow {
        n,
        skipped,
        anees_gn: sum_gn / n as f64,
        anees_h: sum_h / n as f64,
    }
}

fn main() {
    use std::io::Write;

    let mut lm_rng = Rng::new(0xC0FFEE);
    let lms = make_landmarks(&mut lm_rng);

    // The band is a property of the per-seed sample size, not of the
    // seed-averaged mean; report it as a diagnostic on individual sweeps.
    let per_seed_band = 1.96 * (2.0 / (6.0 * M_TRIALS as f64)).sqrt() * 6.0;

    println!(
        "NEES consistency: {} landmarks, sigma_z={}, kappa={}, \
         M={} trials/seed, {} seeds/point",
        N_LM,
        SIG_Z,
        KAPPA,
        M_TRIALS,
        SEEDS.len()
    );
    println!(
        "{:>5} {:>18} {:>18} {:>7} {:>18}",
        "eps", "ANEES(GN) mean±std", "ANEES(H) mean±std", "skip", "per-seed 95% band"
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
        "eps,n_total,skipped_total,anees_gn_mean,anees_gn_std,\
         anees_h_mean,anees_h_std,band_lo,band_hi,m_trials,n_seeds"
    )
    .unwrap();

    for (k, &eps) in EPS_SWEEP.iter().enumerate() {
        let mut gn_vals = Vec::with_capacity(SEEDS.len());
        let mut h_vals = Vec::with_capacity(SEEDS.len());
        let mut n_total = 0usize;
        let mut skipped_total = 0usize;
        for &seed in SEEDS.iter() {
            // Seed streams: unique per (epsilon-index k, seed) with a k-stride
            // (1000) far larger than any seed, so the streams stay distinct
            // regardless of how SEEDS is ordered — no silent collision.
            let stream = 1 + (k as u64) * 1000 + seed;
            let row = run_sweep(eps, &lms, stream, KAPPA);
            gn_vals.push(row.anees_gn);
            h_vals.push(row.anees_h);
            n_total += row.n;
            skipped_total += row.skipped;
        }
        let (gn_m, gn_s) = mean_std(&gn_vals);
        let (h_m, h_s) = mean_std(&h_vals);

        println!(
            "{:>5.2} {:>10.3} ± {:>5.3} {:>10.3} ± {:>5.3} {:>7} [{:.2}, {:.2}]",
            eps,
            gn_m,
            gn_s,
            h_m,
            h_s,
            skipped_total,
            6.0 - per_seed_band,
            6.0 + per_seed_band
        );

        writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{},{},{}",
            eps,
            n_total,
            skipped_total,
            gn_m,
            gn_s,
            h_m,
            h_s,
            6.0 - per_seed_band,
            6.0 + per_seed_band,
            M_TRIALS,
            SEEDS.len()
        )
        .unwrap();
    }

    // Gaussian-kernel control at eps = 0: with kappa -> infinity the
    // pseudo-Huber kernel degenerates to the quadratic Gaussian NLL, the
    // robust-weight discount disappears, and a calibrated setup should
    // return ANEES ~= 6. This isolates the mechanism behind the mildly
    // conservative eps = 0 baseline above (E[w] ~= 0.90 discount on the
    // reported information vs a few-percent efficiency loss of the
    // M-estimator on Gaussian inliers). Console-only; not part of the CSV.
    let kappa_control = 1.0e9;
    let mut gn_vals = Vec::with_capacity(SEEDS.len());
    let mut h_vals = Vec::with_capacity(SEEDS.len());
    for &seed in SEEDS.iter() {
        let stream = 90_000 + seed;
        let row = run_sweep(0.0, &lms, stream, kappa_control);
        gn_vals.push(row.anees_gn);
        h_vals.push(row.anees_h);
    }
    let (gn_m, gn_s) = mean_std(&gn_vals);
    let (h_m, h_s) = mean_std(&h_vals);
    println!(
        "\ncontrol (Gaussian kernel, eps=0.00): ANEES(GN) {:.3} \u{00b1} {:.3}   ANEES(H) {:.3} \u{00b1} {:.3}",
        gn_m, gn_s, h_m, h_s
    );

    println!("\nWrote {}", CSV_PATH);
}
