//! SE(3) NLL second-order Hessian validation — `SE3_ad_letter` §VI / Table II.
//!
//! Imports the production-basis NLL machinery from `crate::nll_bench` and
//! exercises it under both the **fixed** s-parameterized basis and a
//! **naïve** θ-Taylor basis.  Tests:
//!
//!   1. Four-way Table II Hessian comparison (FD-of-value, FD-of-gradient,
//!      naïve nested AD, fixed nested AD).
//!   2. Schwarz self-consistency of the fixed-basis D2 Hessian.
//!   3. Quadratic-limit (κ→∞) sanity check.
//!   4. §IV.B singular-pair trap at depth 1: naïve unfused D'(θ)·ω/θ → NaN
//!      vs fused `d_prime_omega_over_theta` → finite.
//!
//! See `crate::nll_bench` for the public NLL/Hessian functions used here.

use crate::autodiff::ad_trait::AD;
use crate::autodiff::forward_ad::adfn;
use crate::nll_bench::{
    FixedBasis, NaiveBasis, build_problem, hessian_ad_of_analytical_grad, hessian_d2,
    hessian_fd_analytical_grad, hessian_fd_grad, hessian_fd_value, hessian_for,
};
use crate::so3_adsafe::{Vec3G, d_prime_omega_over_theta, scalar_d_prime_s, theta_sq_from_omega};

// =====================================================================
// Diagnostics.
// =====================================================================

fn frob(m: &[[f64; 6]; 6]) -> f64 {
    let mut s = 0.0;
    for i in 0..6 {
        for j in 0..6 {
            s += m[i][j] * m[i][j];
        }
    }
    s.sqrt()
}
fn frob_diff(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> f64 {
    let mut s = 0.0;
    for i in 0..6 {
        for j in 0..6 {
            let d = a[i][j] - b[i][j];
            s += d * d;
        }
    }
    s.sqrt()
}
fn count_nan(m: &[[f64; 6]; 6]) -> usize {
    let mut n = 0;
    for i in 0..6 {
        for j in 0..6 {
            if !m[i][j].is_finite() {
                n += 1;
            }
        }
    }
    n
}

// =====================================================================
// Tests (all focused on second-order behaviour of the NLL).
// =====================================================================

#[test]
fn nll_hessian_four_ways_table_ii() {
    let p = build_problem(0.05);

    let h_value = hessian_fd_value::<FixedBasis>(&p, 1e-3);
    let h_grad = hessian_fd_grad::<FixedBasis>(&p, 1e-5);
    let (h_fixed, h_fixed_swap) = hessian_d2::<FixedBasis>(&p);
    let (h_naive, _) = hessian_d2::<NaiveBasis>(&p);

    let scale = frob(&h_fixed).max(1.0);

    let max_asym = {
        let mut m = 0.0f64;
        for i in 0..6 {
            for j in 0..6 {
                m = m.max((h_fixed[i][j] - h_fixed[j][i]).abs());
            }
        }
        m
    };
    let max_seed_swap = {
        let mut m = 0.0f64;
        for i in 0..6 {
            for j in 0..6 {
                m = m.max((h_fixed[i][j] - h_fixed_swap[i][j]).abs());
            }
        }
        m
    };
    let err_value = frob_diff(&h_fixed, &h_value) / scale;
    let err_grad = frob_diff(&h_fixed, &h_grad) / scale;

    let naive_nan = count_nan(&h_naive);
    let naive_dev = frob_diff(&h_fixed, &h_naive) / scale;

    eprintln!("\n─── SE3_ad_letter Table II: NLL Hessian computed four ways ───");
    eprintln!(
        "  ‖H_fixed‖_F                                    = {:.4e}",
        frob(&h_fixed)
    );
    eprintln!(
        "  (1) FD of value         rel ‖·−H_fixed‖_F      = {:.2e}   (h=1e-3, expected ~1e-2)",
        err_value
    );
    eprintln!(
        "  (2) FD of AD-gradient   rel ‖·−H_fixed‖_F      = {:.2e}   (h=1e-5, expected ~1e-4)",
        err_grad
    );
    eprintln!(
        "  (3) Naïve D2<6>         NaN entries: {}, rel dev from fixed = {:.2e}",
        naive_nan, naive_dev
    );
    eprintln!(
        "  (4) Fixed D2<6>         self-symmetric to {:.2e}, seed-swap {:.2e}",
        max_asym, max_seed_swap
    );
    eprintln!(
        "      (naïve agrees with fixed here: chain rule kills A″(0)/D″(0) terms\n\
        \x20      since s′(δ=0) = 2 ω · ∂ω/∂δ = 0; the §IV.A trap is observed only\n\
        \x20      in derivative-tensor evaluations that seed s directly.)\n"
    );

    assert!(
        count_nan(&h_fixed) == 0,
        "fixed-basis Hessian has non-finite entries"
    );
    assert!(
        max_asym < 1e-10 * scale,
        "fixed-basis Hessian not symmetric: {:.2e}",
        max_asym
    );
    assert!(
        max_seed_swap < 1e-12 * scale,
        "seed-swap inconsistency: {:.2e}",
        max_seed_swap
    );
    assert!(
        err_grad < 5e-3,
        "fixed disagrees with FD of gradient: {:.2e}",
        err_grad
    );
    assert!(
        err_value < 5e-2,
        "fixed disagrees with FD of value: {:.2e}",
        err_value
    );
    assert!(
        naive_nan == 0,
        "naïve basis produced NaN entries at δ = 0: {}",
        naive_nan
    );
}

#[test]
fn nll_hessian_fixed_basis_self_consistency() {
    let p = build_problem(0.05);
    let (h_qr, h_rq) = hessian_d2::<FixedBasis>(&p);
    let scale = frob(&h_qr).max(1.0);
    let mut max_dev = 0.0f64;
    for i in 0..6 {
        for j in 0..6 {
            max_dev = max_dev.max((h_qr[i][j] - h_rq[i][j]).abs());
        }
    }
    eprintln!(
        "  Fixed D2<6> Schwarz self-consistency: max |∂²/∂q∂r − ∂²/∂r∂q| = {:.2e} (scale {:.2e})",
        max_dev, scale
    );
    assert!(max_dev < 1e-12 * scale, "Schwarz violated: {:.2e}", max_dev);
}

#[test]
fn nll_hessian_analytical_grad_routes_match_d2() {
    // Closed-form analytical gradient — wrapped in either FD or single AD —
    // must reproduce the nested-D2 fixed-basis Hessian to within FD step error
    // (FD path) and to ~1e-12 (single-AD path, exact construction).
    let p = build_problem(0.05);
    let (h_fixed, _) = hessian_d2::<FixedBasis>(&p);
    let h_ad_anal = hessian_ad_of_analytical_grad(&p);
    let h_fd_anal = hessian_fd_analytical_grad(&p, 1e-5);
    let scale = frob(&h_fixed).max(1.0);
    let err_ad = frob_diff(&h_fixed, &h_ad_anal) / scale;
    let err_fd = frob_diff(&h_fixed, &h_fd_anal) / scale;
    eprintln!(
        "  AD-of-analytical-grad (1 adfn6 eval):       rel ‖·−H_d2‖_F = {:.2e}",
        err_ad
    );
    eprintln!(
        "  FD-of-analytical-grad (12 f64 grad evals):  rel ‖·−H_d2‖_F = {:.2e}",
        err_fd
    );
    assert!(
        count_nan(&h_ad_anal) == 0 && count_nan(&h_fd_anal) == 0,
        "analytical-gradient routes produced NaN entries"
    );
    assert!(
        err_ad < 1e-10,
        "AD-of-analytical-grad should match D2 to ~1e-12: {:.2e}",
        err_ad
    );
    assert!(
        err_fd < 5e-3,
        "FD-of-analytical-grad disagrees with D2: {:.2e}",
        err_fd
    );
}

#[test]
fn nll_hessian_auto_for_matches_d2() {
    // Automatic forward-over-reverse via `adr_n6` (reverse tape with adfn<6>
    // partials) — no analytical gradient written by hand.  Must match the
    // nested-D2 oracle to ~machine precision.
    let p = build_problem(0.05);
    let (h_d2, _) = hessian_d2::<FixedBasis>(&p);
    let h_for = hessian_for::<FixedBasis>(&p);
    let scale = frob(&h_d2).max(1.0);
    let err = frob_diff(&h_d2, &h_for) / scale;
    let asym = {
        let mut m = 0.0f64;
        for i in 0..6 {
            for j in 0..6 {
                m = m.max((h_for[i][j] - h_for[j][i]).abs());
            }
        }
        m
    };
    eprintln!(
        "  Auto-FoR (1 fwd + 1 rev pass):  rel ‖·−H_d2‖_F = {:.2e}, |H − Hᵀ|_∞ = {:.2e}",
        err, asym
    );
    assert!(count_nan(&h_for) == 0, "auto-FoR Hessian has NaN entries");
    assert!(
        err < 1e-10,
        "auto-FoR should match D2 oracle to ~1e-12: {:.2e}",
        err
    );
    assert!(asym < 1e-10 * scale, "auto-FoR not symmetric: {:.2e}", asym);
}

#[test]
fn nll_hessian_quadratic_limit_matches_fd() {
    let p = build_problem(1.0e6);
    let (h_fixed, _) = hessian_d2::<FixedBasis>(&p);
    let h_grad = hessian_fd_grad::<FixedBasis>(&p, 1e-5);
    let scale = frob(&h_fixed).max(1.0);
    let err = frob_diff(&h_fixed, &h_grad) / scale;
    eprintln!(
        "  Quadratic-limit (κ=1e6): rel ‖H_fixed − H_fd_grad‖_F = {:.2e}",
        err
    );
    assert!(
        count_nan(&h_fixed) == 0,
        "quadratic-limit fixed Hessian has non-finite entries"
    );
    assert!(err < 5e-3, "quadratic-limit disagreement: {:.2e}", err);
}

// =====================================================================
// Third-order verification: the recipe at AD depth 3.
//
// Tier 1 (in so3_adsafe::tests) checks the scalar β̄(s) atom under D3<1>
// — value, β̄', β̄'', β̄''' all match analytical Taylor coefficients.
//
// This test closes the loop end-to-end: the full SE(3) NLL cubic tensor
// computed two ways must agree.
//   * Oracle: D3<6> nested AD on the cost nll_g (one outer + two inner
//     forward-AD layers; reads `result.tangent[i].tangent[j].tangent[k]`).
//   * Recipe: D2<6> nested AD on the analytical gradient
//     nll_gradient_analytical_g (the path advocated by the paper).  Each
//     g[i] is a D2<6>; reading `g[i].tangent[j].tangent[k]` already gives
//     the third tensor entry (one analytic differentiation + two AD).
//
// Agreement to ~1e-10 relative validates that every step of the
// AD-safe stack (s-parameterization, fused atoms, Q̃_r block, point-
// action seam, prior seam, parameterization factor) carries correct
// third derivatives — i.e. the recipe lives up to its name beyond the
// depth-2 Hessian benchmark.
// =====================================================================

#[test]
fn nll_third_tensor_d3_oracle_vs_d2_of_analytical_grad() {
    use crate::autodiff::nested_ad::Dual;
    use crate::nll_bench::{lift_problem, nll_g, nll_gradient_analytical_g};
    use crate::se3_adsafe::Vec6G;

    type D3<const N: usize> = Dual<Dual<Dual<f64, N>, N>, N>;

    let p = build_problem(0.05);

    // ─── Oracle: D3<6> nested AD on the cost ─────────────────────────────
    let (base, prior, sig, lms, k2) = lift_problem::<D3<6>>(&p);
    let delta_d3: Vec6G<D3<6>> = std::array::from_fn(|i| {
        let l1 = Dual::<f64, 6>::seed(0.0, i);
        let l2 = Dual::<Dual<f64, 6>, 6>::seed(l1, i);
        Dual::<Dual<Dual<f64, 6>, 6>, 6>::seed(l2, i)
    });
    let result_d3 = nll_g::<FixedBasis, D3<6>>(&delta_d3, &base, &prior, &sig, &lms, k2);

    let mut tensor_oracle = [[[0.0f64; 6]; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                tensor_oracle[i][j][k] = result_d3.tangent[i].tangent[j].tangent[k];
            }
        }
    }

    // ─── Recipe: D2<6> nested AD on the analytical gradient ──────────────
    type D2<const N: usize> = Dual<Dual<f64, N>, N>;
    let delta_d2: Vec6G<D2<6>> = std::array::from_fn(|i| {
        let l1 = Dual::<f64, 6>::seed(0.0, i);
        Dual::<Dual<f64, 6>, 6>::seed(l1, i)
    });
    let g = nll_gradient_analytical_g::<D2<6>>(&p, &delta_d2);

    let mut tensor_recipe = [[[0.0f64; 6]; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                // g[i] = ∂cost/∂δ_i (D2<6>); g[i].tangent[j].tangent[k] is
                // the (i, j, k) entry of the third tensor.
                tensor_recipe[i][j][k] = g[i].tangent[j].tangent[k];
            }
        }
    }

    // ─── Compare ─────────────────────────────────────────────────────────
    let mut max_diff = 0.0f64;
    let mut frob_oracle_sq = 0.0f64;
    let mut nan_count = 0usize;
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                let o = tensor_oracle[i][j][k];
                let r = tensor_recipe[i][j][k];
                if !o.is_finite() || !r.is_finite() {
                    nan_count += 1;
                }
                max_diff = max_diff.max((o - r).abs());
                frob_oracle_sq += o * o;
            }
        }
    }
    let scale = frob_oracle_sq.sqrt().max(1.0);
    let rel = max_diff / scale;
    eprintln!(
        "─── NLL 3rd tensor: D3<6> oracle vs D2<6> on analytical gradient ───\n  \
         max |D3 − D2-of-grad| = {:.2e},  ‖oracle‖_F = {:.2e},  rel = {:.2e}",
        max_diff, scale, rel
    );
    assert_eq!(
        nan_count, 0,
        "third tensor has {nan_count} non-finite entries"
    );
    assert!(rel < 1e-10, "Third-order NLL disagreement: rel = {rel:.2e}");
}

// =====================================================================
// Table II rows 5–6: §IV.B singular-pair trap at depth 1.
// =====================================================================

fn seed_omega_zero_adfn3() -> (adfn<3>, adfn<3>, Vec3G<adfn<3>>) {
    let omega: Vec3G<adfn<3>> = std::array::from_fn(|i| {
        let mut t = [0.0; 3];
        t[i] = 1.0;
        adfn::new(0.0, t)
    });
    let (s, theta) = theta_sq_from_omega(&omega);
    (s, theta, omega)
}

#[test]
fn single_ad_naive_unfused_d_prime_over_theta_nans_at_omega_zero() {
    let (s, theta, omega) = seed_omega_zero_adfn3();
    let r = scalar_d_prime_s(s, theta) * omega[0] / theta;
    let v = r.to_constant();
    let t = r.tangent();

    eprintln!(
        "─── Table II row 5: single forward AD, naïve unfused D'(θ)·ω_m/θ at ω=0 ───\n  \
         value = {}, tangent = [{}, {}, {}]",
        v, t[0], t[1], t[2]
    );
    let any_non_finite = !v.is_finite() || t.iter().any(|x| !x.is_finite());
    assert!(
        any_non_finite,
        "naïve unfused expression unexpectedly finite at ω=0; the §IV.B \
         singular-pair trap should fire even at depth 1"
    );
}

#[test]
fn single_ad_fused_d_prime_omega_over_theta_finite_at_omega_zero() {
    let (s, theta, omega) = seed_omega_zero_adfn3();
    let r = d_prime_omega_over_theta(s, theta, omega[0]);
    let v = r.to_constant();
    let t = r.tangent();

    eprintln!(
        "─── Table II row 6: single forward AD, fused D'(θ)·ω_m/θ at ω=0 ───\n  \
         value = {:.3e}, tangent = [{:.6e}, {:.6e}, {:.6e}] (expected [1/360, 0, 0])",
        v, t[0], t[1], t[2]
    );
    assert!(
        v.is_finite() && t.iter().all(|x| x.is_finite()),
        "fused result must be finite"
    );
    // Limit: D'·ω_m/θ → ω_m/360, so value = 0 at ω = 0 and ∂/∂ω_m = 1/360.
    assert!(v.abs() < 1e-15, "value at ω=0 should be 0: {}", v);
    assert!(
        (t[0] - 1.0 / 360.0).abs() < 1e-15,
        "∂/∂ω_0 should be 1/360: {}",
        t[0]
    );
    assert!(t[1].abs() < 1e-15 && t[2].abs() < 1e-15);
}

// =====================================================================
// Auto-verified Table I in rust/README.md.
//
// Builds the markdown table from live accuracy measurements (deterministic
// to the rounding precision below) plus hardcoded LOC counts and asserts
// it matches the README block between the HESSIAN_TABLE_{START,END}
// markers.  Setting `UPDATE_README=1` regenerates the README in place
// instead of failing.
//
// Note: the LOC values are constants here, not auto-detected from source.
// If you refactor a function and its line count changes, update both the
// source and the constant in `build_table_i_markdown` below.  CI catches
// stale LOC the same way it catches stale accuracy: the test fails until
// they agree.
// =====================================================================

struct Timings {
    value: f64,
    grad: f64,
    fd_anal: f64,
    naive: f64,
    d2: f64,
    ad_anal: f64,
    auto_for: f64,
}

fn measure_us<R, F: FnMut() -> R>(mut f: F) -> f64 {
    use std::hint::black_box;
    use std::time::Instant;

    for _ in 0..20 {
        black_box(f());
    }
    let inner = 100u32;
    let outer = 20;
    let mut min_per_call = f64::INFINITY;
    for _ in 0..outer {
        let start = Instant::now();
        for _ in 0..inner {
            black_box(f());
        }
        let per_call = start.elapsed().as_secs_f64() * 1e6 / inner as f64;
        if per_call < min_per_call {
            min_per_call = per_call;
        }
    }
    min_per_call
}

fn format_us(us: f64) -> String {
    if us >= 1000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else if us >= 100.0 {
        format!("{:.0} μs", us)
    } else if us >= 10.0 {
        format!("{:.1} μs", us)
    } else {
        format!("{:.2} μs", us)
    }
}

fn measure_timings() -> Timings {
    let p = build_problem(0.05);
    Timings {
        value: measure_us(|| hessian_fd_value::<FixedBasis>(&p, 1e-3)),
        grad: measure_us(|| hessian_fd_grad::<FixedBasis>(&p, 1e-5)),
        fd_anal: measure_us(|| hessian_fd_analytical_grad(&p, 1e-5)),
        naive: measure_us(|| hessian_d2::<NaiveBasis>(&p)),
        d2: measure_us(|| hessian_d2::<FixedBasis>(&p)),
        ad_anal: measure_us(|| hessian_ad_of_analytical_grad(&p)),
        auto_for: measure_us(|| hessian_for::<FixedBasis>(&p)),
    }
}

fn build_table_i_markdown(timings: Option<&Timings>) -> String {
    let p = build_problem(0.05);
    let (h_d2, _) = hessian_d2::<FixedBasis>(&p);
    let scale = frob(&h_d2).max(1.0);

    let h_value = hessian_fd_value::<FixedBasis>(&p, 1e-3);
    let h_grad = hessian_fd_grad::<FixedBasis>(&p, 1e-5);
    let h_fd_anal = hessian_fd_analytical_grad(&p, 1e-5);
    let h_ad_anal = hessian_ad_of_analytical_grad(&p);
    let h_for = hessian_for::<FixedBasis>(&p);
    let (h_naive, _) = hessian_d2::<NaiveBasis>(&p);

    let err_value = frob_diff(&h_d2, &h_value) / scale;
    let err_grad = frob_diff(&h_d2, &h_grad) / scale;
    let err_naive = frob_diff(&h_d2, &h_naive) / scale;
    let err_fd_anal = frob_diff(&h_d2, &h_fd_anal) / scale;
    let err_ad_anal = frob_diff(&h_d2, &h_ad_anal) / scale;
    let err_for = frob_diff(&h_d2, &h_for) / scale;

    let naive_str = if err_naive == 0.0 {
        "0 (chain-rule cancels at δ=0)".to_string()
    } else {
        format!("{:.2e}", err_naive)
    };

    let fmt_t = |get: fn(&Timings) -> f64| -> String {
        timings.map_or_else(|| "—".to_string(), |t| format_us(get(t)))
    };
    let t_value = fmt_t(|t| t.value);
    let t_grad = fmt_t(|t| t.grad);
    let t_fd_anal = fmt_t(|t| t.fd_anal);
    let t_naive = fmt_t(|t| t.naive);
    let t_d2 = fmt_t(|t| t.d2);
    let t_ad_anal = fmt_t(|t| t.ad_anal);
    let t_auto_for = fmt_t(|t| t.auto_for);

    format!(
        "\n\
| # | Method | LOC | Rel. err. vs. oracle | Time |\n\
|---|---|---|---|---|\n\
| 1 | FD of value (no AD) | 43 | {:.2e} | {} |\n\
| 2 | FD of AD-gradient (baseline) | 30 | {:.2e} | {} |\n\
| 6 | FD of analytical gradient (fused basis) | 104 | {:.2e} | {} |\n\
| 3 | Nested AD, naïve basis | 118 | {} | {} |\n\
| 4 | Nested AD, fused basis (oracle) | 30 | 0 (reference) | {} |\n\
| 5 | Seeded AD of analytical gradient, naïve basis | (102) | NaN (depth-0 §IV.B trap) | — |\n\
| 7 | Seeded AD of analytical gradient, fused basis (recipe) | 102 | {:.2e} | {} |\n\
| 8 | Auto FoR (`UnsafeCell` tape, no analytical grad) | 30 | {:.2e} | {} |\n\
",
        err_value,
        t_value,
        err_grad,
        t_grad,
        err_fd_anal,
        t_fd_anal,
        naive_str,
        t_naive,
        t_d2,
        err_ad_anal,
        t_ad_anal,
        err_for,
        t_auto_for,
    )
}

// Strip the trailing data column from each markdown-table row in `block`.
// Used to compare the README table without the (machine-dependent) Time column.
fn strip_last_column(block: &str) -> String {
    block
        .split('\n')
        .map(|line| {
            let trimmed = line.trim();
            if !trimmed.starts_with('|') || !trimmed.ends_with('|') {
                return line.to_string();
            }
            let mut parts: Vec<&str> = line.split('|').collect();
            if parts.len() >= 3 {
                parts.remove(parts.len() - 2);
            }
            parts.join("|")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn readme_table_i_is_up_to_date() {
    const START: &str = "<!-- HESSIAN_TABLE_START -->";
    const END: &str = "<!-- HESSIAN_TABLE_END -->";

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let readme_path = std::path::Path::new(manifest_dir).join("README.md");
    let content = std::fs::read_to_string(&readme_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", readme_path.display()));

    let start_idx = content
        .find(START)
        .unwrap_or_else(|| panic!("missing `{}` marker in README.md", START));
    let end_idx = content
        .find(END)
        .unwrap_or_else(|| panic!("missing `{}` marker in README.md", END));

    let inner_start = start_idx + START.len();
    let current_block = &content[inner_start..end_idx];

    if std::env::var("UPDATE_README").is_ok() {
        let timings = measure_timings();
        let expected_block = build_table_i_markdown(Some(&timings));
        let mut new_content = String::with_capacity(content.len() + expected_block.len());
        new_content.push_str(&content[..inner_start]);
        new_content.push_str(&expected_block);
        new_content.push_str(&content[end_idx..]);
        std::fs::write(&readme_path, &new_content)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", readme_path.display()));
        eprintln!("README.md Table I regenerated (with fresh Time column).");
        return;
    }

    // Routine `cargo test`: compare everything except the (noisy) Time column.
    // CI runs this without UPDATE_README, so timing values must not gate the
    // test — only LOC and accuracy do.
    let expected_block = build_table_i_markdown(None);
    let current_stripped = strip_last_column(current_block);
    let expected_stripped = strip_last_column(&expected_block);

    if current_stripped == expected_stripped {
        return;
    }

    panic!(
        "README.md Table I is out of date (LOC or accuracy column mismatch; \
         the Time column is ignored here).\n\
         --- expected (Time column stripped) ---\n{expected}\
         --- found (Time column stripped) ---\n{found}\
         --- end ---\n\
         Run `UPDATE_README=1 cargo test --lib readme_table_i_is_up_to_date` to regenerate.",
        expected = expected_stripped,
        found = current_stripped,
    );
}
