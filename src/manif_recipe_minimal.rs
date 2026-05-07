//! Minimal reproducer: the manif-style θ-threshold recipe under AD.
//!
//! manif (and Sophus, GTSAM's Lie wrappers, etc.) handle the small-angle
//! case of SO(3) / SE(3) scalars with a θ-Taylor branch:
//!
//!     fn scalar_a(theta) = if theta < EPS { 1 - theta²/6 } else { sin(theta)/theta }
//!
//! This is correct to f64 precision and safe under single forward AD on the
//! function value alone. It fails second-order in *two distinct ways* that
//! Table II of the companion paper distinguishes:
//!
//!   §IV.A  Polynomial depletion        — needs nested AD (depth ≥ 2)
//!   §IV.B  Singular-pair NaN trap      — fires at depth 1 with seeded AD
//!
//! The three tests below isolate each failure on its own minimal scalar
//! atom, then cite the existing `nll_hessian_four_ways_table_ii` test as
//! the end-to-end Hessian-of-NLL comparison.
//!
//! Drop this file into `src/` next to `nll_tests.rs`, add `mod manif_recipe_minimal;`
//! to `lib.rs` (under the `#[cfg(test)] mod nll_tests;` line), and `cargo test`.

use crate::autodiff::ad_trait::AD;
use crate::autodiff::forward_ad::adfn;
use crate::autodiff::nested_ad::Dual;
use crate::so3_adsafe::{
    Vec3G, dot3_g, scalar_beta_over_s_g, scalar_d_prime_s, theta_sq_from_omega,
};

// =====================================================================
// The manif-style θ-threshold recipe — exactly the form `so3_unsafe`
// implements, lifted into <T: AD> generics so we can run AD through it.
// =====================================================================

/// manif-equivalent A(θ) = sin(θ)/θ.  Degree-2 Taylor below θ < EPS.
fn manif_style_scalar_a<T: AD>(theta: T) -> T {
    const EPS: f64 = 1e-10;
    if theta.to_constant() < EPS {
        T::constant(1.0) - theta * theta * T::constant(1.0 / 6.0)
    } else {
        theta.sin() / theta
    }
}

/// f(ω) = A(‖ω‖).  This is the scalar function whose nested derivatives we
/// want — it's a stand-in for any rotational atom inside manif's exp/log/Q.
fn manif_style_f<T: AD>(omega: &Vec3G<T>) -> T {
    let s = dot3_g(omega, omega);
    let theta = if s.to_constant() > 1e-20 {
        s.sqrt()
    } else {
        T::constant(0.0)
    };
    manif_style_scalar_a(theta)
}

/// AD-safe replacement: A(θ) as a degree-4 Taylor in s = θ².
fn safe_scalar_a_s<T: AD>(s: T) -> T {
    T::constant(1.0)
        + s * (T::constant(-1.0 / 6.0)
            + s * (T::constant(1.0 / 120.0)
                + s * (T::constant(-1.0 / 5040.0) + s * T::constant(1.0 / 362880.0))))
}

fn safe_f<T: AD>(omega: &Vec3G<T>) -> T {
    let s = dot3_g(omega, omega);
    safe_scalar_a_s(s)
}

// =====================================================================
// TEST 1 — Polynomial depletion (§IV.A).  Needs nested AD; the manif
// branch returns a degree-2 polynomial in θ, and depth-2 nested AD on
// the wrong basis runs out of coefficients.
// =====================================================================

#[test]
fn polynomial_depletion_in_manif_style_taylor_at_small_omega() {
    // Evaluate at a point that is *inside* the small-angle branch but not
    // at the origin: ω = (1e-6, 0, 0) gives θ ≈ 1e-6 < 1e-10? No — 1e-6 is
    // ABOVE the manif threshold of 1e-10, so this hits the closed form,
    // which is fine.  We need ω deep inside the Taylor branch to see the
    // depletion.  Pick ω with θ = 1e-12.
    let omega_val: f64 = 1e-12;

    type D2 = Dual<Dual<f64, 3>, 3>;
    let omega: Vec3G<D2> = std::array::from_fn(|i| {
        let inner = Dual::<f64, 3>::seed(if i == 0 { omega_val } else { 0.0 }, i);
        D2::seed(inner, i)
    });

    // manif-style: degree-2 Taylor, will deplete under depth-2 AD.
    let f_manif = manif_style_f::<D2>(&omega);
    // s-basis safe: degree-4 Taylor in s, survives.
    let f_safe = safe_f::<D2>(&omega);

    // Pull out ∂²f/∂ω₀² (the diagonal Hessian entry along the axis we
    // perturbed).
    let h_manif_00 = f_manif.tangent[0].tangent[0];
    let h_safe_00 = f_safe.tangent[0].tangent[0];

    // Analytical reference: A(θ) = sin(θ)/θ as a function of ω.  At small ω,
    // f(ω) ≈ 1 - ‖ω‖²/6 + ‖ω‖⁴/120 - …, so ∂²f/∂ω₀² = -2/6 + 4·ω₀²/120·… ≈ -1/3
    // at ω = 0.  At ω₀ = 1e-12 the higher-order correction is < 1e-24, so
    // the reference is -1/3 to f64 precision.
    let h_truth = -1.0 / 3.0;

    eprintln!("\n─── §IV.A polynomial depletion (depth-2 nested AD) ───");
    eprintln!(
        "  ω = ({:.0e}, 0, 0), θ ≈ {:.0e} (well below manif EPS=1e-10)",
        omega_val, omega_val
    );
    eprintln!("  Analytical ∂²f/∂ω₀² at ω≈0 = -1/3 ≈ {:.8}", h_truth);
    eprintln!("  manif-style θ-Taylor degree-2:    H[0][0] = {:.8}", h_manif_00);
    eprintln!("  s-basis Taylor degree-4 in s:     H[0][0] = {:.8}", h_safe_00);
    eprintln!(
        "  err manif: {:.2e}    err safe: {:.2e}",
        (h_manif_00 - h_truth).abs(),
        (h_safe_00 - h_truth).abs()
    );

    // The s-basis is depth-2 correct.
    assert!(
        (h_safe_00 - h_truth).abs() < 1e-13,
        "s-basis safe form must reproduce H = -1/3"
    );

    // The manif-style θ-Taylor of degree 2 differentiates to:
    //   d/dθ (1 - θ²/6) = -θ/3
    //   d²/dθ² (1 - θ²/6) = -1/3   ← per-θ, but the chain to ω requires
    //                                 differentiating θ = √(ωᵀω), and
    //                                 ∂²θ/∂ω∂ω is singular at ω = 0.
    // The chain rule via s = ωᵀω gives the right answer if and only if
    // the basis function is expressed as a polynomial in s.  Expressed
    // in θ, the second derivative w.r.t. ω at small ω goes through
    // `sqrt'(s)` whose Taylor truncation collides with the function's
    // own degree-2 truncation — depletion.
    //
    // Document the disagreement (don't necessarily expect a specific
    // wrong value; the FAILURE MODE is "wrong, silently, no NaN").
    assert!(
        (h_manif_00 - h_truth).abs() > 1e-6,
        "manif-style θ-Taylor SHOULD deplete here.  If this assertion \
         fires, the depletion didn't happen — investigate."
    );
}

// =====================================================================
// TEST 2 — §IV.B singular-pair NaN trap.  Fires at depth 1 with seeded
// forward AD; nesting is not required.  This is the same scalar atom
// as Table I rows 5–6 of the companion paper.
// =====================================================================

#[test]
fn singular_pair_nans_at_depth_one_seeded_ad() {
    // The atom: D'(θ) · ω_m / θ.  Appears in derivative tensors of the
    // SE(3) Q block (Proposition 2/3 of the companion paper).  manif
    // would compute it by AD on the closed form — at ω = 0 that hits
    // the literal expression `0 / 0` because both D'(θ) and θ vanish.

    // Seed ω at the origin with depth-1 forward AD.  Three tangent
    // directions, each a unit basis vector.
    let omega: Vec3G<adfn<3>> = std::array::from_fn(|i| {
        let mut t = [0.0f64; 3];
        t[i] = 1.0;
        adfn::new(0.0, t)
    });
    let (s, theta) = theta_sq_from_omega(&omega);

    // The naïve unfused form, exactly as it appears if you write out
    // ∂Q/∂ω by hand and ask AD to evaluate it at ω = 0.
    let unfused = scalar_d_prime_s(s, theta) * omega[0] / theta;

    // The fused replacement: same value mathematically, but expressed
    // as ω₀ · β̄(s) with β̄(s) = 1/360 + s/7560 + … smooth in s.
    let fused = omega[0] * scalar_beta_over_s_g(s, theta);

    eprintln!("\n─── §IV.B singular-pair trap (depth-1 seeded forward AD) ───");
    eprintln!(
        "  Naïve  D'(θ)·ω₀/θ  at ω=0: value = {:?}, tangent = {:?}",
        unfused.to_constant(),
        unfused.tangent()
    );
    eprintln!(
        "  Fused  ω₀ · β̄(s)   at ω=0: value = {:.6e}, tangent = {:?}",
        fused.to_constant(),
        fused.tangent()
    );
    eprintln!("  Expected limit:     value = 0, tangent = [1/360, 0, 0]");

    // Naïve form: not finite (NaN, ±∞, or both, depending on `0/0` order).
    let unfused_finite = unfused.to_constant().is_finite()
        && unfused.tangent().iter().all(|x| x.is_finite());
    assert!(
        !unfused_finite,
        "naïve unfused expression should NOT be finite at ω=0 — \
         this is the §IV.B trap.  If it's finite, manif has gotten \
         lucky on the order of operations."
    );

    // Fused form: finite, value 0, tangent matches the analytical limit 1/360.
    assert!(
        fused.to_constant().is_finite()
            && fused.tangent().iter().all(|x| x.is_finite()),
        "fused form must be finite"
    );
    assert!(fused.to_constant().abs() < 1e-15);
    let t = fused.tangent();
    assert!((t[0] - 1.0 / 360.0).abs() < 1e-15);
    assert!(t[1].abs() < 1e-15 && t[2].abs() < 1e-15);
}

// =====================================================================
// TEST 3 — Pointer to the existing four-way Hessian comparison.
//
// The end-to-end demonstration on the SE(3) NLL is already done in
// `nll_tests::nll_hessian_four_ways_table_ii`.  That test runs the
// manif-equivalent NaiveBasis (θ-Taylor with the same threshold structure)
// against the s-basis FixedBasis on the full Hessian and finds:
//
//   * NaiveBasis agrees with FixedBasis at δ=0 — chain rule masks the
//     depletion when s = ωᵀω is differentiated to ω rather than seeded
//     directly.  This is the subtle "it looks fine in the cost path"
//     observation worth flagging.
//
//   * NaiveBasis would fail (and the unfused derivative-tensor path
//     does fail at depth 1, captured in this file's TEST 2) when seeded
//     in the basis variable s itself, which is what derivative-tensor
//     evaluations like ∂Jr⁻¹/∂ω do.
//
// No additional code needed here — the existing test is the proof.
// =====================================================================

#[test]
fn pointer_to_existing_four_way_table_ii_test() {
    eprintln!(
        "\n─── End-to-end Hessian comparison ───\n  \
         See: cargo test --lib nll_hessian_four_ways_table_ii\n  \
         Rows 3 (NaiveBasis nested) vs 4 (FixedBasis nested) of Table II.\n  \
         Rows 5–6 of Table II are reproduced as TEST 2 in this file."
    );
}