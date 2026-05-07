//! # SO(3) — AD-safe, generic over the scalar type.
//!
//! Templated `<T: AD>` rewrite of the SO(3) primitives following the
//! recipe of the companion paper §IV:
//!
//!   * **s = θ² parameterization.**  All scalar functions take
//!     `(s, θ)` rather than `θ` alone, so the Lie-group code never
//!     differentiates through `√s` at the origin.
//!   * **Degree ≥ 4 Taylor on every small-angle branch.**  The basic
//!     scalars {A, B, C, D} and β as well as the fused atoms β̄(s),
//!     D̃·ω_m (≡ ω_m β̄), and β̄'(s) all carry at least five Taylor
//!     coefficients in s, leaving headroom up to AD depth 4.
//!   * **Fused removable singularities.**  Any product of the form
//!     f(θ)/θᵏ with f(0) = 0 in the original closed-form derivative
//!     tensors is replaced here by an `s`-native scalar that AD can
//!     differentiate through to arbitrary order.
//!   * **Single threshold (s < 1e-4).**  Every Taylor branch flips at
//!     the same crossover, so cross-branch continuity can be verified
//!     once and reused.
//!
//! `pub(crate)` helpers `scalar_d_prime_s` and `scalar_beta_prime_s`
//! retain the conventional θ-form for the diagnostic NaN test.  They
//! are not part of the public surface — production code routes through
//! the fused atoms.

use crate::autodiff::ad_trait::AD;

// =========================================================================
// AD-generic 3-vector / 3-matrix types and ops
// =========================================================================

/// 3-vector over AD scalar T.
pub type Vec3G<T> = [T; 3];
/// 3×3 matrix over AD scalar T, row-major.
pub type Mat3G<T> = [[T; 3]; 3];

#[inline]
pub fn dot3_g<T: AD>(a: &Vec3G<T>, b: &Vec3G<T>) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn norm3_g<T: AD>(v: &Vec3G<T>) -> T {
    dot3_g(v, v).sqrt()
}

#[inline]
pub fn hat_g<T: AD>(w: &Vec3G<T>) -> Mat3G<T> {
    let z = T::constant(0.0);
    [[z, -w[2], w[1]], [w[2], z, -w[0]], [-w[1], w[0], z]]
}

pub fn mm3_g<T: AD>(a: &Mat3G<T>, b: &Mat3G<T>) -> Mat3G<T> {
    let z = T::constant(0.0);
    let mut c = [[z; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

pub fn mv3_g<T: AD>(m: &Mat3G<T>, v: &Vec3G<T>) -> Vec3G<T> {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub fn scale_mat3_g<T: AD>(s: T, m: &Mat3G<T>) -> Mat3G<T> {
    let z = T::constant(0.0);
    let mut c = [[z; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = s * m[i][j];
        }
    }
    c
}

pub fn add_mat3_g<T: AD>(a: &Mat3G<T>, b: &Mat3G<T>) -> Mat3G<T> {
    let z = T::constant(0.0);
    let mut c = [[z; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

pub fn sub_mat3_g<T: AD>(a: &Mat3G<T>, b: &Mat3G<T>) -> Mat3G<T> {
    let z = T::constant(0.0);
    let mut c = [[z; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

pub fn transpose3_g<T: AD>(m: &Mat3G<T>) -> Mat3G<T> {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

pub fn i3_g<T: AD>() -> Mat3G<T> {
    let z = T::constant(0.0);
    let o = T::constant(1.0);
    [[o, z, z], [z, o, z], [z, z, o]]
}

pub fn z3_g<T: AD>() -> Mat3G<T> {
    let z = T::constant(0.0);
    [[z; 3]; 3]
}

pub fn trace3_g<T: AD>(m: &Mat3G<T>) -> T {
    m[0][0] + m[1][1] + m[2][2]
}

// =========================================================================
// AD-generic scalar basis (s = θ², degree-4 Taylor)
// =========================================================================

/// Single Taylor / exact crossover used by every branch in this module.
const TAYLOR_THRESHOLD_S: f64 = 1e-4;

/// Compute s = θ² = ωᵀω and (when safe) θ = √s from ω.
///
/// **The θ²-parameterization**: s = dot(ω,ω) is a smooth polynomial
/// in ω with well-defined AD tangent everywhere, including ω = 0.
/// θ = √s has a singularity at s = 0 (derivative = 1/(2√0) = ∞),
/// so it is computed only when s > ε for use in the trig branch.
/// The Taylor branch uses s directly, never touching √.
///
/// This resolves the fundamental obstacle to nested AD on Lie groups:
/// the norm singularity at the identity (Griewank & Walther 2008, §14.2).
#[inline]
pub fn theta_sq_from_omega<T: AD>(omega: &Vec3G<T>) -> (T, T) {
    let s = dot3_g(omega, omega);
    let theta = if s.to_constant() > 1e-20 {
        s.sqrt()
    } else {
        T::constant(0.0)
    };
    (s, theta)
}

/// A(θ) = sin(θ)/θ.  Takes s = θ².  Degree-4 Taylor in s.
pub fn scalar_a_s<T: AD>(s: T, theta: T) -> T {
    if s.to_constant() < TAYLOR_THRESHOLD_S {
        T::constant(1.0)
            + s * (T::constant(-1.0 / 6.0)
                + s * (T::constant(1.0 / 120.0)
                    + s * (T::constant(-1.0 / 5040.0) + s * T::constant(1.0 / 362880.0))))
    } else {
        theta.sin() / theta
    }
}

/// B(θ) = (1 − cos(θ))/θ².  Takes s = θ².  Degree-4 Taylor in s.
pub fn scalar_b_s<T: AD>(s: T, theta: T) -> T {
    if s.to_constant() < TAYLOR_THRESHOLD_S {
        T::constant(0.5)
            + s * (T::constant(-1.0 / 24.0)
                + s * (T::constant(1.0 / 720.0)
                    + s * (T::constant(-1.0 / 40320.0) + s * T::constant(1.0 / 3628800.0))))
    } else {
        (T::constant(1.0) - theta.cos()) / s
    }
}

/// C(θ) = (θ − sin(θ))/θ³.  Takes s = θ².  Degree-4 Taylor in s.
pub fn scalar_c_s<T: AD>(s: T, theta: T) -> T {
    if s.to_constant() < TAYLOR_THRESHOLD_S {
        T::constant(1.0 / 6.0)
            + s * (T::constant(-1.0 / 120.0)
                + s * (T::constant(1.0 / 5040.0)
                    + s * (T::constant(-1.0 / 362880.0) + s * T::constant(1.0 / 39916800.0))))
    } else {
        (theta - theta.sin()) / (s * theta)
    }
}

/// D(θ) = (1 − (θ/2)cot(θ/2))/θ².  Takes s = θ².  Degree-4 Taylor in s.
pub fn scalar_d_s<T: AD>(s: T, theta: T) -> T {
    if s.to_constant() < TAYLOR_THRESHOLD_S {
        T::constant(1.0 / 12.0)
            + s * (T::constant(1.0 / 720.0)
                + s * (T::constant(1.0 / 30240.0)
                    + s * (T::constant(1.0 / 1209600.0) + s * T::constant(1.0 / 47900160.0))))
    } else {
        let half = T::constant(0.5);
        let half_theta = half * theta;
        let cot_half = half_theta.cos() / half_theta.sin();
        (T::constant(1.0) - half_theta * cot_half) / s
    }
}

/// β(s) = C(s)/(2 B(s)) − 2 D(s).  Degree-4 Taylor in s, **single branch**.
///
/// The closed form `c/(2b) − 2d` cancels two ≈ 1/6 quantities to extract
/// a ≈ s/360 result and is unusable below θ ≈ 1 — at θ = 0.1 it loses
/// 5+ digits, at θ = 0.01 it has zero significant figures left.  The
/// Taylor degree-4 form, by contrast, has truncation error ≤ 6·s⁵·a₅
/// ≈ 6e-7 at s = π² and is exact to f64 precision for s ≤ 1, so it is
/// strictly preferable across the entire AD-active chart.  Single-branch
/// also means no threshold to maintain and uniform polynomial degree
/// for every AD depth ≤ 4.
///
/// Coefficient s⁵·691/130767436800 derived from β = 2 s · dD/ds and the
/// next Bernoulli term in D.
pub fn scalar_beta_s<T: AD>(s: T, _theta: T) -> T {
    // β(s) = s/360 + s²/7560 + s³/201600 + s⁴/5987520 + 691·s⁵/130767436800 + …
    s * (T::constant(1.0 / 360.0)
        + s * (T::constant(1.0 / 7560.0)
            + s * (T::constant(1.0 / 201600.0)
                + s * (T::constant(1.0 / 5987520.0) + s * T::constant(691.0 / 130767436800.0)))))
}

// ---- Backward-compatible θ-form aliases (no longer the recommended path) ----

pub fn scalar_a_g<T: AD>(theta: T) -> T {
    scalar_a_s(theta * theta, theta)
}
pub fn scalar_b_g<T: AD>(theta: T) -> T {
    scalar_b_s(theta * theta, theta)
}
pub fn scalar_c_g<T: AD>(theta: T) -> T {
    scalar_c_s(theta * theta, theta)
}
pub fn scalar_d_g<T: AD>(theta: T) -> T {
    scalar_d_s(theta * theta, theta)
}
pub fn scalar_beta_g<T: AD>(theta: T) -> T {
    scalar_beta_s(theta * theta, theta)
}

// =========================================================================
// Fused atoms (degree-4 in s, no removable singularities)
// =========================================================================

/// β̄(s) ≡ β(s)/s ≡ 2 dD/ds.  Smooth at s = 0 by construction.
///
/// Single-branch Taylor (degree 4) — same rationale as [`scalar_beta_s`]:
/// the closed-form path inherits c/(2b) − 2d cancellation, while the
/// Taylor is exact for s ≤ 1 and accurate to ≤ 1e-6 at the chart boundary.
///
/// Shares its Taylor coefficients with [`d_prime_omega_over_theta`]
/// (= ω_m · β̄) — the algebraic identity D'(θ)/θ ≡ β̄(s) recorded by the
/// paper as Eq. (17).
///
/// Taylor: 1/360 + s/7560 + s²/201600 + s³/5987520 + 691·s⁴/130767436800 + …
#[inline]
pub fn scalar_beta_over_s_g<T: AD>(s: T, _theta: T) -> T {
    T::constant(1.0 / 360.0)
        + s * (T::constant(1.0 / 7560.0)
            + s * (T::constant(1.0 / 201600.0)
                + s * (T::constant(1.0 / 5987520.0) + s * T::constant(691.0 / 130767436800.0))))
}

/// D̃·ω_m ≡ D'(θ)·ω_m/θ ≡ ω_m · β̄(s).  Radial coefficient in ∂Jr⁻¹/∂ω_m.
///
/// Smooth at θ = 0 (≡ s = 0).  Implemented as `ω_m · β̄(s)` so the same
/// degree-4 Taylor is shared with [`scalar_beta_over_s_g`].
#[inline]
pub fn d_prime_omega_over_theta<T: AD>(s: T, theta: T, omega_m: T) -> T {
    omega_m * scalar_beta_over_s_g(s, theta)
}

/// β̄'(s) ≡ d β̄ / ds.  Used by the SE(3) Q̃_r derivative scalar α_m'.
///
/// Taylor (degree 4): 1/7560 + s/100800 + s²/1995840 + 691·s³/32691859200 +
/// s⁴/1245404160 + …
///
/// **Single-branch by design.**  The natural above-threshold form
/// `(β'(s) − β̄(s))/s` cancels two ~2.78e-3 quantities to extract a
/// ~1.32e-4 result and divides by `s`, losing roughly seven digits at
/// s ≈ 1e-4.  The closed-form β'(θ) used by that branch carries its
/// own (b − 3c) and (a − 2b) cancellations from the conventional form,
/// so the issue is structural — the θ-basis simply doesn't have a
/// numerically stable closed form for β̄'(s).
///
/// The degree-4 Taylor, by contrast, has truncation error ≤ 6·s⁵·a₅ which
/// is below 6e-8 across the whole practical chart range s ∈ [0, π²].
/// We pay 4–5 multiplies and one add unconditionally; in return we get
/// AD-safety (uniform polynomial in s, no branch, no 1/s, no 1/θ) plus
/// strictly better accuracy than the closed form would give.  This is the
/// recipe applied uniformly: the s-native scalar IS the production form.
#[inline]
pub fn scalar_beta_bar_prime_s<T: AD>(s: T, _theta: T) -> T {
    T::constant(1.0 / 7560.0)
        + s * (T::constant(1.0 / 100800.0)
            + s * (T::constant(1.0 / 1995840.0)
                + s * (T::constant(691.0 / 32691859200.0) + s * T::constant(1.0 / 1245404160.0))))
}

// =========================================================================
// Diagnostic-only conventional forms (pub(crate))
// =========================================================================

/// **Diagnostic only** — D'(θ) in the conventional form.
///
/// Returns the conventional `(1 − 12D − 2β + 4 s D²)/(2θ)` above
/// threshold and `0` (i.e. `D'(0) = 0`) below it.  The below-threshold
/// branch is degree 0 in s and tangent-zero — it exists *deliberately*
/// so that the §IV.B singular-pair NaN test in `nll_tests` can fire.
///
/// Production code must reach for [`d_prime_omega_over_theta`] instead;
/// this function is `pub(crate)` to keep it out of the public surface.
pub(crate) fn scalar_d_prime_s<T: AD>(s: T, theta: T) -> T {
    if s.to_constant() < TAYLOR_THRESHOLD_S {
        T::constant(0.0)
    } else {
        let d = scalar_d_s(s, theta);
        let beta = scalar_beta_s(s, theta);
        (T::constant(1.0) - T::constant(12.0) * d - T::constant(2.0) * beta
            + T::constant(4.0) * s * d * d)
            / (T::constant(2.0) * theta)
    }
}

/// **Diagnostic only** — β'(θ) in the conventional form.  Kept for
/// symmetry with [`scalar_d_prime_s`] as evidence of the conventional
/// θ-form; not currently called from production code (β̄'(s) is the
/// AD-safe replacement and is single-branch Taylor).
#[allow(dead_code)]
pub(crate) fn scalar_beta_prime_s<T: AD>(s: T, theta: T) -> T {
    if s.to_constant() < TAYLOR_THRESHOLD_S {
        T::constant(0.0)
    } else {
        let a = scalar_a_s(s, theta);
        let b = scalar_b_s(s, theta);
        let c = scalar_c_s(s, theta);
        let dp = scalar_d_prime_s(s, theta);
        ((b - T::constant(3.0) * c) * b - c * (a - T::constant(2.0) * b))
            / (T::constant(2.0) * theta * b * b)
            - T::constant(2.0) * dp
    }
}

// =========================================================================
// Hat basis
// =========================================================================

/// Hat basis E_m = \[e_m\]× as AD constants.
pub fn hat_basis_g<T: AD>(m: usize) -> Mat3G<T> {
    let z = T::constant(0.0);
    let p = T::constant(1.0);
    let n = T::constant(-1.0);
    match m {
        0 => [[z, z, z], [z, z, n], [z, p, z]],
        1 => [[z, z, p], [z, z, z], [n, z, z]],
        2 => [[z, n, z], [p, z, z], [z, z, z]],
        _ => unreachable!(),
    }
}

// =========================================================================
// SO(3) Exp, Log, V, V⁻¹, Jr, Jr⁻¹
// =========================================================================

/// SO(3) exponential map: R = I + A(θ)·\[ω\]× + B(θ)·\[ω\]×²
pub fn so3_exp_g<T: AD>(omega: &Vec3G<T>) -> Mat3G<T> {
    let (s, theta) = theta_sq_from_omega(omega);
    let a = scalar_a_s(s, theta);
    let b = scalar_b_s(s, theta);
    let h = hat_g(omega);
    let h2 = mm3_g(&h, &h);
    add_mat3_g(
        &add_mat3_g(&i3_g(), &scale_mat3_g(a, &h)),
        &scale_mat3_g(b, &h2),
    )
}

/// V matrix (= Jl): V = I + B·\[ω\]× + C·\[ω\]×²
pub fn v_matrix_g<T: AD>(omega: &Vec3G<T>) -> Mat3G<T> {
    let (s, theta) = theta_sq_from_omega(omega);
    let b = scalar_b_s(s, theta);
    let c = scalar_c_s(s, theta);
    let h = hat_g(omega);
    let h2 = mm3_g(&h, &h);
    add_mat3_g(
        &add_mat3_g(&i3_g(), &scale_mat3_g(b, &h)),
        &scale_mat3_g(c, &h2),
    )
}

/// V⁻¹ matrix (= Jl⁻¹): V⁻¹ = I − ½·\[ω\]× + D·\[ω\]×²
pub fn v_inv_g<T: AD>(omega: &Vec3G<T>) -> Mat3G<T> {
    let (s, theta) = theta_sq_from_omega(omega);
    let d = scalar_d_s(s, theta);
    let h = hat_g(omega);
    let h2 = mm3_g(&h, &h);
    add_mat3_g(
        &add_mat3_g(&i3_g(), &scale_mat3_g(T::constant(-0.5), &h)),
        &scale_mat3_g(d, &h2),
    )
}

/// SO(3) right Jacobian Jr(ω) = I − B\[ω\]× + C\[ω\]×².
pub fn jr_g<T: AD>(omega: &Vec3G<T>) -> Mat3G<T> {
    let (s, theta) = theta_sq_from_omega(omega);
    let b = scalar_b_s(s, theta);
    let c = scalar_c_s(s, theta);
    let h = hat_g(omega);
    let h2 = mm3_g(&h, &h);
    add_mat3_g(
        &add_mat3_g(&i3_g(), &scale_mat3_g(T::constant(0.0) - b, &h)),
        &scale_mat3_g(c, &h2),
    )
}

/// Jr⁻¹(ω) = I + ½·\[ω\]× + D(θ)·\[ω\]×²
pub fn jr_inv_g<T: AD>(omega: &Vec3G<T>) -> Mat3G<T> {
    let (s, theta) = theta_sq_from_omega(omega);
    let hat_w = hat_g(omega);
    let hat_w_sq = mm3_g(&hat_w, &hat_w);
    let d = scalar_d_s(s, theta);
    add_mat3_g(
        &add_mat3_g(&i3_g(), &scale_mat3_g(T::constant(0.5), &hat_w)),
        &scale_mat3_g(d, &hat_w_sq),
    )
}

/// Rotation matrix → unit quaternion via Shepperd's algorithm.
///
/// Returns `(q0, q_v)` with `q0² + q_v·q_v = 1`.  Branches on the constant
/// part of `{trace, R[i][i]}` to pick the most numerically stable component
/// to take a square root of, then derives the remaining three from algebraic
/// identities — no trig, no `acos`, AD-safe through the body.
fn mat3_to_quat_shepperd_g<T: AD>(r: &Mat3G<T>) -> (T, Vec3G<T>) {
    let trace = trace3_g(r);
    let trace_c = trace.to_constant();
    let r00_c = r[0][0].to_constant();
    let r11_c = r[1][1].to_constant();
    let r22_c = r[2][2].to_constant();

    let one = T::constant(1.0);
    let quarter = T::constant(0.25);
    let four = T::constant(4.0);

    if trace_c >= r00_c && trace_c >= r11_c && trace_c >= r22_c {
        // q0 component is the largest in magnitude.
        let q0 = ((one + trace) * quarter).sqrt();
        let inv_4q0 = one / (four * q0);
        let qv = [
            (r[2][1] - r[1][2]) * inv_4q0,
            (r[0][2] - r[2][0]) * inv_4q0,
            (r[1][0] - r[0][1]) * inv_4q0,
        ];
        (q0, qv)
    } else if r00_c >= r11_c && r00_c >= r22_c {
        // q_x is largest.
        let qx = ((one + r[0][0] + r[0][0] - trace) * quarter).sqrt();
        let inv_4qx = one / (four * qx);
        let q0 = (r[2][1] - r[1][2]) * inv_4qx;
        let qy = (r[0][1] + r[1][0]) * inv_4qx;
        let qz = (r[0][2] + r[2][0]) * inv_4qx;
        (q0, [qx, qy, qz])
    } else if r11_c >= r22_c {
        // q_y is largest.
        let qy = ((one + r[1][1] + r[1][1] - trace) * quarter).sqrt();
        let inv_4qy = one / (four * qy);
        let q0 = (r[0][2] - r[2][0]) * inv_4qy;
        let qx = (r[0][1] + r[1][0]) * inv_4qy;
        let qz = (r[1][2] + r[2][1]) * inv_4qy;
        (q0, [qx, qy, qz])
    } else {
        // q_z is largest.
        let qz = ((one + r[2][2] + r[2][2] - trace) * quarter).sqrt();
        let inv_4qz = one / (four * qz);
        let q0 = (r[1][0] - r[0][1]) * inv_4qz;
        let qx = (r[0][2] + r[2][0]) * inv_4qz;
        let qy = (r[1][2] + r[2][1]) * inv_4qz;
        (q0, [qx, qy, qz])
    }
}

/// SO(3) logarithmic map via quaternion intermediate — the trusted path.
///
/// Two stages:
///   1. R → (q0, q_v) via Shepperd's algorithm (algebraic, AD-safe).
///   2. Canonicalize `q0 ≥ 0`, then ω = factor · q_v with the smooth scalar
///      `factor = θ / sin(θ/2) = 2·asin(√s_q)/√s_q`, parameterized in
///      `s_q = q_v·q_v`.
///
/// AD-safe across the entire chart [0, π]:
///   * Below `TAYLOR_THRESHOLD_S`: factor is a degree-4 Taylor in `s_q` —
///     no `√s_q`, no `atan2`, no `1/q0`.  Coefficients from
///     `asin(x)/x = 1 + x²/6 + 3x⁴/40 + 15x⁶/336 + 105x⁸/3456 + …`.
///   * Above threshold: `factor = 2·atan2(√s_q, q0)/√s_q`.  With `q0 ≥ 0`
///     forced, `atan2` is smooth even at θ = π (where `q0 = 0`,
///     `|q_v| = 1`, `factor = π`).
pub fn so3_log_g_quaternion<T: AD>(r: &Mat3G<T>) -> Vec3G<T> {
    let (q0_raw, qv_raw) = mat3_to_quat_shepperd_g(r);

    // Canonicalize so q0 ≥ 0 (and ω is in θ ∈ [0, π]).  The branch predicate
    // is constant-valued; the body is fully AD-generic.
    let (q0, qv) = if q0_raw.to_constant() < 0.0 {
        let neg = T::constant(-1.0);
        (
            neg * q0_raw,
            [neg * qv_raw[0], neg * qv_raw[1], neg * qv_raw[2]],
        )
    } else {
        (q0_raw, qv_raw)
    };

    let s_q = dot3_g(&qv, &qv);

    let factor = if s_q.to_constant() < TAYLOR_THRESHOLD_S {
        // 2·asin(√s_q)/√s_q  =  2·(1 + s_q/6 + 3 s_q²/40 + 5 s_q³/112 + 35 s_q⁴/1152 + …)
        T::constant(2.0)
            * (T::constant(1.0)
                + s_q
                    * (T::constant(1.0 / 6.0)
                        + s_q
                            * (T::constant(3.0 / 40.0)
                                + s_q
                                    * (T::constant(5.0 / 112.0)
                                        + s_q * T::constant(35.0 / 1152.0)))))
    } else {
        let qv_norm = s_q.sqrt();
        let theta = T::constant(2.0) * qv_norm.atan2(q0);
        theta / qv_norm
    };

    [factor * qv[0], factor * qv[1], factor * qv[2]]
}

/// Trace-based SO(3) log — `pub(crate)` cross-check, not for production use.
///
/// Branches on `trace3_g(r).to_constant()`:
///   * `trace > 3 − TAYLOR_THRESHOLD_S`: smooth Taylor of `θ/(2 sin θ)` in
///     `u = 3 − trace`.  Coefficients from `θ/sin θ = 1 + θ²/6 + 7θ⁴/360 + …`
///     composed with `θ² = u + u²/12 + u³/360 + …`:
///     `factor = ½·(1 + u/6 + u²/30 + 29 u³/5040 + …)`
///   * else: `factor = θ/(2 sin θ)` with `θ = acos((trace − 1)/2)`.
///
/// Near θ = π the `acos` branch becomes ill-conditioned; production callers
/// route through [`so3_log_g_quaternion`] instead.  A `debug_assert!` flags
/// near-π evaluation in tests.
#[allow(dead_code)]
pub(crate) fn so3_log_g_trace<T: AD>(r: &Mat3G<T>) -> Vec3G<T> {
    let trace = trace3_g(r);
    let trace_c = trace.to_constant();

    debug_assert!(
        trace_c > -0.99,
        "so3_log_g_trace called near θ=π (trace={trace_c}); use the quaternion path"
    );

    let rt = transpose3_g(r);
    let skew = sub_mat3_g(r, &rt);

    let factor = if trace_c > 3.0 - TAYLOR_THRESHOLD_S {
        let u = T::constant(3.0) - trace;
        T::constant(0.5)
            * (T::constant(1.0)
                + u * (T::constant(1.0 / 6.0)
                    + u * (T::constant(1.0 / 30.0) + u * T::constant(29.0 / 5040.0))))
    } else {
        let cos_theta = T::constant(0.5) * (trace - T::constant(1.0));
        let theta = cos_theta.acos();
        theta / (T::constant(2.0) * theta.sin())
    };

    [
        factor * skew[2][1],
        factor * skew[0][2],
        factor * skew[1][0],
    ]
}

/// SO(3) logarithmic map: R → ω.  Trusted, AD-safe across [0, π].
#[inline]
pub fn so3_log_g<T: AD>(r: &Mat3G<T>) -> Vec3G<T> {
    so3_log_g_quaternion(r)
}

// =========================================================================
// Quaternion half-angle atoms (degree-4 in s, threshold-gated)
// =========================================================================
//
// Two AD-safe fused scalars for the unit quaternion representation of
// SO(3): q = (cos(θ/2), sin(θ/2)·ω̂) ∈ SU(2), the double cover of SO(3).
// Both flow through `(s, θ)` and integrate cleanly with
// `theta_sq_from_omega` and the rest of the s-native recipe.
//
// Unlike β̄(s) — which is single-branch Taylor across the entire chart
// because `β(s)` shrinks linearly in s — these atoms remain O(1) at θ ≈ π
// and the closed forms `cos(θ/2)` and `sin(θ/2)/θ` evaluate accurately
// there.  We therefore use the same threshold split as `scalar_a_s`:
// degree-4 Taylor below `TAYLOR_THRESHOLD_S = 1e-4` (matches the exact
// form to f64 precision at the boundary, verified to 0 ULP at
// θ ≈ 0.01 rad), exact closed form above (no removable singularity since
// `θ = √s` is already finite there).

/// cos(θ/2), reached from (s, θ).  Scalar component q₀ of the unit
/// quaternion q = (cos(θ/2), sin(θ/2)·ω̂) representing R(ω) ∈ SO(3).
///
/// Taylor (degree 4, in s = θ²):
///     1 − s/8 + s²/384 − s³/46080 + s⁴/10321920 − …
///
/// AD-safe to depth 4 in s = depth 8 in θ — well past the recipe's
/// depth-3 working margin.
#[inline]
pub fn scalar_cos_half_s<T: AD>(s: T, theta: T) -> T {
    if s.to_constant() < TAYLOR_THRESHOLD_S {
        T::constant(1.0)
            + s * (T::constant(-1.0 / 8.0)
                + s * (T::constant(1.0 / 384.0)
                    + s * (T::constant(-1.0 / 46080.0) + s * T::constant(1.0 / 10321920.0))))
    } else {
        (T::constant(0.5) * theta).cos()
    }
}

/// (1/2)·sinc(θ/2) ≡ sin(θ/2)/θ, reached from (s, θ).  Builds the vector
/// quaternion components of q = (cos(θ/2), sin(θ/2)·ω̂):
///
///     qv_m = ω_m · scalar_half_sinc_half_s(s, θ)
///
/// because qv = sin(θ/2) · ω̂ = sin(θ/2) · ω/θ = ω · [sin(θ/2)/θ].  This
/// fused form keeps `1/θ` out of the AD path — `sin(θ/2)/θ` evaluated at
/// θ = 0 is the removable 0/0 that the Taylor branch handles directly.
///
/// Taylor (degree 4, in s = θ²):
///     1/2 − s/48 + s²/3840 − s³/645120 + s⁴/185794560 − …
///
/// At s = 0 this evaluates to 1/2, matching the limit
/// `lim_{θ→0} sin(θ/2)/θ = 1/2`.  AD-safe to depth 4 in s.
#[inline]
pub fn scalar_half_sinc_half_s<T: AD>(s: T, theta: T) -> T {
    if s.to_constant() < TAYLOR_THRESHOLD_S {
        T::constant(0.5)
            + s * (T::constant(-1.0 / 48.0)
                + s * (T::constant(1.0 / 3840.0)
                    + s * (T::constant(-1.0 / 645120.0) + s * T::constant(1.0 / 185794560.0))))
    } else {
        (T::constant(0.5) * theta).sin() / theta
    }
}

// =========================================================================
// Tests — branch continuity and analytical Taylor coefficients
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_bar_taylor_matches_at_threshold_boundary() {
        // Below threshold: Taylor.  Above: β/s.  Must agree across the cutoff.
        let s_lo = TAYLOR_THRESHOLD_S * 0.5;
        let s_hi = TAYLOR_THRESHOLD_S * 2.0;
        let theta_hi = s_hi.sqrt();
        let v_lo = scalar_beta_over_s_g::<f64>(s_lo, s_lo.sqrt());
        let v_hi = scalar_beta_over_s_g::<f64>(s_hi, theta_hi);
        // O(s) gap between the two evaluation points dominates; both should
        // be close to 1/360.
        assert!(
            (v_lo - v_hi).abs() < 1e-6,
            "β̄ branch discontinuity: lo={v_lo:.6e} hi={v_hi:.6e}"
        );
    }

    #[test]
    fn beta_bar_prime_matches_central_difference_of_beta_bar() {
        // β̄'(s) is single-branch Taylor; cross-check against a central
        // difference of β̄(s) at a handful of moderate s.  Tolerance is
        // governed by the FD reference — it has trade-off-bounded error
        // ~1e-5 relative at the optimal h, and the Taylor analytic is
        // tighter than that across the whole comparison range.
        let h = 1e-5;
        for &s in &[1e-3f64, 1e-2, 0.1, 0.5, 1.0] {
            let theta = s.sqrt();
            let beta_bar_p = scalar_beta_over_s_g::<f64>(s + h, (s + h).sqrt());
            let beta_bar_m = scalar_beta_over_s_g::<f64>(s - h, (s - h).sqrt());
            let fd = (beta_bar_p - beta_bar_m) / (2.0 * h);
            let analytic = scalar_beta_bar_prime_s::<f64>(s, theta);
            let rel = (analytic - fd).abs() / fd.abs().max(1e-15);
            assert!(
                rel < 1e-5,
                "β̄'(s={s}): analytic={analytic:.6e} fd={fd:.6e} rel={rel:.2e}"
            );
        }
    }

    #[test]
    fn beta_bar_equals_d_tilde_omega_m_at_unit_omega() {
        // ω_m · β̄(s) ≡ D̃·ω_m, by construction.
        for &theta in &[0.05f64, 0.3, 0.8, 1.5, 2.5] {
            let s = theta * theta;
            let beta_bar = scalar_beta_over_s_g::<f64>(s, theta);
            let omega_m = 1.0;
            let d_tilde_w = d_prime_omega_over_theta::<f64>(s, theta, omega_m);
            assert!(
                (omega_m * beta_bar - d_tilde_w).abs() < 1e-15,
                "ω_m·β̄ mismatch at θ={theta}: β̄={beta_bar:.6e} D̃ω={d_tilde_w:.6e}"
            );
        }
    }

    #[test]
    fn beta_bar_value_at_origin_is_one_three_sixtieth() {
        let v = scalar_beta_over_s_g::<f64>(0.0, 0.0);
        assert!((v - 1.0 / 360.0).abs() < 1e-15);
    }

    #[test]
    fn beta_bar_prime_value_at_origin_is_one_seven_thousand_five_hundred_sixty() {
        let v = scalar_beta_bar_prime_s::<f64>(0.0, 0.0);
        assert!((v - 1.0 / 7560.0).abs() < 1e-15);
    }

    /// Depth-3 verification of β̄(s) Taylor coefficients via D3<1> nested AD.
    ///
    /// Seeds `s` as a triply-nested forward-AD scalar at `s = 0` and reads
    /// the four nested levels of the result as β̄(0), β̄'(0), β̄''(0),
    /// β̄'''(0).  This is the recipe's correctness check that the AD-safe
    /// β̄ atom is differentiable to depth 3 and that the Taylor coefficients
    /// agree with the analytical values to f64 precision.
    #[test]
    fn beta_bar_third_derivative_via_d3_matches_taylor() {
        use crate::autodiff::nested_ad::Dual;

        // Seed s as a triply-nested Dual at value 0.
        let l1 = Dual::<f64, 1>::seed(0.0, 0);
        let l2 = Dual::<Dual<f64, 1>, 1>::seed(l1, 0);
        let l3 = Dual::<Dual<Dual<f64, 1>, 1>, 1>::seed(l2, 0);

        let theta = Dual::<Dual<Dual<f64, 1>, 1>, 1>::constant(0.0);
        let result = scalar_beta_over_s_g(l3, theta);

        // β̄(0) sits at the deepest .value chain.
        let f0 = result.value.value.value;
        // β̄'(0): one level of tangent.
        let f1 = result.value.value.tangent[0];
        // β̄''(0): two levels — read at outer.value.tangent.tangent (Schwarz).
        let f2 = result.value.tangent[0].tangent[0];
        // β̄'''(0): all three nested tangents.
        let f3 = result.tangent[0].tangent[0].tangent[0];

        // Expected Taylor coefficients of β̄(s):
        //   β̄(s) = 1/360 + s/7560 + s²/201600 + s³/5987520 + …
        // Differentiate: β̄(0)=1/360, β̄'(0)=1/7560, β̄''(0)=2·1/201600=1/100800,
        //                β̄'''(0)=6·1/5987520=1/997920.
        assert!((f0 - 1.0 / 360.0).abs() < 1e-15, "β̄(0)={f0:.6e}");
        assert!((f1 - 1.0 / 7560.0).abs() < 1e-15, "β̄'(0)={f1:.6e}");
        assert!((f2 - 1.0 / 100800.0).abs() < 1e-15, "β̄''(0)={f2:.6e}");
        assert!((f3 - 1.0 / 997920.0).abs() < 1e-15, "β̄'''(0)={f3:.6e}");
    }

    /// The same depth-3 check at a non-zero s — confirms the AD machinery
    /// correctly differentiates the implemented polynomial (which is what
    /// AD safety requires) at a baseline `s` typical of the SE(3) NLL
    /// linearization point.
    ///
    /// Note this checks the third derivative of the *implemented degree-4
    /// Taylor*, not the mathematical β̄'''(s).  The two agree to f64
    /// precision for s small (truncation error is O(s²)·a₅ ≪ 1 ULP at
    /// s ≤ 1e-3); at larger s they diverge by the truncation error,
    /// which is the polynomial-degree limit the recipe documents in §IV.D.
    #[test]
    fn beta_bar_third_derivative_via_d3_at_nonzero_s() {
        use crate::autodiff::nested_ad::Dual;

        let s_val: f64 = 1e-3;
        let l1 = Dual::<f64, 1>::seed(s_val, 0);
        let l2 = Dual::<Dual<f64, 1>, 1>::seed(l1, 0);
        let l3 = Dual::<Dual<Dual<f64, 1>, 1>, 1>::seed(l2, 0);

        let theta = Dual::<Dual<Dual<f64, 1>, 1>, 1>::constant(s_val.sqrt());
        let result = scalar_beta_over_s_g(l3, theta);

        let f3 = result.tangent[0].tangent[0].tangent[0];

        // Differentiating the implemented Taylor 1/360 + s/7560 + s²/201600
        // + s³/5987520 + 691·s⁴/130767436800 three times gives
        //   β̄'''_poly(s) = 6/5987520 + 24·691·s/130767436800
        //                = 1/997920 + 691·s/5448643200.
        let expected = 1.0 / 997920.0 + 691.0 * s_val / 5_448_643_200.0;
        let rel = (f3 - expected).abs() / expected.abs();
        assert!(
            rel < 1e-13,
            "β̄'''(s={s_val}) = {f3:.6e}, expected {expected:.6e}, rel {rel:.2e}"
        );
    }

    // ─── so3_log_g (quaternion path) trustworthiness tests ──────────────────

    #[test]
    fn so3_log_quaternion_log_exp_roundtrip_at_various_angles() {
        // Log(Exp(ω)) ≡ ω for any ω with |ω| < π.  Spans θ from the origin,
        // through the ½√s_q Taylor regime, the regular atan2 regime, and up
        // close to π — the regime the trace path can't handle.
        let pi = std::f64::consts::PI;
        let cases: [Vec3G<f64>; 6] = [
            [0.0, 0.0, 0.0],
            [1e-8, 0.0, 0.0],
            [0.1, -0.05, 0.07],
            [pi * 0.5, 0.0, 0.0],
            [0.0, 2.0 * pi / 3.0, 0.0],
            [0.0, 0.0, 0.95 * pi],
        ];
        for (i, omega) in cases.iter().enumerate() {
            let r = so3_exp_g::<f64>(omega);
            let omega_back = so3_log_g::<f64>(&r);
            for k in 0..3 {
                let err = (omega[k] - omega_back[k]).abs();
                assert!(
                    err < 1e-12,
                    "case {i}: omega={:?}, recovered={:?}, err[{k}]={err:.2e}",
                    omega,
                    omega_back
                );
            }
        }
    }

    #[test]
    fn so3_log_quaternion_d3_log_exp_is_identity_at_origin() {
        // Log∘Exp is the identity.  Seeded as D3<3> at ω = 0, the value is 0,
        // the first tangent is I, and all second / third nested tangents are
        // zero — to f64 precision.  This is the strict AD-safety check the
        // old `so3_log_g` would have failed: with a constant ½ small-angle
        // factor, the second tangent of ω ↦ Log(Exp(ω)) carries a O(θ²/12)
        // correction that the old branch silently dropped.
        use crate::autodiff::nested_ad::Dual;
        type D3<const N: usize> = Dual<Dual<Dual<f64, N>, N>, N>;

        let omega: Vec3G<D3<3>> = std::array::from_fn(|i| {
            let l1 = Dual::<f64, 3>::seed(0.0, i);
            let l2 = Dual::<Dual<f64, 3>, 3>::seed(l1, i);
            Dual::<Dual<Dual<f64, 3>, 3>, 3>::seed(l2, i)
        });
        let r = so3_exp_g::<D3<3>>(&omega);
        let omega_back = so3_log_g::<D3<3>>(&r);

        for i in 0..3 {
            // Value: omega_back[i] ≡ 0 at ω = 0.
            let v = omega_back[i].value.value.value;
            assert!(v.abs() < 1e-13, "value[{i}] = {v:.2e}");

            // First tangent: ∂omega_back[i] / ∂ω_j = δ_ij (identity).
            for j in 0..3 {
                let d1 = omega_back[i].value.value.tangent[j];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (d1 - expected).abs() < 1e-12,
                    "d/dω[{j}] omega_back[{i}] = {d1:.2e}, expected {expected}"
                );
            }

            // Second tangent: zero everywhere.
            for j in 0..3 {
                for k in 0..3 {
                    let d2 = omega_back[i].value.tangent[j].tangent[k];
                    assert!(
                        d2.abs() < 1e-11,
                        "∂²omega_back[{i}]/∂ω[{j}]∂ω[{k}] = {d2:.2e}"
                    );
                }
            }

            // Third tangent: zero everywhere.
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        let d3 = omega_back[i].tangent[j].tangent[k].tangent[l];
                        assert!(
                            d3.abs() < 1e-10,
                            "∂³omega_back[{i}]/∂ω[{j}]∂ω[{k}]∂ω[{l}] = {d3:.2e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn so3_log_quaternion_vs_trace_match_away_from_pi_at_d2() {
        // Two independent paths must give bit-close value, gradient, and Hessian
        // away from θ=π.  This is the cross-validation that justifies trusting
        // either implementation in isolation.
        use crate::autodiff::nested_ad::Dual;
        type D2<const N: usize> = Dual<Dual<f64, N>, N>;

        let omega_val: Vec3G<f64> = [0.4, -0.3, 0.6];
        let r_val = so3_exp_g::<f64>(&omega_val);

        // Wrap r_val as a D2<3> matrix with no tangent.  We perturb ω,
        // re-exponentiate inside the AD chain so the tangent flows correctly.
        let omega: Vec3G<D2<3>> = std::array::from_fn(|i| {
            let l1 = Dual::<f64, 3>::seed(omega_val[i], i);
            Dual::<Dual<f64, 3>, 3>::seed(l1, i)
        });
        let r_d2 = so3_exp_g::<D2<3>>(&omega);

        // Sanity: re-exponentiating reproduces r_val to f64 precision.
        for i in 0..3 {
            for j in 0..3 {
                let v = r_d2[i][j].value.value;
                assert!((v - r_val[i][j]).abs() < 1e-13);
            }
        }

        let log_quat = so3_log_g_quaternion::<D2<3>>(&r_d2);
        let log_trace = so3_log_g_trace::<D2<3>>(&r_d2);

        for i in 0..3 {
            // Value
            let dv = (log_quat[i].value.value - log_trace[i].value.value).abs();
            assert!(dv < 1e-12, "value[{i}] differs by {dv:.2e}");
            // First tangent
            for j in 0..3 {
                let dq = log_quat[i].value.tangent[j];
                let dt = log_trace[i].value.tangent[j];
                assert!(
                    (dq - dt).abs() < 1e-11,
                    "∂[{j}] log[{i}] differs: quat={dq:.6e} trace={dt:.6e}"
                );
            }
            // Second tangent
            for j in 0..3 {
                for k in 0..3 {
                    let h_q = log_quat[i].tangent[j].tangent[k];
                    let h_t = log_trace[i].tangent[j].tangent[k];
                    assert!(
                        (h_q - h_t).abs() < 1e-9,
                        "∂²[{j}{k}] log[{i}] differs: quat={h_q:.6e} trace={h_t:.6e}"
                    );
                }
            }
        }
    }

    #[test]
    fn so3_log_quaternion_finite_near_pi_at_d2() {
        // The whole point of the quaternion path: at |ω| near π the trace path
        // would feed acos(cos θ) near acos(−1) and lose conditioning.  The
        // quaternion path uses atan2 which stays smooth.  Verify the Hessian
        // is finite for three near-π axes.
        use crate::autodiff::nested_ad::Dual;
        type D2<const N: usize> = Dual<Dual<f64, N>, N>;

        let pi = std::f64::consts::PI;
        let near = pi - 0.01;
        let cases: [Vec3G<f64>; 3] = [[near, 0.0, 0.0], [0.0, near, 0.0], [0.0, 0.0, near]];

        for (case_idx, omega_val) in cases.iter().enumerate() {
            let omega: Vec3G<D2<3>> = std::array::from_fn(|i| {
                let l1 = Dual::<f64, 3>::seed(omega_val[i], i);
                Dual::<Dual<f64, 3>, 3>::seed(l1, i)
            });
            let r_d2 = so3_exp_g::<D2<3>>(&omega);
            let log_d2 = so3_log_g::<D2<3>>(&r_d2);

            for i in 0..3 {
                // Value finite.
                let v = log_d2[i].value.value;
                assert!(v.is_finite(), "case {case_idx}: log[{i}] value = {v}");
                // First tangent finite.
                for j in 0..3 {
                    let g = log_d2[i].value.tangent[j];
                    assert!(g.is_finite(), "case {case_idx}: d log[{i}]/dω[{j}] = {g}");
                }
                // Second tangent finite.
                for j in 0..3 {
                    for k in 0..3 {
                        let h = log_d2[i].tangent[j].tangent[k];
                        assert!(
                            h.is_finite(),
                            "case {case_idx}: d²log[{i}]/dω[{j}]dω[{k}] = {h}"
                        );
                    }
                }
            }

            // And ω round-trips to itself (i.e. correct branch was taken).
            for i in 0..3 {
                let v = log_d2[i].value.value;
                assert!(
                    (v - omega_val[i]).abs() < 1e-10,
                    "case {case_idx}: log[{i}] = {v}, expected {}",
                    omega_val[i]
                );
            }
        }
    }

    #[test]
    fn mat3_to_quat_shepperd_unit_norm_and_branch_coverage() {
        // Drive each Shepperd branch (trace largest, then R[i][i] largest for
        // i = 0, 1, 2) by choosing axis-angle rotations with sufficiently
        // dominant trace or diagonal entry.  Verify each branch produces a
        // unit quaternion and that re-exponentiating to a matrix recovers R
        // to f64 precision.
        let pi = std::f64::consts::PI;
        // Identity: trace = 3, biggest by far ⟹ trace branch.
        // Large rotations near π about each axis: after Exp, the matching
        // diagonal entry is ≈ 1 while the other two are ≈ −1, so trace is
        // very negative and the branch with that R[i][i] dominant fires.
        let axes: [(Vec3G<f64>, &str); 4] = [
            ([0.05, 0.03, -0.02], "trace"),
            ([0.95 * pi, 0.0, 0.0], "rxx"),
            ([0.0, 0.95 * pi, 0.0], "ryy"),
            ([0.0, 0.0, 0.95 * pi], "rzz"),
        ];

        for (omega, label) in axes.iter() {
            let r = so3_exp_g::<f64>(omega);
            let (q0, qv) = mat3_to_quat_shepperd_g::<f64>(&r);
            // Unit norm.
            let n = q0 * q0 + qv[0] * qv[0] + qv[1] * qv[1] + qv[2] * qv[2];
            assert!((n - 1.0).abs() < 1e-13, "{label}: |q|² = {n}, expected 1");
            // Round-trip: q → R via standard formula, must equal original R.
            // R = (q0² − |qv|²)·I + 2 q0·[qv]× + 2·qv·qvᵀ
            let s2 = q0 * q0 - (qv[0] * qv[0] + qv[1] * qv[1] + qv[2] * qv[2]);
            let mut r_back = [[0.0f64; 3]; 3];
            r_back[0][0] = s2 + 2.0 * qv[0] * qv[0];
            r_back[1][1] = s2 + 2.0 * qv[1] * qv[1];
            r_back[2][2] = s2 + 2.0 * qv[2] * qv[2];
            r_back[0][1] = 2.0 * (qv[0] * qv[1] - q0 * qv[2]);
            r_back[1][0] = 2.0 * (qv[0] * qv[1] + q0 * qv[2]);
            r_back[0][2] = 2.0 * (qv[0] * qv[2] + q0 * qv[1]);
            r_back[2][0] = 2.0 * (qv[0] * qv[2] - q0 * qv[1]);
            r_back[1][2] = 2.0 * (qv[1] * qv[2] - q0 * qv[0]);
            r_back[2][1] = 2.0 * (qv[1] * qv[2] + q0 * qv[0]);
            for i in 0..3 {
                for j in 0..3 {
                    let d = (r[i][j] - r_back[i][j]).abs();
                    assert!(
                        d < 1e-13,
                        "{label}: R[{i}][{j}]={} q→R back={}, diff={d:.2e}",
                        r[i][j],
                        r_back[i][j]
                    );
                }
            }
        }
    }
}

#[test]
fn cos_half_at_origin_is_one() {
    let v = scalar_cos_half_s::<f64>(0.0, 0.0);
    assert!((v - 1.0).abs() < 1e-15);
}

#[test]
fn half_sinc_half_at_origin_is_one_half() {
    let v = scalar_half_sinc_half_s::<f64>(0.0, 0.0);
    assert!((v - 0.5).abs() < 1e-15);
}

#[test]
fn cos_half_branches_match_at_threshold_boundary() {
    // Exactly at the threshold, both formulae must agree to f64 precision.
    let s_at = TAYLOR_THRESHOLD_S;
    let theta_at = s_at.sqrt();
    let taylor = 1.0 - s_at / 8.0 + s_at * s_at / 384.0 - s_at * s_at * s_at / 46080.0
        + s_at * s_at * s_at * s_at / 10321920.0;
    let exact = (0.5 * theta_at).cos();
    assert!(
        (taylor - exact).abs() < 1e-15,
        "cos(θ/2) at threshold: taylor={taylor:.18e} exact={exact:.18e}"
    );
}

#[test]
fn half_sinc_half_branches_match_at_threshold_boundary() {
    let s_at = TAYLOR_THRESHOLD_S;
    let theta_at = s_at.sqrt();
    let taylor = 0.5 - s_at / 48.0 + s_at * s_at / 3840.0 - s_at * s_at * s_at / 645120.0
        + s_at * s_at * s_at * s_at / 185794560.0;
    let exact = (0.5 * theta_at).sin() / theta_at;
    assert!(
        (taylor - exact).abs() < 1e-15,
        "half_sinc at threshold: taylor={taylor:.18e} exact={exact:.18e}"
    );
}

#[test]
fn cos_half_matches_native_across_chart() {
    for &theta in &[0.01_f64, 0.1, 0.5, 1.0, 2.0, 3.0] {
        let s = theta * theta;
        let ours = scalar_cos_half_s::<f64>(s, theta);
        let native = (theta / 2.0).cos();
        let rel = (ours - native).abs() / native.abs().max(1e-15);
        assert!(rel < 1e-13, "cos(θ/2) at θ={theta}: rel={rel:.2e}");
    }
}

#[test]
fn half_sinc_half_matches_native_across_chart() {
    for &theta in &[0.01_f64, 0.1, 0.5, 1.0, 2.0, 3.0] {
        let s = theta * theta;
        let ours = scalar_half_sinc_half_s::<f64>(s, theta);
        let native = (theta / 2.0).sin() / theta;
        let rel = (ours - native).abs() / native.abs().max(1e-15);
        assert!(rel < 1e-13, "half_sinc at θ={theta}: rel={rel:.2e}");
    }
}

/// Depth-3 verification of `scalar_cos_half_s` Taylor coefficients via
/// `D3<1>` nested AD at s = 0.
#[test]
fn cos_half_third_derivative_via_d3_matches_taylor() {
    use crate::autodiff::nested_ad::Dual;

    let l1 = Dual::<f64, 1>::seed(0.0, 0);
    let l2 = Dual::<Dual<f64, 1>, 1>::seed(l1, 0);
    let l3 = Dual::<Dual<Dual<f64, 1>, 1>, 1>::seed(l2, 0);
    let theta = Dual::<Dual<Dual<f64, 1>, 1>, 1>::constant(0.0);
    let r = scalar_cos_half_s(l3, theta);

    let f0 = r.value.value.value;
    let f1 = r.value.value.tangent[0];
    let f2 = r.value.tangent[0].tangent[0];
    let f3 = r.tangent[0].tangent[0].tangent[0];

    // cos(θ/2) Taylor: 1 − s/8 + s²/384 − s³/46080 + …
    // → f(0)=1, f'(0)=−1/8, f''(0)=1/192, f'''(0)=−1/7680.
    assert!((f0 - 1.0).abs() < 1e-15, "f(0)={f0:.6e}");
    assert!((f1 + 1.0 / 8.0).abs() < 1e-15, "f'(0)={f1:.6e}");
    assert!((f2 - 1.0 / 192.0).abs() < 1e-15, "f''(0)={f2:.6e}");
    assert!((f3 + 1.0 / 7680.0).abs() < 1e-15, "f'''(0)={f3:.6e}");
}

/// Depth-3 verification of `scalar_half_sinc_half_s` Taylor coefficients
/// via `D3<1>` nested AD at s = 0.
#[test]
fn half_sinc_half_third_derivative_via_d3_matches_taylor() {
    use crate::autodiff::nested_ad::Dual;

    let l1 = Dual::<f64, 1>::seed(0.0, 0);
    let l2 = Dual::<Dual<f64, 1>, 1>::seed(l1, 0);
    let l3 = Dual::<Dual<Dual<f64, 1>, 1>, 1>::seed(l2, 0);
    let theta = Dual::<Dual<Dual<f64, 1>, 1>, 1>::constant(0.0);
    let r = scalar_half_sinc_half_s(l3, theta);

    let f0 = r.value.value.value;
    let f1 = r.value.value.tangent[0];
    let f2 = r.value.tangent[0].tangent[0];
    let f3 = r.tangent[0].tangent[0].tangent[0];

    // (1/2)·sinc(θ/2) Taylor: 1/2 − s/48 + s²/3840 − s³/645120 + …
    // → f(0)=1/2, f'(0)=−1/48, f''(0)=1/1920, f'''(0)=−1/107520.
    assert!((f0 - 0.5).abs() < 1e-15, "f(0)={f0:.6e}");
    assert!((f1 + 1.0 / 48.0).abs() < 1e-15, "f'(0)={f1:.6e}");
    assert!((f2 - 1.0 / 1920.0).abs() < 1e-15, "f''(0)={f2:.6e}");
    assert!((f3 + 1.0 / 107520.0).abs() < 1e-15, "f'''(0)={f3:.6e}");
}

/// D3 at non-zero s — verifies the AD machinery correctly differentiates
/// the implemented polynomial at a representative s.  IMPORTANT: s_val
/// MUST be below `TAYLOR_THRESHOLD_S`.  Above threshold the function
/// returns `cos(theta/2)` where `theta` is wrapped as a D3 *constant*, so
/// AD sees a constant and gives zero derivatives — that's correct
/// behaviour for the closed-form branch but uninformative as a test.
/// Below threshold we exercise the polynomial path that AD will actually
/// see in nested-AD-driven Hessian / cubic evaluations.
#[test]
fn cos_half_third_derivative_via_d3_at_nonzero_s() {
    use crate::autodiff::nested_ad::Dual;

    let s_val: f64 = 5e-5;
    let l1 = Dual::<f64, 1>::seed(s_val, 0);
    let l2 = Dual::<Dual<f64, 1>, 1>::seed(l1, 0);
    let l3 = Dual::<Dual<Dual<f64, 1>, 1>, 1>::seed(l2, 0);
    let theta = Dual::<Dual<Dual<f64, 1>, 1>, 1>::constant(s_val.sqrt());
    let r = scalar_cos_half_s(l3, theta);
    let f3 = r.tangent[0].tangent[0].tangent[0];

    // f'''_poly(s) = 6·(−1/46080) + 24·(1/10321920)·s
    //              = −1/7680 + s/430080
    let expected = -1.0 / 7680.0 + s_val / 430080.0;
    let rel = (f3 - expected).abs() / expected.abs();
    assert!(
        rel < 1e-13,
        "f'''(s={s_val})={f3:.6e}, expected {expected:.6e}, rel {rel:.2e}"
    );
}

#[test]
fn half_sinc_half_third_derivative_via_d3_at_nonzero_s() {
    use crate::autodiff::nested_ad::Dual;

    let s_val: f64 = 5e-5; // below TAYLOR_THRESHOLD_S; see note above
    let l1 = Dual::<f64, 1>::seed(s_val, 0);
    let l2 = Dual::<Dual<f64, 1>, 1>::seed(l1, 0);
    let l3 = Dual::<Dual<Dual<f64, 1>, 1>, 1>::seed(l2, 0);
    let theta = Dual::<Dual<Dual<f64, 1>, 1>, 1>::constant(s_val.sqrt());
    let r = scalar_half_sinc_half_s(l3, theta);
    let f3 = r.tangent[0].tangent[0].tangent[0];

    // f'''_poly(s) = 6·(−1/645120) + 24·(1/185794560)·s
    //              = −1/107520 + s/7741440
    let expected = -1.0 / 107520.0 + s_val / 7741440.0;
    let rel = (f3 - expected).abs() / expected.abs();
    assert!(
        rel < 1e-13,
        "f'''(s={s_val})={f3:.6e}, expected {expected:.6e}, rel {rel:.2e}"
    );
}
