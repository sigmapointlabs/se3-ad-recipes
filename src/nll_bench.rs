//! Test/bench support: AD-generic SE(3) NLL and three Hessian routines.
//!
//! Hosts the production-basis NLL machinery from the SE3_ad_letter §VI / Table II
//! experiments so it can be reused by both the in-tree tests (`nll_tests.rs`)
//! and the criterion benchmark (`benches/nll_hessian.rs`). Bench targets are
//! external crates from the library's perspective, so anything they need must
//! be `pub`.
//!
//! The naïve θ-Taylor basis is intentionally kept private to `nll_tests.rs` —
//! it has no use outside the four-way comparison test.

use crate::Vec6;
use crate::autodiff::ad_trait::AD;
use crate::autodiff::forward_ad::adfn;
use crate::autodiff::nested_ad::{D2, Dual};
use crate::projective;
use crate::se3_adsafe::{Mat6G, PoseG, Vec6G, adjoint_g, pose_to_g, se3_jr_g, se3_jr_inv_g};
use crate::se3_unsafe::Pose;
use crate::so3_adsafe::{
    Mat3G, Vec3G, add_mat3_g, hat_g, i3_g, mm3_g, mv3_g, scale_mat3_g, sub_mat3_g,
    theta_sq_from_omega, trace3_g, transpose3_g,
};

// =====================================================================
// SE(3) Exp/Log basis trait — production (fixed) impl.
// =====================================================================

/// Strategy for SE(3) Exp/Log.  Lets tests/benches swap in alternative
/// scalar bases (e.g. the naïve θ-Taylor used by the four-way trap test).
pub trait SE3Basis {
    fn pose_exp<T: AD>(xi: &Vec6G<T>) -> PoseG<T>;
    fn pose_log<T: AD>(p: &PoseG<T>) -> Vec6G<T>;
}

/// Production basis: `scalar_*_s` (degree-4 Taylor in s = θ²).
pub struct FixedBasis;
impl SE3Basis for FixedBasis {
    fn pose_exp<T: AD>(xi: &Vec6G<T>) -> PoseG<T> {
        PoseG::exp(xi)
    }
    fn pose_log<T: AD>(p: &PoseG<T>) -> Vec6G<T> {
        p.log()
    }
}

// ---------------------------------------------------------------------
// Naïve θ-Taylor basis (research/diagnostic use only).
//
// The s-zero branch returns `θ = const(0)` (no tangent), so a θ-based
// Taylor `1 − θ²/6` evaluated against `θ = const(0)` is *constant 1*
// with all higher tangents zero — the polynomial-depletion trap of
// §IV.A.  Used by `nll_hessian_four_ways_table_ii` and the Table II
// benchmark to demonstrate that nested-AD cost is independent of the
// scalar basis (depletion only changes results, not work done).
// ---------------------------------------------------------------------

fn naive_a<T: AD>(s: T, theta: T) -> T {
    if s.to_constant() < 1e-20 {
        T::constant(1.0) - theta * theta * T::constant(1.0 / 6.0)
    } else {
        theta.sin() / theta
    }
}
fn naive_b<T: AD>(s: T, theta: T) -> T {
    if s.to_constant() < 1e-20 {
        T::constant(0.5) - theta * theta * T::constant(1.0 / 24.0)
    } else {
        (T::constant(1.0) - theta.cos()) / s
    }
}
fn naive_c<T: AD>(s: T, theta: T) -> T {
    if s.to_constant() < 1e-20 {
        T::constant(1.0 / 6.0) - theta * theta * T::constant(1.0 / 120.0)
    } else {
        (theta - theta.sin()) / (s * theta)
    }
}
fn naive_d<T: AD>(s: T, theta: T) -> T {
    if s.to_constant() < 1e-20 {
        T::constant(1.0 / 12.0) + theta * theta * T::constant(1.0 / 720.0)
    } else {
        let half = T::constant(0.5);
        let half_theta = half * theta;
        let cot_half = half_theta.cos() / half_theta.sin();
        (T::constant(1.0) - half_theta * cot_half) / s
    }
}

fn naive_so3_exp<T: AD>(omega: &Vec3G<T>) -> Mat3G<T> {
    let (s, theta) = theta_sq_from_omega(omega);
    let a = naive_a(s, theta);
    let b = naive_b(s, theta);
    let h = hat_g(omega);
    let h2 = mm3_g(&h, &h);
    add_mat3_g(
        &add_mat3_g(&i3_g(), &scale_mat3_g(a, &h)),
        &scale_mat3_g(b, &h2),
    )
}
fn naive_v_matrix<T: AD>(omega: &Vec3G<T>) -> Mat3G<T> {
    let (s, theta) = theta_sq_from_omega(omega);
    let b = naive_b(s, theta);
    let c = naive_c(s, theta);
    let h = hat_g(omega);
    let h2 = mm3_g(&h, &h);
    add_mat3_g(
        &add_mat3_g(&i3_g(), &scale_mat3_g(b, &h)),
        &scale_mat3_g(c, &h2),
    )
}
fn naive_v_inv<T: AD>(omega: &Vec3G<T>) -> Mat3G<T> {
    let (s, theta) = theta_sq_from_omega(omega);
    let d = naive_d(s, theta);
    let h = hat_g(omega);
    let h2 = mm3_g(&h, &h);
    add_mat3_g(
        &add_mat3_g(&i3_g(), &scale_mat3_g(T::constant(-0.5), &h)),
        &scale_mat3_g(d, &h2),
    )
}
fn naive_so3_log<T: AD>(r: &Mat3G<T>) -> Vec3G<T> {
    let cos_theta = T::constant(0.5) * (trace3_g(r) - T::constant(1.0));
    if cos_theta.to_constant() > 1.0 - 1e-10 {
        let rt = transpose3_g(r);
        let skew = sub_mat3_g(r, &rt);
        let half = T::constant(0.5);
        [half * skew[2][1], half * skew[0][2], half * skew[1][0]]
    } else {
        let theta = cos_theta.acos();
        let factor = theta / (T::constant(2.0) * theta.sin());
        let rt = transpose3_g(r);
        let skew = sub_mat3_g(r, &rt);
        [
            factor * skew[2][1],
            factor * skew[0][2],
            factor * skew[1][0],
        ]
    }
}

/// Naïve θ-Taylor basis — see module-level note above.
pub struct NaiveBasis;
impl SE3Basis for NaiveBasis {
    fn pose_exp<T: AD>(xi: &Vec6G<T>) -> PoseG<T> {
        let omega: Vec3G<T> = [xi[0], xi[1], xi[2]];
        let t: Vec3G<T> = [xi[3], xi[4], xi[5]];
        let rot = naive_so3_exp(&omega);
        let v = naive_v_matrix(&omega);
        let trans = mv3_g(&v, &t);
        PoseG { rot, trans }
    }
    fn pose_log<T: AD>(p: &PoseG<T>) -> Vec6G<T> {
        let omega = naive_so3_log(&p.rot);
        let vi = naive_v_inv(&omega);
        let t = mv3_g(&vi, &p.trans);
        [omega[0], omega[1], omega[2], t[0], t[1], t[2]]
    }
}

// =====================================================================
// AD-generic NLL cost (Gaussian pose prior + pseudo-Huber landmarks).
// =====================================================================

/// Pseudo-Huber kernel ρ(s) = κ²·(√(1 + s/κ²) − 1).
pub fn pseudo_huber<T: AD>(s: T, kappa_sq: T) -> T {
    let one = T::constant(1.0);
    let inv_k2 = one / kappa_sq;
    kappa_sq * ((one + s * inv_k2).sqrt() - one)
}

/// Pinhole projection π(x') = (x'₀/x'₂, x'₁/x'₂).
pub fn project_g<T: AD>(xp: &Vec3G<T>) -> [T; 2] {
    let inv_z = T::constant(1.0) / xp[2];
    [xp[0] * inv_z, xp[1] * inv_z]
}

/// One landmark term: ρ(‖L·(π(T·x) − z)‖²).
pub fn landmark_term_g<T: AD>(
    pose: &PoseG<T>,
    x_world: &Vec3G<T>,
    z: &[T; 2],
    l_white: &[[T; 2]; 2],
    kappa_sq: T,
) -> T {
    let xp = [
        pose.rot[0][0] * x_world[0]
            + pose.rot[0][1] * x_world[1]
            + pose.rot[0][2] * x_world[2]
            + pose.trans[0],
        pose.rot[1][0] * x_world[0]
            + pose.rot[1][1] * x_world[1]
            + pose.rot[1][2] * x_world[2]
            + pose.trans[1],
        pose.rot[2][0] * x_world[0]
            + pose.rot[2][1] * x_world[1]
            + pose.rot[2][2] * x_world[2]
            + pose.trans[2],
    ];
    let pi = project_g(&xp);
    let r = [pi[0] - z[0], pi[1] - z[1]];
    let rw = [
        l_white[0][0] * r[0] + l_white[0][1] * r[1],
        l_white[1][0] * r[0] + l_white[1][1] * r[1],
    ];
    let s = rw[0] * rw[0] + rw[1] * rw[1];
    pseudo_huber(s, kappa_sq)
}

/// Gaussian prior term: ½ ξ_errᵀ Σ⁻¹ ξ_err  with  ξ_err = log(pose⁻¹·prior).
pub fn prior_term_g<B: SE3Basis, T: AD>(
    pose: &PoseG<T>,
    t_prior: &PoseG<T>,
    sigma_inv_sq_diag: &[T; 6],
) -> T {
    let rel = pose.inverse().compose(t_prior);
    let xi_err = B::pose_log(&rel);
    let mut acc = T::constant(0.0);
    for i in 0..6 {
        acc += xi_err[i] * xi_err[i] * sigma_inv_sq_diag[i];
    }
    acc * T::constant(0.5)
}

/// Total NLL: prior + pseudo-Huber landmark terms.  pose = base · exp(δ).
pub fn nll_g<B: SE3Basis, T: AD>(
    delta: &Vec6G<T>,
    base: &PoseG<T>,
    t_prior: &PoseG<T>,
    sigma_inv_sq_diag: &[T; 6],
    landmarks: &[(Vec3G<T>, [T; 2], [[T; 2]; 2])],
    kappa_sq: T,
) -> T {
    let pose = base.compose(&B::pose_exp(delta));
    let mut cost = prior_term_g::<B, T>(&pose, t_prior, sigma_inv_sq_diag);
    for (x, z, l) in landmarks.iter() {
        cost += landmark_term_g(&pose, x, z, l, kappa_sq);
    }
    cost
}

// =====================================================================
// Problem construction.
// =====================================================================

/// A representative SE(3) NLL evaluation point: pose with prior, landmarks,
/// observation noise, and the pseudo-Huber threshold κ.
pub struct Problem {
    pub base: Pose,
    pub t_prior: Pose,
    pub sigma_inv_sq_diag: [f64; 6],
    pub landmarks: Vec<([f64; 3], [f64; 2], [[f64; 2]; 2])>,
    pub kappa: f64,
}

/// Five-landmark monocular pose-with-prior problem at depths 3..7.
pub fn build_problem(kappa: f64) -> Problem {
    let base = Pose::exp(&[0.30, -0.20, 0.40, 0.50, -0.30, 0.70]);
    let prior_offset: Vec6 = [0.05, -0.03, 0.04, 0.10, -0.06, 0.08];
    let t_prior = base.compose(&Pose::exp(&prior_offset));

    let s_rot = 1.0 / (0.1f64 * 0.1);
    let s_t = 1.0 / (0.2f64 * 0.2);
    let sigma_inv_sq_diag = [s_rot, s_rot, s_rot, s_t, s_t, s_t];

    let xs: [[f64; 3]; 5] = [
        [0.5, 0.2, 4.0],
        [-0.7, 0.3, 5.5],
        [0.1, -0.6, 3.0],
        [0.8, 0.5, 7.0],
        [-0.4, -0.2, 6.0],
    ];
    let z_offset = 0.005;
    let mut landmarks = Vec::with_capacity(xs.len());
    for (i, x) in xs.iter().enumerate() {
        let xp = projective::transform_point(&base.rot, &base.trans, x);
        let pi = projective::project(&xp);
        let sgn = if i % 2 == 0 { 1.0 } else { -1.0 };
        let z = [pi[0] + sgn * z_offset, pi[1] - sgn * z_offset];
        let l_white = [[1.0 / 0.003, 0.0], [0.0, 1.0 / 0.0035]];
        landmarks.push((*x, z, l_white));
    }

    Problem {
        base,
        t_prior,
        sigma_inv_sq_diag,
        landmarks,
        kappa,
    }
}

/// Lift an `f64` problem instance into AD scalar `T`.
pub fn lift_problem<T: AD>(
    p: &Problem,
) -> (
    PoseG<T>,
    PoseG<T>,
    [T; 6],
    Vec<(Vec3G<T>, [T; 2], [[T; 2]; 2])>,
    T,
) {
    let base_g = pose_to_g::<T>(&p.base);
    let prior_g = pose_to_g::<T>(&p.t_prior);
    let sig: [T; 6] = std::array::from_fn(|i| T::constant(p.sigma_inv_sq_diag[i]));
    let lms: Vec<(Vec3G<T>, [T; 2], [[T; 2]; 2])> = p
        .landmarks
        .iter()
        .map(|(x, z, l)| {
            let xg: Vec3G<T> = [T::constant(x[0]), T::constant(x[1]), T::constant(x[2])];
            let zg: [T; 2] = [T::constant(z[0]), T::constant(z[1])];
            let lg: [[T; 2]; 2] = [
                [T::constant(l[0][0]), T::constant(l[0][1])],
                [T::constant(l[1][0]), T::constant(l[1][1])],
            ];
            (xg, zg, lg)
        })
        .collect();
    let kappa_sq = T::constant(p.kappa * p.kappa);
    (base_g, prior_g, sig, lms, kappa_sq)
}

// =====================================================================
// NLL value, gradient, Hessian.
// =====================================================================

/// Plain `f64` evaluation.
pub fn nll_f64<B: SE3Basis>(p: &Problem, delta: &Vec6) -> f64 {
    let (base_g, prior_g, sig, lms, k2) = lift_problem::<f64>(p);
    nll_g::<B, f64>(delta, &base_g, &prior_g, &sig, &lms, k2)
}

/// Gradient via single-seed `adfn<6>`.  This is the **mixed-AD** workhorse:
/// each scalar op carries one f64 value plus one 6-vector tangent.
pub fn nll_gradient<B: SE3Basis>(p: &Problem, delta: &Vec6) -> Vec6 {
    let (base_g, prior_g, sig, lms, k2) = lift_problem::<adfn<6>>(p);
    let delta_ad: Vec6G<adfn<6>> = std::array::from_fn(|i| {
        let mut t = [0.0; 6];
        t[i] = 1.0;
        adfn::new(delta[i], t)
    });
    let r = nll_g::<B, adfn<6>>(&delta_ad, &base_g, &prior_g, &sig, &lms, k2);
    r.tangent()
}

/// (1) Hessian by central FD of the scalar value.  4-point stencil off-diagonal.
pub fn hessian_fd_value<B: SE3Basis>(p: &Problem, h: f64) -> [[f64; 6]; 6] {
    let mut h_mat = [[0.0f64; 6]; 6];
    let f0 = nll_f64::<B>(p, &[0.0; 6]);
    for i in 0..6 {
        let mut dp = [0.0; 6];
        dp[i] = h;
        let mut dm = [0.0; 6];
        dm[i] = -h;
        let fp = nll_f64::<B>(p, &dp);
        let fm = nll_f64::<B>(p, &dm);
        h_mat[i][i] = (fp - 2.0 * f0 + fm) / (h * h);
    }
    for i in 0..6 {
        for j in (i + 1)..6 {
            let mut dpp = [0.0; 6];
            dpp[i] = h;
            dpp[j] = h;
            let mut dpm = [0.0; 6];
            dpm[i] = h;
            dpm[j] = -h;
            let mut dmp = [0.0; 6];
            dmp[i] = -h;
            dmp[j] = h;
            let mut dmm = [0.0; 6];
            dmm[i] = -h;
            dmm[j] = -h;
            let v = (nll_f64::<B>(p, &dpp) - nll_f64::<B>(p, &dpm) - nll_f64::<B>(p, &dmp)
                + nll_f64::<B>(p, &dmm))
                / (4.0 * h * h);
            h_mat[i][j] = v;
            h_mat[j][i] = v;
        }
    }
    h_mat
}

/// (2) Hessian by central FD of the AD gradient — **mixed-AD** path:
/// `2·6 = 12` `adfn<6>` gradient evaluations plus an outer FD layer.
pub fn hessian_fd_grad<B: SE3Basis>(p: &Problem, h: f64) -> [[f64; 6]; 6] {
    let mut h_mat = [[0.0f64; 6]; 6];
    for i in 0..6 {
        let mut dp = [0.0; 6];
        dp[i] = h;
        let mut dm = [0.0; 6];
        dm[i] = -h;
        let gp = nll_gradient::<B>(p, &dp);
        let gm = nll_gradient::<B>(p, &dm);
        for j in 0..6 {
            h_mat[j][i] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    h_mat
}

/// (4) Hessian by depth-2 nested AD — **pure nested-AD** path: a single
/// `D2<6>` evaluation; every scalar op carries a 6-vector first tangent
/// plus a 6×6 second-order block.  Returns both extraction orders so callers
/// can verify Schwarz symmetry separately.
pub fn hessian_d2<B: SE3Basis>(p: &Problem) -> ([[f64; 6]; 6], [[f64; 6]; 6]) {
    let (base_g, prior_g, sig, lms, k2) = lift_problem::<D2<6>>(p);
    let delta: Vec6G<D2<6>> = std::array::from_fn(|i| {
        let inner = Dual::<f64, 6>::seed(0.0, i);
        D2::<6>::seed(inner, i)
    });
    let r = nll_g::<B, D2<6>>(&delta, &base_g, &prior_g, &sig, &lms, k2);
    let mut h_qr = [[0.0f64; 6]; 6];
    let mut h_rq = [[0.0f64; 6]; 6];
    for q in 0..6 {
        for r_idx in 0..6 {
            h_qr[q][r_idx] = r.tangent[q].tangent[r_idx];
            h_rq[q][r_idx] = r.tangent[r_idx].tangent[q];
        }
    }
    (h_qr, h_rq)
}

/// Hessian via **automatic forward-over-reverse** (no analytical gradient!).
///
/// Path:
///   * outer  = single forward AD with `adfn<6>` over the input δ
///   * inner  = reverse-mode tape (`adr_n6`) whose **partials are `adfn<6>`**,
///     so the backward sweep transports the outer tangent through
///     the gradient computation.
///
/// One forward pass + one reverse pass = exact Hessian, machine precision,
/// no FD step.  Per-op cost is `O(n)` (one `adfn<6>` worth of arithmetic
/// on each tape entry), in contrast to nested forward-over-forward
/// (`hessian_d2`) which costs `O(n²)` per op.  This is the textbook
/// `jax.hessian = jacfwd(jacrev(f))` recipe, instantiated at fixed n=6.
pub fn hessian_for<B: SE3Basis>(p: &Problem) -> [[f64; 6]; 6] {
    use crate::autodiff::reverse_ad_n6::{adr_n6, tape_n6_backward_into, tape_n6_clear};
    use std::cell::UnsafeCell;

    // Thread-local adjoint scratch reused across calls.  `tape_n6_backward_into`
    // resizes-and-fills it in place, so steady-state Hessian evaluations
    // pay no per-call ~6 KB malloc/free for the adjoint buffer.  Same
    // single-writer invariant as the tape itself: the `&mut` borrow lives
    // only for the duration of one backward sweep.
    thread_local! {
        static ADJ_SCRATCH: UnsafeCell<Vec<adfn<6>>> = UnsafeCell::new(Vec::with_capacity(4096));
    }

    tape_n6_clear();

    // Seed δ = 0 with adfn<6> directional tangents.  Each input becomes an
    // `adr_n6` whose primal is `(0, e_i)` — the tape now has the outer
    // forward tangent baked into every recorded partial.
    let delta_inputs: Vec6G<adr_n6> = std::array::from_fn(|i| {
        let mut t = [0.0f64; 6];
        t[i] = 1.0;
        adr_n6::new_input(adfn::<6>::new(0.0, t))
    });
    let input_indices: [usize; 6] = std::array::from_fn(|i| delta_inputs[i].index());

    // Forward pass — runs the AD-generic NLL with `T = adr_n6`, building
    // the reverse tape with `adfn<6>` partials.
    let (base, prior, sig, lms, k2) = lift_problem::<adr_n6>(p);
    let result = nll_g::<B, adr_n6>(&delta_inputs, &base, &prior, &sig, &lms, k2);

    // Reverse sweep into the thread-local scratch — adjoints come back as
    // `adfn<6>`; each input's adjoint tangent is one row of the Hessian.
    let mut h = [[0.0f64; 6]; 6];
    ADJ_SCRATCH.with(|s| {
        // SAFETY: single writer; the `&mut` lives only for this block.
        let adj = unsafe { &mut *s.get() };
        tape_n6_backward_into(result.index(), adj);
        for i in 0..6 {
            let t = adj[input_indices[i]].tangent();
            for k in 0..6 {
                h[i][k] = t[k];
            }
        }
    });
    h
}

// =====================================================================
// Closed-form analytical NLL gradient — used by the "AD of seeded
// gradient (fused basis)" Table II row.
// =====================================================================
//
// Direct chain-rule construction with no AD on the cost:
//
//   prior:    g = -Ad(rel⁻¹)ᵀ · Jr⁻¹(ξ_err)ᵀ · Σ⁻¹ · ξ_err
//             where rel = pose⁻¹·t_prior and ξ_err = log(rel).
//
//   landmark: g = 2·ρ′(s) · J_×(x)ᵀ·Rᵀ·Pᵀ·Lᵀ·L·r
//             where xp = pose·x, r = π(xp) − z, rw = L·r, s = ‖rw‖²,
//             P  = ∂π/∂xp (analytical, project_jacobian_g),
//             J_× = [-[x]×, I_3] (constant in x).
//
// Returning Vec6G<T> means callers can drop in T = f64 (FD outer layer)
// or T = adfn<6> (single AD outer layer → exact Hessian, one eval).

// ─────────────────────────────────────────────────────────────────────
// Analytical SE(3) NLL gradient (paper §VII.b: analytical Lie-group
// Jacobian + flat-space chain rule, no AD on the cost).
//
// Per the chain-rule split in eq. (14) of the SE3_ad_letter, the only
// piece that requires Lie-group calculus is the **point-action Jacobian**
//
//     J_act = ∂(T·x)/∂δ  =  R · J_×(x)         with  J_×(x) = [−[x]×, I]
//
// Everything past `y = T·x` (projection, whitening, robust kernel) is
// flat-space scalar arithmetic that's cheap to write out by hand.
// We compute `∇_y r` directly via the closed-form chain
//
//     ∇_y r = 2 ρ′(s) · (∂_y π)ᵀ · Lᵀ · L · (π(y) − z),
//
// then pre-multiply by `J_actᵀ = J_×(x)ᵀ · Rᵀ` (last factor uses
// J_×(x)ᵀ = [[x]× ; I] in two short lines).  Returning Vec6G<T> means
// callers can drop in T = f64 (FD outer layer) or T = adfn<6> (single
// AD outer layer → exact Hessian, one eval).
// ─────────────────────────────────────────────────────────────────────

/// y = R·x + t, AD-generic.
fn pose_act_g<T: AD>(pose: &PoseG<T>, x: &Vec3G<T>) -> Vec3G<T> {
    let rx = mv3_g(&pose.rot, x);
    [
        rx[0] + pose.trans[0],
        rx[1] + pose.trans[1],
        rx[2] + pose.trans[2],
    ]
}

/// 6×6 transposed-matrix multiply: `r[i] = Σⱼ M[j][i]·v[j]`.
fn mtv6_g<T: AD>(m: &Mat6G<T>, v: &Vec6G<T>) -> Vec6G<T> {
    let z = T::constant(0.0);
    let mut r = [z; 6];
    for i in 0..6 {
        let mut s = z;
        for j in 0..6 {
            s += m[j][i] * v[j];
        }
        r[i] = s;
    }
    r
}

/// ∇_y r for one whitened pseudo-Huber reprojection residual at the
/// camera-frame point `y` — closed-form flat-space chain rule.
///
///   r(y)    = ρ(‖L·(π(y) − z)‖²),     ρ(s) = κ²·(√(1+s/κ²) − 1)
///   ∇_y r   = 2 ρ′(s) · (∂_y π)ᵀ · Lᵀ · L · (π(y) − z)
///   (∂_y π) = (1/y₃)·[[1, 0, −u], [0, 1, −v]]   with (u, v) = π(y)
fn landmark_grad_y<T: AD>(y: &Vec3G<T>, z: &[T; 2], l: &[[T; 2]; 2], kappa_sq: T) -> Vec3G<T> {
    let inv_z = T::constant(1.0) / y[2];
    let u = y[0] * inv_z;
    let v = y[1] * inv_z;
    let r0 = u - z[0];
    let r1 = v - z[1];
    let rw0 = l[0][0] * r0 + l[0][1] * r1;
    let rw1 = l[1][0] * r0 + l[1][1] * r1;
    let s = rw0 * rw0 + rw1 * rw1;
    let rho_prime = T::constant(0.5) / (T::constant(1.0) + s / kappa_sq).sqrt();
    // q = Lᵀ·rw  (= Lᵀ·L·r);  ∇_y r = 2 ρ′ · (∂_y π)ᵀ · q
    let q0 = l[0][0] * rw0 + l[1][0] * rw1;
    let q1 = l[0][1] * rw0 + l[1][1] * rw1;
    let two_rp_inv_z = T::constant(2.0) * rho_prime * inv_z;
    [
        two_rp_inv_z * q0,
        two_rp_inv_z * q1,
        -two_rp_inv_z * (u * q0 + v * q1),
    ]
}

/// Analytical NLL gradient: closed-form chain rule, no AD on the cost.
///
/// Generic in `T: AD` so `T = f64` (for FD-of-analytical-gradient) and
/// `T = adfn<6>` (for AD-of-seeded-gradient, single-eval Hessian) both work.
///
/// Returns ∂Cost(δ)/∂δ where Cost(δ) = cost(base·exp(δ)).  Internally
/// builds the local right-perturbation gradient `g_local(δ)` at `pose(δ)`
/// — using analytical Jr⁻¹/Ad for the prior and analytical J_act for
/// landmarks — and multiplies by `Jr(δ)ᵀ` so outer AD picks up the
/// (∂Jr/∂δ)ᵀ·g term that turns g_local's Jacobian into the true Hessian.
pub fn nll_gradient_analytical_g<T: AD>(p: &Problem, delta: &Vec6G<T>) -> Vec6G<T> {
    let base_g: PoseG<T> = pose_to_g(&p.base);
    let prior_g: PoseG<T> = pose_to_g(&p.t_prior);
    let pose: PoseG<T> = base_g.compose(&PoseG::exp(delta));

    let z = T::constant(0.0);
    let mut g_local: Vec6G<T> = [z; 6];

    // Prior: g_local += −Ad(rel⁻¹)ᵀ · Jr⁻¹(ξ)ᵀ · Σ⁻¹·ξ.
    let rel = pose.inverse().compose(&prior_g);
    let xi = rel.log();
    let mut s_xi: Vec6G<T> = [z; 6];
    for i in 0..6 {
        s_xi[i] = T::constant(p.sigma_inv_sq_diag[i]) * xi[i];
    }
    let jri_t_v = mtv6_g(&se3_jr_inv_g(&xi), &s_xi);
    let rel_inv = rel.inverse();
    let ad_t_v = mtv6_g(&adjoint_g(&rel_inv.rot, &rel_inv.trans), &jri_t_v);
    for i in 0..6 {
        g_local[i] -= ad_t_v[i];
    }

    // Landmarks: g_local += J_×(x)ᵀ · Rᵀ · ∇_y r,  with J_×(x)ᵀ·w = [x×w; w].
    let kappa_sq = T::constant(p.kappa * p.kappa);
    let rt = transpose3_g(&pose.rot);
    for (x, z_arr, l_arr) in p.landmarks.iter() {
        let xg: Vec3G<T> = std::array::from_fn(|i| T::constant(x[i]));
        let za: [T; 2] = [T::constant(z_arr[0]), T::constant(z_arr[1])];
        let lw: [[T; 2]; 2] = [
            [T::constant(l_arr[0][0]), T::constant(l_arr[0][1])],
            [T::constant(l_arr[1][0]), T::constant(l_arr[1][1])],
        ];
        let y = pose_act_g(&pose, &xg);
        let grad_y = landmark_grad_y(&y, &za, &lw, kappa_sq);
        let w3 = mv3_g(&rt, &grad_y); // Rᵀ · ∇_y r
        g_local[0] += xg[1] * w3[2] - xg[2] * w3[1];
        g_local[1] += xg[2] * w3[0] - xg[0] * w3[2];
        g_local[2] += xg[0] * w3[1] - xg[1] * w3[0];
        g_local[3] += w3[0];
        g_local[4] += w3[1];
        g_local[5] += w3[2];
    }

    // Apply the parameterization Jacobian: ∂Cost/∂δ = Jr(δ)ᵀ · g_local.
    mtv6_g(&se3_jr_g(delta), &g_local)
}

/// Hessian via FD of the analytical gradient (12 analytical-gradient evals).
/// Pure f64 — no AD at all.
pub fn hessian_fd_analytical_grad(p: &Problem, h: f64) -> [[f64; 6]; 6] {
    let mut h_mat = [[0.0f64; 6]; 6];
    for i in 0..6 {
        let mut dp = [0.0; 6];
        dp[i] = h;
        let mut dm = [0.0; 6];
        dm[i] = -h;
        let gp = nll_gradient_analytical_g::<f64>(p, &dp);
        let gm = nll_gradient_analytical_g::<f64>(p, &dm);
        for j in 0..6 {
            h_mat[j][i] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    h_mat
}

/// Hessian via single forward AD of the analytical gradient — **the
/// canonical mixed-AD path**: one `adfn<6>` eval, no FD, no nested duals.
pub fn hessian_ad_of_analytical_grad(p: &Problem) -> [[f64; 6]; 6] {
    let delta: Vec6G<adfn<6>> = std::array::from_fn(|i| {
        let mut t = [0.0; 6];
        t[i] = 1.0;
        adfn::new(0.0, t)
    });
    let g = nll_gradient_analytical_g::<adfn<6>>(p, &delta);
    let mut h_mat = [[0.0f64; 6]; 6];
    for i in 0..6 {
        let row = g[i].tangent();
        for j in 0..6 {
            h_mat[i][j] = row[j];
        }
    }
    h_mat
}
