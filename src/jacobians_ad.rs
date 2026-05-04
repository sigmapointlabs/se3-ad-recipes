//! # AD-generic SE(3) derivative tensors
//!
//! The chart-Hessian piece of the Lie-group recipe.  All primitives have
//! been promoted to two dedicated modules:
//!
//! * [`crate::so3_adsafe`] — `<T: AD>` SO(3) scalars (A, B, C, D, β),
//!   fused atoms (β̄, β̄', D̃·ω_m), exp/log/V/V⁻¹/Jr/Jr⁻¹, and the hat
//!   basis E_m.
//! * [`crate::se3_adsafe`] — `PoseG`, adjoint, `Q̃_r`, and the SE(3)
//!   right Jacobian and its inverse.
//!
//! What lives here is just the derivative-tensor scaffolding that
//! consumes those atoms:
//!
//! * `djr_inv_slice_g`, `dqr_omega_slice_g`, `dqr_t_slice_g` — slabs of
//!   `∂Jr⁻¹/∂ω_m`, `∂Q̃_r/∂ω_m`, `∂Q̃_r/∂t_m`.
//! * `se3_jr_inv_derivative_g`, `se3_jr_derivative_g` — assembled 6×6×6
//!   tensors of `∂(Jr^SE3)⁻¹/∂ξ` and `∂Jr^SE3/∂ξ`.
//! * `recentering_hessian_g`, `recentering_hessian_at_g` — chart
//!   re-centering Hessian.
//! * `alpha_m_prime_fused` — the AD-safe rewrite of
//!   `∂[(ωᵀt) β̄(s)]/∂ω_m`, expressed as `t_m β̄(s) + 2 ω_m (ωᵀt) β̄'(s)`
//!   with no `1/s` or `1/θ` factors.

use crate::autodiff::ad_trait::AD;
use crate::se3_adsafe::{
    Mat6G, PoseG, Vec6G, blocks_6x6_g, mm6_g, q_tilde_r_g, se3_jr_g, se3_jr_inv_g,
};
use crate::so3_adsafe::{
    Mat3G, Vec3G, add_mat3_g, d_prime_omega_over_theta, dot3_g, hat_basis_g, hat_g, jr_g, mm3_g,
    scalar_beta_bar_prime_s, scalar_beta_over_s_g, scalar_d_s, scale_mat3_g, theta_sq_from_omega,
    z3_g,
};

// =========================================================================
// Fused SE(3) coupling derivative scalar
// =========================================================================

/// α_m' ≡ ∂[(ωᵀt) · β̄(s)] / ∂ω_m, the fused SE(3) coupling derivative
/// scalar that appears in `∂Q̃_r/∂ω_m`.
///
/// **AD-safe rewrite (paper §V.D recipe step 4).**  The conventional
/// form `β·(t_m·s − 2 ω_m wdot)/s² + β'·ω_m wdot/(s·θ)` carries a
/// removable `(β̃ − 2 β̄)/s` factor that AD cannot detect.  Apply the
/// product rule directly to `α = (ωᵀt) · β̄(s)`:
///
/// ```text
///     α_m' = t_m · β̄(s)  +  2 ω_m · (ωᵀt) · β̄'(s)
/// ```
///
/// Pure polynomial-in-s arithmetic — no division by `s`, no `1/θ`, no
/// branch threshold needed by α_m' itself (its sub-atoms β̄ and β̄' carry
/// the unified `s < 1e-4` cutoff internally).  AD-safe at every depth.
#[inline]
pub fn alpha_m_prime_fused<T: AD>(s: T, theta: T, omega_m: T, t_m: T, wdot_t: T) -> T {
    let beta_bar = scalar_beta_over_s_g(s, theta);
    let beta_bar_prime = scalar_beta_bar_prime_s(s, theta);
    t_m * beta_bar + T::constant(2.0) * omega_m * wdot_t * beta_bar_prime
}

// =========================================================================
// Derivative tensor slabs
// =========================================================================

/// AD-generic ∂Jr⁻¹/∂ω_m (Proposition 2 of the companion paper).
///
/// Takes s = θ² to enable fused singular-product evaluation at θ = 0.
pub fn djr_inv_slice_g<T: AD>(
    omega: &Vec3G<T>,
    hat_w_sq: &Mat3G<T>,
    s: T,
    theta: T,
    d: T,
    m: usize,
) -> Mat3G<T> {
    let em = hat_basis_g::<T>(m);
    let z = T::constant(0.0);

    // Fused product D'·ω_m/θ — smooth at θ = 0 (paper §V.D step 4).
    let radial_coeff = d_prime_omega_over_theta(s, theta, omega[m]);

    let mut result = [[z; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let em_wt = if i == m { omega[j] } else { z };
            let w_emt = if j == m { omega[i] } else { z };
            let kron = if i == j { T::constant(1.0) } else { z };
            result[i][j] = T::constant(0.5) * em[i][j]
                + radial_coeff * hat_w_sq[i][j]
                + d * (em_wt + w_emt - T::constant(2.0) * omega[m] * kron);
        }
    }
    result
}

/// AD-generic ∂Qr/∂ω_m (Proposition 3 of the companion paper).
///
/// Takes s = θ² for fused singular-product evaluation; the axial
/// scalar α and its derivative α_m' both flow through fused atoms.
#[allow(clippy::too_many_arguments)]
pub fn dqr_omega_slice_g<T: AD>(
    omega: &Vec3G<T>,
    t: &Vec3G<T>,
    hat_w: &Mat3G<T>,
    hat_w_sq: &Mat3G<T>,
    s: T,
    theta: T,
    d: T,
    m: usize,
) -> Mat3G<T> {
    let hat_t = hat_g(t);
    let em = hat_basis_g::<T>(m);
    let z = T::constant(0.0);

    let wt = mm3_g(hat_w, &hat_t);
    let tw = mm3_g(&hat_t, hat_w);
    let sym_wt = add_mat3_g(&wt, &tw);
    let em_ht = mm3_g(&em, &hat_t);
    let ht_em = mm3_g(&hat_t, &em);
    let wdot_t = dot3_g(omega, t);

    // Fused atoms (paper §V.D step 4) — smooth at θ = 0.
    let radial_coeff = d_prime_omega_over_theta(s, theta, omega[m]);
    let alpha_m_p = alpha_m_prime_fused(s, theta, omega[m], t[m], wdot_t);
    let alpha = wdot_t * scalar_beta_over_s_g(s, theta);

    let mut result = [[z; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let em_wt = if i == m { omega[j] } else { z };
            let w_emt = if j == m { omega[i] } else { z };
            let kron = if i == j { T::constant(1.0) } else { z };
            let hat_prod = em_wt + w_emt - T::constant(2.0) * omega[m] * kron;
            result[i][j] = radial_coeff * sym_wt[i][j]
                + d * (em_ht[i][j] + ht_em[i][j])
                + alpha_m_p * hat_w_sq[i][j]
                + alpha * hat_prod;
        }
    }
    result
}

/// AD-generic ∂Qr/∂t_m (Proposition 4 of the companion paper).
pub fn dqr_t_slice_g<T: AD>(
    omega: &Vec3G<T>,
    hat_w_sq: &Mat3G<T>,
    s: T,
    theta: T,
    d: T,
    m: usize,
) -> Mat3G<T> {
    let em = hat_basis_g::<T>(m);
    let z = T::constant(0.0);

    // axial_coeff = ω_m · β̄(s) — single fused scalar; smooth at s = 0.
    let axial_coeff = omega[m] * scalar_beta_over_s_g(s, theta);

    let mut result = [[z; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let em_wt = if i == m { omega[j] } else { z };
            let w_emt = if j == m { omega[i] } else { z };
            let kron = if i == j { T::constant(1.0) } else { z };
            result[i][j] = T::constant(0.5) * em[i][j]
                + d * (em_wt + w_emt - T::constant(2.0) * omega[m] * kron)
                + axial_coeff * hat_w_sq[i][j];
        }
    }
    result
}

/// AD-generic SE(3) inverse Jacobian derivative tensor.
///
/// Returns T\[m\] = ∂(Jr^SE3)⁻¹/∂ξ_m as 6×6 matrices, m = 0..5.
pub fn se3_jr_inv_derivative_g<T: AD>(xi: &Vec6G<T>) -> [Mat6G<T>; 6] {
    let omega: Vec3G<T> = [xi[0], xi[1], xi[2]];
    let t: Vec3G<T> = [xi[3], xi[4], xi[5]];
    let (s, theta) = theta_sq_from_omega(&omega);
    let hat_w = hat_g(&omega);
    let hat_w_sq = mm3_g(&hat_w, &hat_w);
    let d = scalar_d_s(s, theta);

    let z3 = z3_g::<T>();
    let mut tensor = [blocks_6x6_g(&z3, &z3, &z3, &z3); 6];

    for m in 0..3 {
        let p_m = djr_inv_slice_g(&omega, &hat_w_sq, s, theta, d, m);
        let dqr_m = dqr_omega_slice_g(&omega, &t, &hat_w, &hat_w_sq, s, theta, d, m);
        tensor[m] = blocks_6x6_g(&p_m, &z3, &dqr_m, &p_m);
    }

    for m in 0..3 {
        let dqr_tm = dqr_t_slice_g(&omega, &hat_w_sq, s, theta, d, m);
        tensor[m + 3] = blocks_6x6_g(&z3, &z3, &dqr_tm, &z3);
    }

    tensor
}

/// AD-generic forward Jacobian derivative tensor ∂Jr^{SE3}/∂ξ.
pub fn se3_jr_derivative_g<T: AD>(xi: &Vec6G<T>) -> [Mat6G<T>; 6] {
    let omega: Vec3G<T> = [xi[0], xi[1], xi[2]];
    let t: Vec3G<T> = [xi[3], xi[4], xi[5]];
    let (s, theta) = theta_sq_from_omega(&omega);
    let hat_w = hat_g(&omega);
    let hat_w_sq = mm3_g(&hat_w, &hat_w);
    let d = scalar_d_s(s, theta);
    let jr_val = jr_g(&omega);
    let q_tilde = q_tilde_r_g(&omega, &t);
    let z3 = z3_g::<T>();
    let neg = T::constant(-1.0);
    let mut tensor = [blocks_6x6_g(&z3, &z3, &z3, &z3); 6];

    for m in 0..3 {
        let p_m = djr_inv_slice_g(&omega, &hat_w_sq, s, theta, d, m);
        let dqr_m = dqr_omega_slice_g(&omega, &t, &hat_w, &hat_w_sq, s, theta, d, m);
        let s_m = scale_mat3_g(neg, &mm3_g(&jr_val, &mm3_g(&p_m, &jr_val)));
        let dll_m = scale_mat3_g(
            neg,
            &add_mat3_g(
                &add_mat3_g(
                    &mm3_g(&s_m, &mm3_g(&q_tilde, &jr_val)),
                    &mm3_g(&jr_val, &mm3_g(&dqr_m, &jr_val)),
                ),
                &mm3_g(&jr_val, &mm3_g(&q_tilde, &s_m)),
            ),
        );
        tensor[m] = blocks_6x6_g(&s_m, &z3, &dll_m, &s_m);
    }

    for m in 0..3 {
        let dqr_tm = dqr_t_slice_g(&omega, &hat_w_sq, s, theta, d, m);
        let dll_tm = scale_mat3_g(neg, &mm3_g(&jr_val, &mm3_g(&dqr_tm, &jr_val)));
        tensor[m + 3] = blocks_6x6_g(&z3, &z3, &dll_tm, &z3);
    }
    tensor
}

// =========================================================================
// AD-generic re-centering Hessian
// =========================================================================

/// 6×6×6 tensor over AD scalar T.
pub type Tensor666G<T> = [[[T; 6]; 6]; 6];

/// AD-generic Hessian at general c (not just c = 0).
///
/// Uses `se3_jr_derivative_g` at c — the key function that was missing
/// from the original construction.
pub fn recentering_hessian_at_g<T: AD>(xi_bar: &Vec6G<T>, c: &Vec6G<T>) -> Tensor666G<T> {
    let base = PoseG::<T>::exp(xi_bar);
    let perturbed = base.compose(&PoseG::<T>::exp(c));
    let xi_prime = perturbed.log();

    let d_tensor = se3_jr_inv_derivative_g(&xi_prime);
    let jri = se3_jr_inv_g(&xi_prime);
    let jr_c = se3_jr_g(c);
    let j_c = mm6_g(&jri, &jr_c);
    let g_tensor = se3_jr_derivative_g(c);

    // Precompute D[m]·Jr(c): the chain rule requires the D-tensor
    // to act on Jr(c)·δc, not δc directly.
    let z = T::constant(0.0);
    let z6 = [[z; 6]; 6];
    let mut d_jr = [z6; 6];
    for m in 0..6 {
        d_jr[m] = mm6_g(&d_tensor[m], &jr_c);
    }

    let mut hf = [[[z; 6]; 6]; 6];
    for i in 0..6 {
        for p in 0..6 {
            for q in 0..6 {
                let mut val = z;
                // Term A: Σ_m (D[m]·Jr(c))[i][p] · J[m][q]
                for m in 0..6 {
                    val += d_jr[m][i][p] * j_c[m][q];
                }
                // Term B: Σ_r Jr⁻¹[i][r] · G[q][r][p]
                for r in 0..6 {
                    val += jri[i][r] * g_tensor[q][r][p];
                }
                hf[i][p][q] = val;
            }
        }
    }
    hf
}

/// AD-generic re-centering Hessian.
///
/// H^F\[i\]\[p\]\[q\] = ∂²F_i/∂c_p∂c_q |_{c=0} where F(c) = Log(Exp(ξ̄)·Exp(c)) − ξ̄.
///
/// When evaluated with `adfn<6>`, the tangent part gives the cubic tensor.
pub fn recentering_hessian_g<T: AD>(xi_bar: &Vec6G<T>) -> Tensor666G<T> {
    let d_tensor = se3_jr_inv_derivative_g(xi_bar);
    let jri = se3_jr_inv_g(xi_bar);
    let z = T::constant(0.0);

    let mut hf = [[[z; 6]; 6]; 6];

    for i in 0..6 {
        for p in 0..6 {
            for q in 0..6 {
                // Term A: Σ_m D[m][i][p] · Jr⁻¹[m][q]
                let mut val = z;
                for m in 0..6 {
                    val += d_tensor[m][i][p] * jri[m][q];
                }

                // Term B: Σ_r Jr⁻¹[i][r] · G[q][r][p]
                // G is the constant derivative of forward Jr^SE3 at identity
                for r in 0..6 {
                    let g = forward_jr_deriv_at_identity_val(q, r, p);
                    if g != 0.0 {
                        val += jri[i][r] * T::constant(g);
                    }
                }

                hf[i][p][q] = val;
            }
        }
    }
    hf
}

/// Constant G tensor (f64), used by both generic and concrete versions.
fn forward_jr_deriv_at_identity_val(q: usize, r: usize, p: usize) -> f64 {
    use crate::so3_unsafe::HAT_BASIS;
    if q < 3 {
        if r < 3 && p < 3 {
            return -0.5 * HAT_BASIS[q][r][p];
        }
        if r >= 3 && p >= 3 {
            return -0.5 * HAT_BASIS[q][r - 3][p - 3];
        }
        0.0
    } else {
        let q3 = q - 3;
        if r >= 3 && p < 3 {
            return -0.5 * HAT_BASIS[q3][r - 3][p];
        }
        0.0
    }
}
