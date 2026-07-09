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

/// AD-generic SE(3) Jacobian derivative tensor: six 6×6 slices
/// `[∂Jr/∂ξ_0, ..., ∂Jr/∂ξ_5]`. Re-exported as
/// `api::expert::types::JacobianDerivative6` under a semantic name.
pub type JacobianDerivative6G<T> = [Mat6G<T>; 6];

/// AD-generic SE(3) inverse Jacobian derivative tensor.
///
/// Returns T\[m\] = ∂(Jr^SE3)⁻¹/∂ξ_m as 6×6 matrices, m = 0..5.
pub fn se3_jr_inv_derivative_g<T: AD>(xi: &Vec6G<T>) -> JacobianDerivative6G<T> {
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
pub fn se3_jr_derivative_g<T: AD>(xi: &Vec6G<T>) -> JacobianDerivative6G<T> {
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

/// Directional derivative of the SE(3) right Jacobian:
/// `Σ_m direction[m] · ∂Jr/∂ξ_m`.
///
/// Equivalent to [`se3_jr_derivative_g`] contracted along its first
/// index. Provided as a convenience for the single-direction case
/// (Hessian-vector products in iterative solvers); when several
/// directions are needed in a hot loop, compute the full tensor once
/// and reuse it.
pub fn se3_jr_directional_derivative_g<T: AD>(xi: &Vec6G<T>, direction: &Vec6G<T>) -> Mat6G<T> {
    let tensor = se3_jr_derivative_g(xi);
    let z = T::constant(0.0);
    let mut result = [[z; 6]; 6];
    for m in 0..6 {
        for i in 0..6 {
            for j in 0..6 {
                result[i][j] += direction[m] * tensor[m][i][j];
            }
        }
    }
    result
}

/// Directional derivative of the SE(3) inverse right Jacobian:
/// `Σ_m direction[m] · ∂Jr⁻¹/∂ξ_m`.
///
/// See [`se3_jr_directional_derivative_g`] for the use-case.
pub fn se3_jr_inv_directional_derivative_g<T: AD>(xi: &Vec6G<T>, direction: &Vec6G<T>) -> Mat6G<T> {
    let tensor = se3_jr_inv_derivative_g(xi);
    let z = T::constant(0.0);
    let mut result = [[z; 6]; 6];
    for m in 0..6 {
        for i in 0..6 {
            for j in 0..6 {
                result[i][j] += direction[m] * tensor[m][i][j];
            }
        }
    }
    result
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

// =========================================================================
// Gaussian moment contractions (Isserlis / Wick)
// =========================================================================

/// Second-order mean shift of a zero-mean Gaussian pushed through a chart
/// map with Hessian tensor `h` (index order `[out][in][in]`, e.g. from
/// [`recentering_hessian_g`]).  For `ξ ~ N(0, Σ)` and
/// `φ(ξ) ≈ φ(0) + J·ξ + ½ H:ξξ`:
///
/// ```text
/// E[φ(ξ)] ≈ φ(0) + μ,    μ_i = ½ Σ_pq H[i][p][q] · Σ[p][q]
/// ```
pub fn second_order_mean_shift_g<T: AD>(h: &Tensor666G<T>, sigma: &Mat6G<T>) -> Vec6G<T> {
    let z = T::constant(0.0);
    let half = T::constant(0.5);
    let mut mu = [z; 6];
    for i in 0..6 {
        let mut acc = z;
        for p in 0..6 {
            for q in 0..6 {
                acc += h[i][p][q] * sigma[p][q];
            }
        }
        mu[i] = half * acc;
    }
    mu
}

/// Isserlis (Wick) covariance correction for the same second-order
/// transport.  With `y = J·ξ + ½ H:ξξ` and `ξ ~ N(0, Σ)`, the Gaussian
/// fourth-moment identity `E[ξ_a ξ_b ξ_c ξ_d] = Σ_ab Σ_cd + Σ_ac Σ_bd +
/// Σ_ad Σ_bc` reduces the quartic term to
///
/// ```text
/// Cov(y) = J Σ Jᵀ + Δ,    Δ_ij = ½ tr(H_i Σ H_j Σ),   (H_i)_pq = H[i][p][q]
/// ```
///
/// The `J × H` cross term carries only third Gaussian moments and
/// vanishes exactly, so `Δ` is exact for a purely quadratic map.  For a
/// smooth map the same O(σ⁴) order also carries a linear×cubic cross
/// term of comparable (sometimes dominant) magnitude — see
/// [`linear_cubic_covariance_correction_g`]; a complete second-order
/// transport needs both.
pub fn isserlis_covariance_correction_g<T: AD>(h: &Tensor666G<T>, sigma: &Mat6G<T>) -> Mat6G<T> {
    let z = T::constant(0.0);
    let half = T::constant(0.5);
    // P_i = H_i·Σ;  Δ_ij = ½ tr(P_i·P_j) = ½ Σ_ab P_i[a][b]·P_j[b][a].
    let p: [Mat6G<T>; 6] = std::array::from_fn(|i| mm6_g(&h[i], sigma));
    let mut delta = [[z; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut acc = z;
            for a in 0..6 {
                for b in 0..6 {
                    acc += p[i][a][b] * p[j][b][a];
                }
            }
            let v = half * acc;
            delta[i][j] = v;
            delta[j][i] = v;
        }
    }
    delta
}

/// 6×6×6×6 tensor over AD scalar T (index order `[out][in][in][in]`,
/// symmetric in the trailing three).
pub type Tensor6666G<T> = [[[[T; 6]; 6]; 6]; 6];

/// Linear×cubic Isserlis covariance correction — the *other* O(σ⁴) term.
///
/// Extending the transport to third order, `y = J·ξ + ½ H:ξξ + ⅙ C:ξξξ`,
/// the Gaussian fourth-moment identity applied to the `J × C` cross term
/// gives (with `M[j][p] = Σ_qr C[j][p][q][r]·Σ[q][r]`):
///
/// ```text
/// Δ_LC = ½ (J Σ Mᵀ + M Σ Jᵀ)
/// ```
///
/// Together, `Cov(y) = J Σ Jᵀ + Δ_QQ + Δ_LC + O(σ⁶)` with `Δ_QQ` from
/// [`isserlis_covariance_correction_g`].  On SE(3) chart maps the two
/// terms are of comparable magnitude (in composition-propagation settings
/// the linear×cubic channel dominates and is negative — the group's
/// inward curvature contracting uncertainty); a complete O(σ⁴) transport
/// includes both.
///
/// The cubic tensor need not come from an analytical stack: one
/// `adfn<6>`-seeded evaluation of [`recentering_hessian_at_g`] at `c = 0`
/// yields `C[i][p][q][r]` as the tangent of `H[i][p][q]` (the mixed-AD
/// recipe; see the in-tree transport test for the five-line construction).
pub fn linear_cubic_covariance_correction_g<T: AD>(
    jac: &Mat6G<T>,
    cubic: &Tensor6666G<T>,
    sigma: &Mat6G<T>,
) -> Mat6G<T> {
    let z = T::constant(0.0);
    let half = T::constant(0.5);

    // M[j][p] = Σ_qr C[j][p][q][r]·Σ[q][r]
    let mut m = [[z; 6]; 6];
    for j in 0..6 {
        for p in 0..6 {
            let mut acc = z;
            for q in 0..6 {
                for r in 0..6 {
                    acc += cubic[j][p][q][r] * sigma[q][r];
                }
            }
            m[j][p] = acc;
        }
    }

    // A = J·Σ·Mᵀ;  Δ_LC = ½ (A + Aᵀ).
    let js = mm6_g(jac, sigma);
    let mut delta = [[z; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut a_ij = z;
            let mut a_ji = z;
            for k in 0..6 {
                a_ij += js[i][k] * m[j][k];
                a_ji += js[j][k] * m[i][k];
            }
            delta[i][j] = half * (a_ij + a_ji);
        }
    }
    delta
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::se3_adsafe::mv6_g;

    /// SplitMix64 + two-uniform Box–Muller, deterministic.
    struct TestRng(u64);
    impl TestRng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    const XI_BAR: Vec6G<f64> = [0.3, -0.2, 0.4, 0.5, -0.3, 0.7];

    /// Correlated test covariance Σ = L·Lᵀ from an explicit lower-triangular
    /// factor (rotation σ ≈ 0.15, translation σ ≈ 0.2, mild cross terms).
    fn sigma_factor() -> [[f64; 6]; 6] {
        let mut l = [[0.0f64; 6]; 6];
        let d = [0.15, 0.13, 0.16, 0.20, 0.18, 0.22];
        for i in 0..6 {
            l[i][i] = d[i];
            for j in 0..i {
                l[i][j] = 0.02 * ((i + 2 * j) % 3) as f64;
            }
        }
        l
    }

    fn sigma_from_factor(l: &[[f64; 6]; 6]) -> Mat6G<f64> {
        let mut s = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    s[i][j] += l[i][k] * l[j][k];
                }
            }
        }
        s
    }

    fn draw_xi(rng: &mut TestRng, l: &[[f64; 6]; 6]) -> Vec6G<f64> {
        let n: [f64; 6] = std::array::from_fn(|_| rng.normal());
        std::array::from_fn(|i| (0..=i).map(|k| l[i][k] * n[k]).sum())
    }

    fn frob6(m: &Mat6G<f64>) -> f64 {
        m.iter().flatten().map(|v| v * v).sum::<f64>().sqrt()
    }

    fn frob6_diff(a: &Mat6G<f64>, b: &Mat6G<f64>) -> f64 {
        let mut s = 0.0;
        for i in 0..6 {
            for j in 0..6 {
                s += (a[i][j] - b[i][j]).powi(2);
            }
        }
        s.sqrt()
    }

    /// On a *purely quadratic* map y = ½ H:ξξ the two contractions are the
    /// exact mean and covariance — Monte Carlo must converge to them.
    /// H is a real recentering Hessian so all index symmetries are honest.
    #[test]
    fn isserlis_contractions_match_mc_on_quadratic_map() {
        let h = recentering_hessian_g::<f64>(&XI_BAR);
        let l = sigma_factor();
        let sigma = sigma_from_factor(&l);

        let mu = second_order_mean_shift_g::<f64>(&h, &sigma);
        let delta = isserlis_covariance_correction_g::<f64>(&h, &sigma);

        let n = 200_000usize;
        let mut rng = TestRng(42);
        let mut mean = [0.0f64; 6];
        let mut cov = [[0.0f64; 6]; 6];
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let xi = draw_xi(&mut rng, &l);
            let y: [f64; 6] = std::array::from_fn(|i| {
                let mut acc = 0.0;
                for p in 0..6 {
                    for q in 0..6 {
                        acc += h[i][p][q] * xi[p] * xi[q];
                    }
                }
                0.5 * acc
            });
            for i in 0..6 {
                mean[i] += y[i];
            }
            samples.push(y);
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }
        for y in &samples {
            for i in 0..6 {
                for j in 0..6 {
                    cov[i][j] += (y[i] - mean[i]) * (y[j] - mean[j]);
                }
            }
        }
        for row in cov.iter_mut() {
            for v in row.iter_mut() {
                *v /= (n - 1) as f64;
            }
        }

        let mean_scale = mu.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for i in 0..6 {
            assert!(
                (mean[i] - mu[i]).abs() < 0.02 * mean_scale,
                "mean[{i}]: MC {:.5e} vs ½H:Σ {:.5e}",
                mean[i],
                mu[i]
            );
        }
        let rel = frob6_diff(&cov, &delta) / frob6(&delta);
        assert!(
            rel < 0.05,
            "quadratic-map covariance: MC vs Isserlis rel = {rel:.3e}"
        );
    }

    /// Cubic tensor of F(c) = Log(Exp(ξ̄)·Exp(c)) − ξ̄ at c = 0 via the
    /// mixed-AD recipe: one `adfn<6>`-seeded evaluation of the general-chart
    /// Hessian; `C[i][p][q][r]` is the tangent of `H[i][p][q]`.
    fn recentering_cubic_seeded(xi_bar: &Vec6G<f64>) -> Tensor6666G<f64> {
        use crate::autodiff::forward_ad::adfn;
        type T = adfn<6>;
        let xb: Vec6G<T> = std::array::from_fn(|i| T::new_constant(xi_bar[i]));
        let c: Vec6G<T> = std::array::from_fn(|r| {
            let mut t = [0.0; 6];
            t[r] = 1.0;
            T::new(0.0, t)
        });
        let h = recentering_hessian_at_g::<T>(&xb, &c);
        std::array::from_fn(|i| {
            std::array::from_fn(|p| std::array::from_fn(|q| h[i][p][q].tangent()))
        })
    }

    /// 5-node Gauss–Hermite (physicists') nodes and weights.
    const GH5_NODES: [f64; 5] = [
        -2.0201828704560856,
        -0.9585724646138185,
        0.0,
        0.9585724646138185,
        2.0201828704560856,
    ];
    const GH5_WEIGHTS: [f64; 5] = [
        0.019953242059045913,
        0.3936193231522412,
        0.9453087204829419,
        0.3936193231522412,
        0.019953242059045913,
    ];

    /// Application miniature: transport ξ ~ N(0, Σ) through the chart-change
    /// map F(c) = Log(Exp(ξ̄)·Exp(c)) − ξ̄.  Ground truth is a deterministic
    /// 5⁶-point tensor Gauss–Hermite grid (exact through degree-9 moments —
    /// no MC noise), so the O(σ⁴) corrections are cleanly resolvable:
    /// (½H:Σ, JΣJᵀ + Δ_QQ + Δ_LC) must beat (0, JΣJᵀ) by a wide margin.
    #[test]
    fn corrected_covariance_transport_beats_first_order() {
        let l = sigma_factor();
        let sigma = sigma_from_factor(&l);

        let jac = se3_jr_inv_g::<f64>(&XI_BAR);
        let h = recentering_hessian_g::<f64>(&XI_BAR);
        let cubic = recentering_cubic_seeded(&XI_BAR);
        let mu = second_order_mean_shift_g::<f64>(&h, &sigma);
        let d_qq = isserlis_covariance_correction_g::<f64>(&h, &sigma);
        let d_lc = linear_cubic_covariance_correction_g::<f64>(&jac, &cubic, &sigma);

        eprintln!(
            "transport corrections: ‖Δ_QQ‖ = {:.3e}, ‖Δ_LC‖ = {:.3e}",
            frob6(&d_qq),
            frob6(&d_lc)
        );

        // First-order transport JΣJᵀ, then the fully corrected version.
        let mut cov1 = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for a in 0..6 {
                    for b in 0..6 {
                        cov1[i][j] += jac[i][a] * sigma[a][b] * jac[j][b];
                    }
                }
            }
        }
        let cov2: Mat6G<f64> =
            std::array::from_fn(|i| std::array::from_fn(|j| cov1[i][j] + d_qq[i][j] + d_lc[i][j]));

        // Gauss–Hermite ground truth: ξ = √2·L·u, weight Πwₖ/π³.
        let base = PoseG::<f64>::exp(&XI_BAR);
        let norm = std::f64::consts::PI.powi(3);
        let sqrt2 = std::f64::consts::SQRT_2;
        let mut mean_gh = [0.0f64; 6];
        let mut second_gh = [[0.0f64; 6]; 6];
        let mut idx = [0usize; 6];
        loop {
            let mut w = 1.0;
            let mut u = [0.0f64; 6];
            for (k, &ik) in idx.iter().enumerate() {
                w *= GH5_WEIGHTS[ik];
                u[k] = GH5_NODES[ik];
            }
            w /= norm;
            let c: Vec6G<f64> =
                std::array::from_fn(|i| sqrt2 * (0..=i).map(|k| l[i][k] * u[k]).sum::<f64>());
            let xi_prime = base.compose(&PoseG::exp(&c)).log();
            let y: [f64; 6] = std::array::from_fn(|i| xi_prime[i] - XI_BAR[i]);
            for i in 0..6 {
                mean_gh[i] += w * y[i];
                for j in 0..6 {
                    second_gh[i][j] += w * y[i] * y[j];
                }
            }
            // Odometer over the 5⁶ grid.
            let mut k = 0;
            loop {
                idx[k] += 1;
                if idx[k] < 5 {
                    break;
                }
                idx[k] = 0;
                k += 1;
                if k == 6 {
                    break;
                }
            }
            if k == 6 {
                break;
            }
        }
        let mut cov_gh = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                cov_gh[i][j] = second_gh[i][j] - mean_gh[i] * mean_gh[j];
            }
        }

        // Mean: ½H:Σ must capture the bias to O(σ⁴).
        let err_mean_0: f64 = mean_gh.iter().map(|v| v * v).sum::<f64>().sqrt();
        let err_mean_2: f64 = mean_gh
            .iter()
            .zip(&mu)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(
            err_mean_2 < 0.1 * err_mean_0,
            "mean shift: corrected {err_mean_2:.3e} vs uncorrected {err_mean_0:.3e}"
        );

        // Covariance: JΣJᵀ + Δ_QQ + Δ_LC must beat JΣJᵀ by a wide margin.
        let err1 = frob6_diff(&cov1, &cov_gh);
        let err2 = frob6_diff(&cov2, &cov_gh);
        eprintln!(
            "transport vs GH ground truth: mean {err_mean_0:.3e} → {err_mean_2:.3e}, \
             cov {err1:.3e} → {err2:.3e}"
        );
        assert!(
            err2 < 0.2 * err1,
            "covariance: corrected {err2:.3e} vs first-order {err1:.3e} \
             (Δ_QQ {:.3e}, Δ_LC {:.3e})",
            frob6(&d_qq),
            frob6(&d_lc)
        );

        // Sanity: J = Jr⁻¹(ξ̄) linearizes F — check against directional FD.
        let h_fd = 1e-6;
        for dir in 0..6 {
            let mut cp = [0.0; 6];
            cp[dir] = h_fd;
            let mut cm = [0.0; 6];
            cm[dir] = -h_fd;
            let fp = base.compose(&PoseG::exp(&cp)).log();
            let fm = base.compose(&PoseG::exp(&cm)).log();
            let col: [f64; 6] = std::array::from_fn(|i| (fp[i] - fm[i]) / (2.0 * h_fd));
            let jcol = mv6_g(&jac, &{
                let mut e = [0.0; 6];
                e[dir] = 1.0;
                e
            });
            for i in 0..6 {
                assert!(
                    (col[i] - jcol[i]).abs() < 1e-6,
                    "J column {dir} row {i}: FD {:.6e} vs Jr⁻¹ {:.6e}",
                    col[i],
                    jcol[i]
                );
            }
        }
    }
}
