//! # SE_2(3) Jacobians and Adjoint — AD-safe.
//!
//! Adjoint matrix, right Jacobian, and inverse right Jacobian for the
//! extended pose group SE_2(3), templated over `T: AD`.  All operations
//! reuse SE(3) machinery from [`crate::so3_adsafe`] and
//! [`crate::se3_adsafe`] wherever possible — the SE_2(3) versions are
//! essentially "two SE(3) operations sharing a rotation block."
//!
//! ## Adjoint
//!
//! For ξ = (R, v, p) ∈ SE_2(3), the adjoint `Ad_ξ ∈ ℝ⁹ˣ⁹` is
//!
//! ```text
//!         | R       0  0 |
//! Ad_ξ =  | [v]×R   R  0 |
//!         | [p]×R   0  R |
//! ```
//!
//! Both `[v]×R` and `[p]×R` mirror the SE(3) adjoint's translation-
//! rotation coupling block, applied independently.  The (ν, ρ) and
//! (ρ, ν) blocks are zero — velocity and position do not couple to
//! each other under the adjoint, only to ω.
//!
//! ## Right Jacobian
//!
//! For ξ = (ω, ν, ρ) ∈ ℝ⁹, the right Jacobian `Jr(ξ)` is the lower
//! block-triangular matrix
//!
//! ```text
//!          | Jr_SO3(ω)            0           0          |
//! Jr(ξ) =  | −Jr·Q̃_r(ω, ν)·Jr   Jr_SO3(ω)   0          |
//!          | −Jr·Q̃_r(ω, ρ)·Jr   0           Jr_SO3(ω)  |
//! ```
//!
//! with `Jr = Jr_SO3(ω) = `[`crate::so3_adsafe::jr_g`] and `Q̃_r` =
//! [`crate::se3_adsafe::q_tilde_r_g`].  `Q̃_r` is the inverse-side coupling
//! block; the forward block (the paper's `Q_r`) is the `−Jr·Q̃_r·Jr`
//! sandwich, matching the convention in [`crate::se3_adsafe::se3_jr_g`].
//!
//! ## Inverse Right Jacobian
//!
//! Block-triangular inverse: the sandwich on the coupling block
//! simplifies because `−A⁻¹·(−A·Q̃_r·A)·A⁻¹ = Q̃_r`, so the coupling
//! blocks of `Jr⁻¹` are the bare `Q̃_r`.
//!
//! ```text
//!             | Jr⁻¹             0      0     |
//! Jr⁻¹(ξ) =   | Q̃_r(ω, ν)       Jr⁻¹   0     |
//!             | Q̃_r(ω, ρ)       0      Jr⁻¹  |
//! ```

use crate::autodiff::ad_trait::AD;
use crate::se3_adsafe::q_tilde_r_g;
use crate::se23_adsafe::{Mat9G, Vec9G, blocks_9x9_g};
use crate::so3_adsafe::{
    Mat3G, Vec3G, hat_g, jr_g, jr_inv_g, mm3_g, mv3_g, scale_mat3_g, transpose3_g, z3_g,
};

// ─── Adjoint ────────────────────────────────────────────────────────────

/// SE_2(3) adjoint matrix `Ad_(R,v,p) ∈ ℝ⁹ˣ⁹`.
///
/// Block structure:
///
/// ```text
///   Ad = | R       0  0 |
///        | [v]×R   R  0 |
///        | [p]×R   0  R |
/// ```
///
/// Computes the two coupling blocks `[v]×R` and `[p]×R` (each a single
/// 3×3 multiply) and assembles via [`blocks_9x9_g`].  The (ν, ρ) and
/// (ρ, ν) blocks are zero.
pub fn adjoint_se23_g<T: AD>(rot: &Mat3G<T>, vel: &Vec3G<T>, pos: &Vec3G<T>) -> Mat9G<T> {
    let v_skew = hat_g(vel);
    let p_skew = hat_g(pos);
    let v_skew_r = mm3_g(&v_skew, rot);
    let p_skew_r = mm3_g(&p_skew, rot);
    let z = z3_g::<T>();
    blocks_9x9_g(
        rot, &z, &z, // row 0:  R       0  0
        &v_skew_r, rot, &z, // row 1:  [v]×R   R  0
        &p_skew_r, &z, rot, // row 2:  [p]×R   0  R
    )
}

/// SE_2(3) adjoint of the inverse pose: `Ad_(g⁻¹) = Ad_g⁻¹`.
///
/// Inverts the pose first, then takes the adjoint:
///
/// ```text
///   g⁻¹      = (Rᵀ, −Rᵀv, −Rᵀp)
///   Ad(g⁻¹)  = adjoint_se23_g(Rᵀ, −Rᵀv, −Rᵀp)
/// ```
///
/// Equivalent to inverting the 9×9 adjoint directly but far cheaper:
/// one transpose and two 3-vector negations versus a 9×9 inverse.
pub fn adjoint_inv_se23_g<T: AD>(rot: &Mat3G<T>, vel: &Vec3G<T>, pos: &Vec3G<T>) -> Mat9G<T> {
    let rt = transpose3_g(rot);
    let rt_v = mv3_g(&rt, vel);
    let rt_p = mv3_g(&rt, pos);
    let z = T::constant(0.0);
    let neg_rt_v = [z - rt_v[0], z - rt_v[1], z - rt_v[2]];
    let neg_rt_p = [z - rt_p[0], z - rt_p[1], z - rt_p[2]];
    adjoint_se23_g(&rt, &neg_rt_v, &neg_rt_p)
}

// ─── Right Jacobian ─────────────────────────────────────────────────────

/// SE_2(3) right Jacobian `Jr(ξ) ∈ ℝ⁹ˣ⁹` where ξ = \[ω; ν; ρ\].
///
/// Block structure:
///
/// ```text
///   Jr = | Jr_SO3(ω)             0           0          |
///        | −Jr·Q̃_r(ω,ν)·Jr      Jr_SO3(ω)   0          |
///        | −Jr·Q̃_r(ω,ρ)·Jr      0           Jr_SO3(ω)  |
/// ```
///
/// The off-diagonal coupling blocks use the same `−Jr·Q̃_r·Jr` sandwich
/// as [`crate::se3_adsafe::se3_jr_g`].
pub fn se23_jr_g<T: AD>(xi: &Vec9G<T>) -> Mat9G<T> {
    let omega: Vec3G<T> = [xi[0], xi[1], xi[2]];
    let nu: Vec3G<T> = [xi[3], xi[4], xi[5]];
    let rho: Vec3G<T> = [xi[6], xi[7], xi[8]];

    let jr_so3 = jr_g(&omega);
    let q_tilde_nu = q_tilde_r_g(&omega, &nu);
    let q_tilde_rho = q_tilde_r_g(&omega, &rho);

    let neg_one = T::constant(-1.0);
    let ll_nu = scale_mat3_g(neg_one, &mm3_g(&jr_so3, &mm3_g(&q_tilde_nu, &jr_so3)));
    let ll_rho = scale_mat3_g(neg_one, &mm3_g(&jr_so3, &mm3_g(&q_tilde_rho, &jr_so3)));

    let z = z3_g::<T>();
    blocks_9x9_g(
        &jr_so3, &z, &z, // row 0
        &ll_nu, &jr_so3, &z, // row 1
        &ll_rho, &z, &jr_so3, // row 2
    )
}

/// SE_2(3) inverse right Jacobian `Jr⁻¹(ξ) ∈ ℝ⁹ˣ⁹` where ξ = \[ω; ν; ρ\].
///
/// Block-triangular inverse of [`se23_jr_g`].  The coupling blocks
/// simplify to the bare `Q̃_r` because `−A⁻¹·(−A·Q̃_r·A)·A⁻¹ = Q̃_r`:
///
/// ```text
///   Jr⁻¹ = | Jr⁻¹             0      0     |
///          | Q̃_r(ω, ν)       Jr⁻¹   0     |
///          | Q̃_r(ω, ρ)       0      Jr⁻¹  |
/// ```
///
/// Only the SO(3) inverse right Jacobian is computed (via
/// [`crate::so3_adsafe::jr_inv_g`]); no 3×3 matrix inversion is performed in
/// this function.
pub fn se23_jr_inv_g<T: AD>(xi: &Vec9G<T>) -> Mat9G<T> {
    let omega: Vec3G<T> = [xi[0], xi[1], xi[2]];
    let nu: Vec3G<T> = [xi[3], xi[4], xi[5]];
    let rho: Vec3G<T> = [xi[6], xi[7], xi[8]];

    let jri = jr_inv_g(&omega);
    let q_tilde_nu = q_tilde_r_g(&omega, &nu);
    let q_tilde_rho = q_tilde_r_g(&omega, &rho);

    let z = z3_g::<T>();
    blocks_9x9_g(
        &jri,
        &z,
        &z, // row 0
        &q_tilde_nu,
        &jri,
        &z, // row 1
        &q_tilde_rho,
        &z,
        &jri, // row 2
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::se23_adsafe::ExtendedPoseG;
    use crate::so3_adsafe::{so3_exp_g, v_matrix_g};
    use crate::{I3, I9, Mat3, Mat9, Z3, extract_block3_from9, mm, scale_mat};

    type ExtendedPose = ExtendedPoseG<f64>;

    fn approx_eq_mat9(a: &Mat9, b: &Mat9, tol: f64) -> bool {
        for i in 0..9 {
            for j in 0..9 {
                if (a[i][j] - b[i][j]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    fn approx_eq_mat3(a: &Mat3, b: &Mat3, tol: f64) -> bool {
        for i in 0..3 {
            for j in 0..3 {
                if (a[i][j] - b[i][j]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    // ═══ Adjoint tests ═══════════════════════════════════════════════════

    #[test]
    fn adjoint_at_identity_is_i9() {
        let ad = adjoint_se23_g::<f64>(&I3, &[0.0; 3], &[0.0; 3]);
        assert!(approx_eq_mat9(&ad, &I9, 1e-15));
    }

    #[test]
    fn adjoint_block_structure() {
        let r = so3_exp_g::<f64>(&[0.3, -0.2, 0.5]);
        let v = [1.0, 2.0, 3.0];
        let p = [4.0, 5.0, 6.0];
        let ad = adjoint_se23_g::<f64>(&r, &v, &p);

        // Diagonal blocks: all R
        assert!(approx_eq_mat3(&extract_block3_from9(&ad, 0, 0), &r, 1e-15));
        assert!(approx_eq_mat3(&extract_block3_from9(&ad, 3, 3), &r, 1e-15));
        assert!(approx_eq_mat3(&extract_block3_from9(&ad, 6, 6), &r, 1e-15));

        // Coupling blocks: [v]×R and [p]×R
        let v_skew_r = mm(&hat_g::<f64>(&v), &r);
        let p_skew_r = mm(&hat_g::<f64>(&p), &r);
        assert!(approx_eq_mat3(
            &extract_block3_from9(&ad, 3, 0),
            &v_skew_r,
            1e-15
        ));
        assert!(approx_eq_mat3(
            &extract_block3_from9(&ad, 6, 0),
            &p_skew_r,
            1e-15
        ));

        // Off-diagonal zeros
        assert!(approx_eq_mat3(&extract_block3_from9(&ad, 0, 3), &Z3, 1e-15));
        assert!(approx_eq_mat3(&extract_block3_from9(&ad, 0, 6), &Z3, 1e-15));
        assert!(approx_eq_mat3(&extract_block3_from9(&ad, 3, 6), &Z3, 1e-15));
        assert!(approx_eq_mat3(&extract_block3_from9(&ad, 6, 3), &Z3, 1e-15));
    }

    #[test]
    fn adjoint_velocity_position_no_cross_coupling() {
        // (ν, ρ) and (ρ, ν) blocks of the adjoint are always zero —
        // velocity and position don't couple in the bracket.
        let r = so3_exp_g::<f64>(&[0.4, -0.1, 0.3]);
        let v = [2.0, -1.0, 0.5];
        let p = [-1.5, 0.8, 2.2];
        let ad = adjoint_se23_g::<f64>(&r, &v, &p);
        assert!(approx_eq_mat3(&extract_block3_from9(&ad, 3, 6), &Z3, 1e-15));
        assert!(approx_eq_mat3(&extract_block3_from9(&ad, 6, 3), &Z3, 1e-15));
    }

    #[test]
    fn adjoint_is_group_homomorphism() {
        // Ad(g_a · g_b) = Ad(g_a) · Ad(g_b)
        // Strong correctness check that doesn't peek at internal blocks.
        let g_a = ExtendedPose::exp(&[0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 0.7, 0.3, 0.5]);
        let g_b = ExtendedPose::exp(&[-0.1, 0.15, -0.2, 0.3, -0.4, 0.6, 0.2, 0.5, -0.3]);
        let g_ab = g_a.compose(&g_b);

        let ad_a = adjoint_se23_g::<f64>(&g_a.rot, &g_a.vel, &g_a.pos);
        let ad_b = adjoint_se23_g::<f64>(&g_b.rot, &g_b.vel, &g_b.pos);
        let ad_ab = adjoint_se23_g::<f64>(&g_ab.rot, &g_ab.vel, &g_ab.pos);

        let product = mm(&ad_a, &ad_b);
        assert!(
            approx_eq_mat9(&product, &ad_ab, 1e-12),
            "Ad(g_a · g_b) should equal Ad(g_a) · Ad(g_b)"
        );
    }

    #[test]
    fn adjoint_inv_is_inverse_of_adjoint() {
        let g = ExtendedPose::exp(&[0.3, -0.5, 0.2, 1.0, -2.0, 0.5, 0.8, 1.2, -0.4]);
        let ad = adjoint_se23_g::<f64>(&g.rot, &g.vel, &g.pos);
        let ad_inv = adjoint_inv_se23_g::<f64>(&g.rot, &g.vel, &g.pos);

        let product = mm(&ad, &ad_inv);
        assert!(
            approx_eq_mat9(&product, &I9, 1e-12),
            "Ad · Ad_inv should be I9"
        );
        let product_other = mm(&ad_inv, &ad);
        assert!(
            approx_eq_mat9(&product_other, &I9, 1e-12),
            "Ad_inv · Ad should be I9"
        );
    }

    #[test]
    fn adjoint_inv_matches_adjoint_of_inverse_pose() {
        // adjoint_inv(g) == adjoint(g.inverse()).
        let g = ExtendedPose::exp(&[0.2, 0.3, -0.1, 1.5, -0.5, 0.8, 0.4, -0.7, 0.9]);
        let g_inv = g.inverse();

        let ad_inv_via_helper = adjoint_inv_se23_g::<f64>(&g.rot, &g.vel, &g.pos);
        let ad_inv_via_pose = adjoint_se23_g::<f64>(&g_inv.rot, &g_inv.vel, &g_inv.pos);

        assert!(
            approx_eq_mat9(&ad_inv_via_helper, &ad_inv_via_pose, 1e-15),
            "adjoint_inv(g) should equal adjoint(g.inverse())"
        );
    }

    // ═══ Right Jacobian tests ═══════════════════════════════════════════

    #[test]
    fn jr_at_zero_is_i9() {
        let j = se23_jr_g::<f64>(&[0.0; 9]);
        assert!(approx_eq_mat9(&j, &I9, 1e-10));
    }

    #[test]
    fn jr_pure_translation_has_neg_half_hat_offdiagonals() {
        // ω = 0: Jr_SO3 = I and Q̃_r(0, t) = ½[t]×, so the −Jr·Q̃_r·Jr
        // sandwich collapses to −½[t]×.
        let nu = [1.0_f64, 2.0, 3.0];
        let rho = [4.0_f64, 5.0, 6.0];
        let xi = [0.0, 0.0, 0.0, nu[0], nu[1], nu[2], rho[0], rho[1], rho[2]];
        let j = se23_jr_g::<f64>(&xi);

        // Diagonal blocks: I3
        assert!(approx_eq_mat3(&extract_block3_from9(&j, 0, 0), &I3, 1e-12));
        assert!(approx_eq_mat3(&extract_block3_from9(&j, 3, 3), &I3, 1e-12));
        assert!(approx_eq_mat3(&extract_block3_from9(&j, 6, 6), &I3, 1e-12));

        // Coupling blocks: −½[ν]× and −½[ρ]×
        let neg_half_hat_nu = scale_mat(-0.5, &hat_g::<f64>(&nu));
        let neg_half_hat_rho = scale_mat(-0.5, &hat_g::<f64>(&rho));
        assert!(approx_eq_mat3(
            &extract_block3_from9(&j, 3, 0),
            &neg_half_hat_nu,
            1e-12
        ));
        assert!(approx_eq_mat3(
            &extract_block3_from9(&j, 6, 0),
            &neg_half_hat_rho,
            1e-12
        ));

        // Cross-coupling blocks still zero
        assert!(approx_eq_mat3(&extract_block3_from9(&j, 3, 6), &Z3, 1e-15));
        assert!(approx_eq_mat3(&extract_block3_from9(&j, 6, 3), &Z3, 1e-15));
    }

    #[test]
    fn jr_block_diagonal_when_pure_rotation() {
        // ν = ρ = 0  ⇒  Q-blocks vanish, Jr is block-diagonal with
        // diag(Jr_SO3, Jr_SO3, Jr_SO3).
        let omega = [0.3_f64, -0.2, 0.5];
        let xi = [omega[0], omega[1], omega[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let j = se23_jr_g::<f64>(&xi);

        assert!(approx_eq_mat3(&extract_block3_from9(&j, 3, 0), &Z3, 1e-12));
        assert!(approx_eq_mat3(&extract_block3_from9(&j, 6, 0), &Z3, 1e-12));

        let jr_so3 = jr_g::<f64>(&omega);
        assert!(approx_eq_mat3(
            &extract_block3_from9(&j, 0, 0),
            &jr_so3,
            1e-15
        ));
        assert!(approx_eq_mat3(
            &extract_block3_from9(&j, 3, 3),
            &jr_so3,
            1e-15
        ));
        assert!(approx_eq_mat3(
            &extract_block3_from9(&j, 6, 6),
            &jr_so3,
            1e-15
        ));
    }

    #[test]
    fn jr_so3_block_matches_so3_v_matrix_of_negated_omega() {
        // SO(3) identity: Jr(ω) = Jl(-ω) = V(-ω).
        let omega = [0.3_f64, -0.2, 0.5];
        let neg_omega = [-omega[0], -omega[1], -omega[2]];
        let jr_so3_via_v = v_matrix_g::<f64>(&neg_omega);

        let xi = [omega[0], omega[1], omega[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let j = se23_jr_g::<f64>(&xi);
        let jr_so3_from_se23 = extract_block3_from9(&j, 0, 0);

        assert!(
            approx_eq_mat3(&jr_so3_from_se23, &jr_so3_via_v, 1e-12),
            "(0,0) block of Jr_SE_2(3) should equal v_matrix(-ω) = Jl_SO3(-ω) = Jr_SO3(ω)"
        );
    }

    #[test]
    fn jr_coupling_blocks_match_neg_jr_qr_jr() {
        // Coupling blocks of Jr_SE_2(3) are −Jr_SO3·Q̃_r(ω,·)·Jr_SO3.
        let omega = [0.3_f64, -0.2, 0.5];
        let nu = [1.0_f64, 2.0, 3.0];
        let rho = [4.0_f64, 5.0, 6.0];
        let xi = [
            omega[0], omega[1], omega[2], nu[0], nu[1], nu[2], rho[0], rho[1], rho[2],
        ];

        let j = se23_jr_g::<f64>(&xi);
        let jr_so3 = jr_g::<f64>(&omega);

        let expected_nu = scale_mat(
            -1.0,
            &mm(&jr_so3, &mm(&q_tilde_r_g::<f64>(&omega, &nu), &jr_so3)),
        );
        assert!(approx_eq_mat3(
            &extract_block3_from9(&j, 3, 0),
            &expected_nu,
            1e-15
        ));

        let expected_rho = scale_mat(
            -1.0,
            &mm(&jr_so3, &mm(&q_tilde_r_g::<f64>(&omega, &rho), &jr_so3)),
        );
        assert!(approx_eq_mat3(
            &extract_block3_from9(&j, 6, 0),
            &expected_rho,
            1e-15
        ));
    }

    #[test]
    fn jr_velocity_position_independent() {
        // (ν, ρ) and (ρ, ν) blocks of Jr are always zero.
        let omega = [0.4_f64, 0.3, -0.2];
        let nu = [0.5_f64, 1.0, -0.5];
        let rho = [1.0_f64, -1.5, 2.0];
        let xi = [
            omega[0], omega[1], omega[2], nu[0], nu[1], nu[2], rho[0], rho[1], rho[2],
        ];
        let j = se23_jr_g::<f64>(&xi);
        assert!(approx_eq_mat3(&extract_block3_from9(&j, 3, 6), &Z3, 1e-15));
        assert!(approx_eq_mat3(&extract_block3_from9(&j, 6, 3), &Z3, 1e-15));
    }

    #[test]
    fn jr_jr_inv_is_i9() {
        let xi = [0.3_f64, -0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let j = se23_jr_g::<f64>(&xi);
        let ji = se23_jr_inv_g::<f64>(&xi);

        let product = mm(&j, &ji);
        assert!(
            approx_eq_mat9(&product, &I9, 1e-10),
            "Jr · Jr_inv should be I9"
        );
        let product_other = mm(&ji, &j);
        assert!(
            approx_eq_mat9(&product_other, &I9, 1e-10),
            "Jr_inv · Jr should be I9"
        );
    }

    #[test]
    fn jr_inv_at_zero_is_i9() {
        let ji = se23_jr_inv_g::<f64>(&[0.0; 9]);
        assert!(approx_eq_mat9(&ji, &I9, 1e-10));
    }

    #[test]
    fn jr_finite_difference_check() {
        // Verify Jr column-by-column via finite differences of the exp map:
        //   exp(ξ + ε δξ) ≈ exp(ξ) · exp(Jr(ξ) ε δξ)
        // ⇒ log(exp(ξ)⁻¹ · exp(ξ + ε e_k)) / ε ≈ column k of Jr(ξ).
        let xi: [f64; 9] = [0.2, 0.3, -0.1, 0.5, 1.0, -0.5, 0.4, 0.6, 0.8];
        let j = se23_jr_g::<f64>(&xi);

        let g_xi = ExtendedPose::exp(&xi);
        let g_xi_inv = g_xi.inverse();

        let eps = 1e-6_f64;

        for k in 0..9 {
            let mut xi_pert = xi;
            xi_pert[k] += eps;
            let g_pert = ExtendedPose::exp(&xi_pert);

            let rel = g_xi_inv.compose(&g_pert);
            let log_rel = rel.log();

            for i in 0..9 {
                let expected = j[i][k];
                let actual = log_rel[i] / eps;
                assert!(
                    (expected - actual).abs() < 1e-4,
                    "Jr[{}][{}]: analytic={:.6e}  numerical={:.6e}  diff={:.2e}",
                    i,
                    k,
                    expected,
                    actual,
                    (expected - actual).abs()
                );
            }
        }
    }
}
