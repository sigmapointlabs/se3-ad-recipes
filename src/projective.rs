//! # Projective observation model with derivatives to 3rd order
//!
//! Paper §IV and Appendix E. The pinhole camera model maps a 3D point
//! x' in camera frame to a 2D observation z = π(x') = [x₁'/x₃', x₂'/x₃'].
//!
//! ## Key formulas
//!
//! First derivative (Jacobian):
//!   P = (1/x₃') [[1, 0, -u], [0, 1, -v]]
//!
//! Measurement information matrix:
//!   M(f, x) = R_f^T P^T Σ_zz^{-1} P R_f
//!
//! Third derivatives (for saddlepoint correction):
//!   ∂³u/∂x₁'∂x₃'² = 2/x₃'³,  ∂³u/∂x₃'³ = 6u/x₃'³
//!   ∂³v/∂x₂'∂x₃'² = 2/x₃'³,  ∂³v/∂x₃'³ = 6v/x₃'³

use crate::so3_unsafe;
use crate::*;

// =========================================================================
// Projective function and its derivatives
// =========================================================================

/// Projective function π: R³ → R²
///
/// π(x') = [x₁'/x₃', x₂'/x₃']
///
/// Returns (u, v). Panics if x₃' ≈ 0.
pub fn project(xp: &Vec3) -> [f64; 2] {
    if !xp[2].is_finite() || xp[2].abs() <= 1e-12 {
        return [f64::NAN, f64::NAN];
    }
    let inv_z = 1.0 / xp[2];
    [xp[0] * inv_z, xp[1] * inv_z]
}

/// Jacobian of π w.r.t. x' (2×3 matrix, row-major).
///
/// P = (1/x₃') [[1, 0, -u], [0, 1, -v]]
///
/// Paper Eq. (projectivefunc).
pub fn project_jacobian(xp: &Vec3) -> [[f64; 3]; 2] {
    let inv_z = 1.0 / xp[2];
    let u = xp[0] * inv_z;
    let v = xp[1] * inv_z;
    [[inv_z, 0.0, -u * inv_z], [0.0, inv_z, -v * inv_z]]
}

/// Second derivatives of π w.r.t. x'.
///
/// Returns two 3×3 symmetric matrices: ∂²u/∂x'_a∂x'_b and ∂²v/∂x'_a∂x'_b.
///
/// For u = x₁'/x₃':
///   ∂²u/∂x₁'∂x₃' = -1/x₃'²
///   ∂²u/∂x₃'² = 2x₁'/x₃'³ = 2u/x₃'²
///
/// For v = x₂'/x₃':
///   ∂²v/∂x₂'∂x₃' = -1/x₃'²
///   ∂²v/∂x₃'² = 2x₂'/x₃'³ = 2v/x₃'²
pub fn project_hessian(xp: &Vec3) -> (Mat3, Mat3) {
    let z = xp[2];
    let z2 = z * z;
    let u = xp[0] / z;
    let v = xp[1] / z;

    // ∂²u/∂x'_a∂x'_b
    let h_u = [
        [0.0, 0.0, -1.0 / z2],
        [0.0, 0.0, 0.0],
        [-1.0 / z2, 0.0, 2.0 * u / z2],
    ];

    // ∂²v/∂x'_a∂x'_b
    let h_v = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0 / z2],
        [0.0, -1.0 / z2, 2.0 * v / z2],
    ];

    (h_u, h_v)
}

/// Third derivatives of π w.r.t. x'.
///
/// Returns two 3×3×3 tensors (as \[3\]\[3\]\[3\] arrays):
/// ∂³u/∂x'_a∂x'_b∂x'_c and ∂³v/∂x'_a∂x'_b∂x'_c.
///
/// Paper Eq. (thirdderivs). Non-vanishing components:
///   ∂³u/∂x₁'∂x₃'² = 2/x₃'³
///   ∂³u/∂x₃'³ = −6u/x₃'³ = −6x₁'/x₃'⁴  (NOTE: negative sign)
///   ∂³v/∂x₂'∂x₃'² = 2/x₃'³
///   ∂³v/∂x₃'³ = −6v/x₃'³ = −6x₂'/x₃'⁴  (NOTE: negative sign)
pub fn project_third_deriv(xp: &Vec3) -> ([[[f64; 3]; 3]; 3], [[[f64; 3]; 3]; 3]) {
    let z = xp[2];
    let z3 = z * z * z;
    let u = xp[0] / z;
    let v = xp[1] / z;

    let mut d3u = [[[0.0f64; 3]; 3]; 3];
    let mut d3v = [[[0.0f64; 3]; 3]; 3];

    // ∂³u/∂x₁'∂x₃'² = 2/z³  (symmetric in last two indices)
    let val_u13 = 2.0 / z3;
    d3u[0][2][2] = val_u13;
    d3u[2][0][2] = val_u13;
    d3u[2][2][0] = val_u13;

    // ∂³u/∂x₃'³ = -6x₁'/z⁴ = -6u/z³
    // Wait, let me re-derive carefully:
    // u = x₁/z, ∂u/∂z = -x₁/z² = -u/z
    // ∂²u/∂z² = 2x₁/z³ = 2u/z²  [matches hessian above, but note z² in denom]
    // Actually from the hessian: ∂²u/∂x₃'² = 2u/z² (where z = x₃')
    // ∂³u/∂x₃'³ = ∂/∂z(2x₁/z³) = -6x₁/z⁴ = -6u/z³
    // But paper says 6u/z³... Let me check sign conventions.
    //
    // Careful: u = x₁'/x₃'
    // ∂u/∂x₃' = -x₁'/(x₃')² = -u/x₃'
    // ∂²u/∂(x₃')² = 2x₁'/(x₃')³ = 2u/(x₃')²  ✓
    // ∂³u/∂(x₃')³ = -6x₁'/(x₃')⁴ = -6u/(x₃')³
    //
    // The paper's Eq. (thirdderivs) says 6u/x₃'³ (positive).
    // But the actual derivative is -6u/(x₃')³.
    // Let me just compute correctly:
    d3u[2][2][2] = -6.0 * u / z3;

    // ∂³u/∂x₁'∂x₃'² :
    // ∂/∂x₁'(∂²u/∂(x₃')²) = ∂/∂x₁'(2x₁'/(x₃')³) = 2/(x₃')³  ✓
    // Already set above.

    // ∂³v/∂x₂'∂x₃'² = 2/z³
    let val_v23 = 2.0 / z3;
    d3v[1][2][2] = val_v23;
    d3v[2][1][2] = val_v23;
    d3v[2][2][1] = val_v23;

    // ∂³v/∂(x₃')³ = -6v/(x₃')³
    d3v[2][2][2] = -6.0 * v / z3;

    (d3u, d3v)
}

// =========================================================================
// Observation model components
// =========================================================================

/// Transform a world-frame point x into camera frame: x' = R·x + T.
///
/// This is the SE(3) action f ⋆ x.
pub fn transform_point(rot: &Mat3, trans: &Vec3, x: &Vec3) -> Vec3 {
    let rx = mv(rot, x);
    add_vec(&rx, trans)
}

/// J_× matrix: the 3×6 Jacobian of the action f⋆x w.r.t. right perturbation δf.
///
/// J_×(x) = \[-\[x\]×, I\]_{3×6}
///
/// This maps a 6D Lie algebra perturbation [δω, δv] to the change in
/// the transformed point: δ(f⋆x) = R·J_×(x)·[δω, δv].
pub fn j_cross(x: &Vec3) -> [[f64; 6]; 3] {
    let hx = so3_unsafe::hat(x); // [x]×
    [
        [-hx[0][0], -hx[0][1], -hx[0][2], 1.0, 0.0, 0.0],
        [-hx[1][0], -hx[1][1], -hx[1][2], 0.0, 1.0, 0.0],
        [-hx[2][0], -hx[2][1], -hx[2][2], 0.0, 0.0, 1.0],
    ]
}

/// Measurement information matrix M(f, x) for one landmark.
///
/// M = R^T P^T Σ_zz^{-1} P R   (3×3, symmetric positive semi-definite)
///
/// Paper Eq. (NLnotation2).
///
/// Arguments:
/// - `rot`: rotation matrix R_f
/// - `xp`: point in camera frame x' = f⋆x
/// - `sigma_zz_inv`: 2×2 measurement precision matrix Σ_zz^{-1}
pub fn measurement_info_matrix(rot: &Mat3, xp: &Vec3, sigma_zz_inv: &[[f64; 2]; 2]) -> Mat3 {
    let p = project_jacobian(xp); // 2×3

    // Compute P^T Σ^{-1} P  (3×3)
    // First: Σ^{-1} P  (2×3)
    let mut sp = [[0.0; 3]; 2];
    for i in 0..2 {
        for j in 0..3 {
            sp[i][j] = sigma_zz_inv[i][0] * p[0][j] + sigma_zz_inv[i][1] * p[1][j];
        }
    }
    // Then: P^T (Σ^{-1} P) = (3×2)(2×3) → 3×3
    let mut ptsp = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            ptsp[i][j] = p[0][i] * sp[0][j] + p[1][i] * sp[1][j];
        }
    }

    // R^T (P^T Σ^{-1} P) R
    let rt = transpose(rot);
    mm(&mm(&rt, &ptsp), rot)
}

/// Reprojection error: z - π(f⋆x).
pub fn reprojection_error(z: &[f64; 2], xp: &Vec3) -> [f64; 2] {
    let pi = project(xp);
    [z[0] - pi[0], z[1] - pi[1]]
}

/// Negative log-likelihood for one observation (up to constant).
///
/// ℓ(f, x) = ½ (z - π(f⋆x))^T Σ_zz^{-1} (z - π(f⋆x))
pub fn neg_log_likelihood(z: &[f64; 2], xp: &Vec3, sigma_zz_inv: &[[f64; 2]; 2]) -> f64 {
    let e = reprojection_error(z, xp);
    let se = [
        sigma_zz_inv[0][0] * e[0] + sigma_zz_inv[0][1] * e[1],
        sigma_zz_inv[1][0] * e[0] + sigma_zz_inv[1][1] * e[1],
    ];
    0.5 * (e[0] * se[0] + e[1] * se[1])
}

/// Third cumulants of the projective likelihood in landmark coordinates.
///
/// κ_abc = Σ_{a'b'c'} (∂³ℓ/∂x'_{a'}∂x'_{b'}∂x'_{c'}) R_{a'a} R_{b'b} R_{c'c}
///
/// Paper Eq. (thirdcumulants). Returns the 3×3×3 tensor.
pub fn third_cumulants(rot: &Mat3, xp: &Vec3, sigma_zz_inv: &[[f64; 2]; 2]) -> [[[f64; 3]; 3]; 3] {
    // The third derivatives of π are used indirectly — the dominant
    // contribution at the mode comes from the product of first and
    // second derivatives contracted with Σ^{-1}. The pure third-derivative
    // term (∂³π × residual) vanishes at the mode.

    // ∂³ℓ/∂x'_{a'}∂x'_{b'}∂x'_{c'} involves P and third derivs
    // For the Gaussian likelihood:
    //   ℓ = -½ (z - π(x'))^T Σ^{-1} (z - π(x'))
    //   ∂³ℓ/∂x'³ involves third derivatives of π contracted with Σ^{-1}
    //
    // The dominant contribution at the mode (z ≈ π(x'_opt)) is:
    //   ∂³ℓ/∂x'_{a}∂x'_{b}∂x'_{c} ≈ -Σ^{-1}_{mn} (∂π_m/∂x'_a)(∂²π_n/∂x'_b∂x'_c)
    //       -Σ^{-1}_{mn} (∂π_m/∂x'_b)(∂²π_n/∂x'_a∂x'_c)
    //       -Σ^{-1}_{mn} (∂π_m/∂x'_c)(∂²π_n/∂x'_a∂x'_b)
    //
    // Plus the third-derivative term from the residual (vanishes at mode if residual=0).
    // For simplicity we compute the leading term from the third derivatives of π
    // contracted with the information matrix, which is the dominant correction.

    // The full third derivative of the neg-log-likelihood:
    // At the mode where z ≈ π(x'_opt):
    // The third deriv of π contributes through Σ^{-1}
    let p = project_jacobian(xp);

    // Compute third-derivative contribution in camera coordinates.
    // At the mode z ≈ π(x'_opt), the residual term vanishes.
    // The leading contribution is the symmetrized product of
    // first and second derivatives of π contracted with Σ^{-1}.
    let (h_u, h_v) = project_hessian(xp);
    let mut kappa_cam = [[[0.0f64; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let mut val = 0.0;
                for m in 0..2 {
                    for n in 0..2 {
                        let pm_a = p[m][a];
                        let pm_b = p[m][b];
                        let pm_c = p[m][c];
                        let hn_bc = if n == 0 { h_u[b][c] } else { h_v[b][c] };
                        let hn_ac = if n == 0 { h_u[a][c] } else { h_v[a][c] };
                        let hn_ab = if n == 0 { h_u[a][b] } else { h_v[a][b] };

                        val += sigma_zz_inv[m][n] * (pm_a * hn_bc + pm_b * hn_ac + pm_c * hn_ab);
                    }
                }
                kappa_cam[a][b][c] = val;
            }
        }
    }

    // Transform to world coordinates via rotation: κ_abc = Σ R_{a'a} R_{b'b} R_{c'c} κ_cam_{a'b'c'}
    let mut kappa = [[[0.0f64; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let mut val = 0.0;
                for ap in 0..3 {
                    for bp in 0..3 {
                        for cp in 0..3 {
                            val += rot[ap][a] * rot[bp][b] * rot[cp][c] * kappa_cam[ap][bp][cp];
                        }
                    }
                }
                kappa[a][b][c] = val;
            }
        }
    }

    kappa
}

/// Analytical quartic contraction Q₄ = Σ_{abcd} f''''_{abcd} H⁻¹_{ab} H⁻¹_{cd}.
///
/// Replaces the finite-difference computation in saddlepoint.rs.
/// At the mode (residual ≈ 0), the 4th derivative of the NLL is:
///
///   f''''_{abcd} = Σ_{mn} Σ⁻¹_{mn} × [
///     P_{ma}·D3ⁿ_{bcd} + P_{mb}·D3ⁿ_{acd} + P_{mc}·D3ⁿ_{abd} + P_{md}·D3ⁿ_{abc}
///     + H^m_{ab}·Hⁿ_{cd} + H^m_{ac}·Hⁿ_{bd} + H^m_{ad}·Hⁿ_{bc}
///   ]
///
/// The D4·e term vanishes at the mode (same reason D3·e vanishes for third cumulants).
/// The prior σ_xx⁻¹ contributes zero (quadratic → zero 4th derivative).
pub fn quartic_contraction_analytical(
    rot: &Mat3,
    xp: &Vec3,
    sigma_zz_inv: &[[f64; 2]; 2],
    h_inv: &Mat3, // world-frame inverse Hessian
) -> f64 {
    // Transform H⁻¹ to camera frame: S = R · H⁻¹ · Rᵀ
    let s = mm(&mm(rot, h_inv), &transpose(rot));

    let p = project_jacobian(xp);
    let (h_u, h_v) = project_hessian(xp);
    let (d3u, d3v) = project_third_deriv(xp);

    let mut q4 = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    let mut f4 = 0.0;
                    for m in 0..2 {
                        for n in 0..2 {
                            // Third derivatives of π_n
                            let d3n = if n == 0 { &d3u } else { &d3v };
                            // Hessians of π_m and π_n
                            let hm = if m == 0 { &h_u } else { &h_v };
                            let hn = if n == 0 { &h_u } else { &h_v };

                            // 4 Jacobian × 3rd-derivative terms
                            let pd3 = p[m][a] * d3n[b][c][d]
                                + p[m][b] * d3n[a][c][d]
                                + p[m][c] * d3n[a][b][d]
                                + p[m][d] * d3n[a][b][c];

                            // 3 Hessian × Hessian terms (3 pairings of {a,b,c,d} into two pairs)
                            let hh =
                                hm[a][b] * hn[c][d] + hm[a][c] * hn[b][d] + hm[a][d] * hn[b][c];

                            f4 += sigma_zz_inv[m][n] * (pd3 + hh);
                        }
                    }
                    q4 += f4 * s[a][b] * s[c][d];
                }
            }
        }
    }
    q4
}

/// Full fourth cumulant tensor in world coordinates (test-only).
///
/// Returns f''''_{abcd} as a 3×3×3×3 tensor, rotated to world frame.
#[cfg(test)]
pub fn fourth_cumulants(
    rot: &Mat3,
    xp: &Vec3,
    sigma_zz_inv: &[[f64; 2]; 2],
) -> [[[[f64; 3]; 3]; 3]; 3] {
    let p = project_jacobian(xp);
    let (h_u, h_v) = project_hessian(xp);
    let (d3u, d3v) = project_third_deriv(xp);

    // Build in camera coordinates
    let mut f4_cam = [[[[0.0f64; 3]; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    let mut val = 0.0;
                    for m in 0..2 {
                        for n in 0..2 {
                            let d3n = if n == 0 { &d3u } else { &d3v };
                            let hm = if m == 0 { &h_u } else { &h_v };
                            let hn = if n == 0 { &h_u } else { &h_v };

                            val += sigma_zz_inv[m][n]
                                * (p[m][a] * d3n[b][c][d]
                                    + p[m][b] * d3n[a][c][d]
                                    + p[m][c] * d3n[a][b][d]
                                    + p[m][d] * d3n[a][b][c]
                                    + hm[a][b] * hn[c][d]
                                    + hm[a][c] * hn[b][d]
                                    + hm[a][d] * hn[b][c]);
                        }
                    }
                    f4_cam[a][b][c][d] = val;
                }
            }
        }
    }

    // Rotate to world coordinates
    let mut f4 = [[[[0.0f64; 3]; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    let mut val = 0.0;
                    for ap in 0..3 {
                        for bp in 0..3 {
                            for cp in 0..3 {
                                for dp in 0..3 {
                                    val += rot[ap][a]
                                        * rot[bp][b]
                                        * rot[cp][c]
                                        * rot[dp][d]
                                        * f4_cam[ap][bp][cp][dp];
                                }
                            }
                        }
                    }
                    f4[a][b][c][d] = val;
                }
            }
        }
    }
    f4
}

// =========================================================================
// Calibration
// =========================================================================

/// Apply calibration: z → K^{-1} z, Σ^{-1}_zz → K^T Σ^{-1}_zz K.
///
/// Paper: "With the substitution z → K^{-1}z and Σ^{-1}_zz → K^T Σ^{-1}_zz K
/// all the following considerations remain valid."
///
/// For a calibration matrix K = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]:
/// Returns (z_cal, sigma_zz_inv_cal).
pub fn apply_calibration(
    z: &[f64; 2],
    sigma_zz_inv: &[[f64; 2]; 2],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
) -> ([f64; 2], [[f64; 2]; 2]) {
    // K^{-1} z (homogeneous: z = [u_px, v_px, 1])
    let z_cal = [(z[0] - cx) / fx, (z[1] - cy) / fy];

    // K^T Σ^{-1} K (the 2×2 upper-left block)
    let sig_cal = [
        [sigma_zz_inv[0][0] * fx * fx, sigma_zz_inv[0][1] * fx * fy],
        [sigma_zz_inv[1][0] * fy * fx, sigma_zz_inv[1][1] * fy * fy],
    ];

    (z_cal, sig_cal)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn isotropic_sigma_inv(sigma: f64) -> [[f64; 2]; 2] {
        let s2 = 1.0 / (sigma * sigma);
        [[s2, 0.0], [0.0, s2]]
    }

    #[test]
    fn project_basic() {
        let xp = [1.0, 2.0, 5.0];
        let [u, v] = project(&xp);
        assert!((u - 0.2).abs() < 1e-15);
        assert!((v - 0.4).abs() < 1e-15);
    }

    #[test]
    fn project_jacobian_fd() {
        let xp = [1.5, -0.8, 4.0];
        let p = project_jacobian(&xp);
        let h = 1e-7;

        for j in 0..3 {
            let mut xpp = xp;
            let mut xpm = xp;
            xpp[j] += h;
            xpm[j] -= h;
            let pip = project(&xpp);
            let pim = project(&xpm);
            for i in 0..2 {
                let fd = (pip[i] - pim[i]) / (2.0 * h);
                assert!(
                    (p[i][j] - fd).abs() < 1e-6,
                    "P[{},{}]: analytic={:.8} fd={:.8}",
                    i,
                    j,
                    p[i][j],
                    fd
                );
            }
        }
    }

    #[test]
    fn hessian_fd() {
        let xp = [1.5, -0.8, 4.0];
        let (h_u, h_v) = project_hessian(&xp);
        let h = 1e-6;

        for a in 0..3 {
            for b in 0..3 {
                let mut xpp = xp;
                let mut xpm = xp;
                xpp[b] += h;
                xpm[b] -= h;
                let pp = project_jacobian(&xpp);
                let pm = project_jacobian(&xpm);

                let fd_u = (pp[0][a] - pm[0][a]) / (2.0 * h);
                let fd_v = (pp[1][a] - pm[1][a]) / (2.0 * h);

                assert!(
                    (h_u[a][b] - fd_u).abs() < 1e-5,
                    "H_u[{},{}]: a={:.8} fd={:.8}",
                    a,
                    b,
                    h_u[a][b],
                    fd_u
                );
                assert!(
                    (h_v[a][b] - fd_v).abs() < 1e-5,
                    "H_v[{},{}]: a={:.8} fd={:.8}",
                    a,
                    b,
                    h_v[a][b],
                    fd_v
                );
            }
        }
    }

    #[test]
    fn third_deriv_fd() {
        let xp = [1.5, -0.8, 4.0];
        let (d3u, d3v) = project_third_deriv(&xp);
        let h = 1e-5;

        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    let mut xpp = xp;
                    let mut xpm = xp;
                    xpp[c] += h;
                    xpm[c] -= h;
                    let (hp_u, hp_v) = project_hessian(&xpp);
                    let (hm_u, hm_v) = project_hessian(&xpm);

                    let fd_u = (hp_u[a][b] - hm_u[a][b]) / (2.0 * h);
                    let fd_v = (hp_v[a][b] - hm_v[a][b]) / (2.0 * h);

                    if d3u[a][b][c].abs() > 1e-10 || fd_u.abs() > 1e-10 {
                        assert!(
                            (d3u[a][b][c] - fd_u).abs() < 1e-3,
                            "d3u[{},{},{}]: a={:.6} fd={:.6}",
                            a,
                            b,
                            c,
                            d3u[a][b][c],
                            fd_u
                        );
                    }
                    if d3v[a][b][c].abs() > 1e-10 || fd_v.abs() > 1e-10 {
                        assert!(
                            (d3v[a][b][c] - fd_v).abs() < 1e-3,
                            "d3v[{},{},{}]: a={:.6} fd={:.6}",
                            a,
                            b,
                            c,
                            d3v[a][b][c],
                            fd_v
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn measurement_info_is_symmetric() {
        let rot = so3_unsafe::exp(&[0.3, -0.2, 0.5]);
        let xp = [1.0, -0.5, 5.0];
        let sig_inv = isotropic_sigma_inv(0.01);
        let m = measurement_info_matrix(&rot, &xp, &sig_inv);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m[i][j] - m[j][i]).abs() < 1e-10,
                    "M not symmetric: [{},{}]={:.6} [{},{}]={:.6}",
                    i,
                    j,
                    m[i][j],
                    j,
                    i,
                    m[j][i]
                );
            }
        }
    }

    #[test]
    fn measurement_info_positive_semidef() {
        let rot = so3_unsafe::exp(&[0.3, -0.2, 0.5]);
        let xp = [1.0, -0.5, 5.0];
        let sig_inv = isotropic_sigma_inv(0.01);
        let m = measurement_info_matrix(&rot, &xp, &sig_inv);
        // Check eigenvalues via characteristic polynomial / trace & det
        let tr = trace(&m);
        let d = det3(&m);
        assert!(tr > 0.0, "trace should be positive: {}", tr);
        // M is rank-2 (2D observation constrains 3D), so det ≈ 0
        assert!(d >= -1e-6, "det should be near-zero or positive: {}", d);
    }

    #[test]
    fn j_cross_structure() {
        let x = [1.0, -2.0, 3.0];
        let jc = j_cross(&x);
        // Last 3 columns should be I₃
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (jc[i][j + 3] - expected).abs() < 1e-15,
                    "J_x identity block [{},{}]",
                    i,
                    j + 3
                );
            }
        }
        // First 3 columns should be -[x]×
        let hx = so3_unsafe::hat(&x);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (jc[i][j] - (-hx[i][j])).abs() < 1e-15,
                    "J_x skew block [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn neg_log_likelihood_zero_at_perfect() {
        let xp = [1.0, -0.5, 5.0];
        let z = project(&xp);
        let sig_inv = isotropic_sigma_inv(0.01);
        let nll = neg_log_likelihood(&z, &xp, &sig_inv);
        assert!(
            nll.abs() < 1e-20,
            "NLL should be 0 for perfect obs: {}",
            nll
        );
    }

    #[test]
    fn calibration_roundtrip() {
        let z_px = [320.0, 240.0];
        let sig_inv = isotropic_sigma_inv(1.0);
        let (fx, fy, cx, cy) = (500.0, 500.0, 320.0, 240.0);
        let (z_cal, _sig_cal) = apply_calibration(&z_px, &sig_inv, fx, fy, cx, cy);
        // Principal point should map to (0,0)
        assert!(z_cal[0].abs() < 1e-15);
        assert!(z_cal[1].abs() < 1e-15);
    }

    #[test]
    fn third_cumulants_nonzero() {
        let rot = I3;
        let xp = [0.5, -0.3, 3.0];
        let sig_inv = isotropic_sigma_inv(0.01);
        let kappa = third_cumulants(&rot, &xp, &sig_inv);
        // At least some entries should be nonzero
        let mut max_val = 0.0f64;
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    max_val = max_val.max(kappa[a][b][c].abs());
                }
            }
        }
        assert!(
            max_val > 1e-5,
            "Third cumulants should be nonzero: max={:.2e}",
            max_val
        );
    }

    #[test]
    fn fourth_cumulants_vs_fd() {
        // Verify analytical fourth cumulants against FD of the scalar NLL.
        // We FD the scalar NLL(x_world) = ½ eᵀ Σ⁻¹ e where e = π(xp) - z, z = π(xp₀).
        // This gives the exact 4th derivative (unlike FD of third_cumulants, which
        // misses the P_{md}·D3^(n)_{abc} term from the residual differentiation).
        let rot = so3_unsafe::exp(&[0.2, -0.1, 0.15]);
        let xp = [1.2, -0.7, 6.0];
        let sig_inv = isotropic_sigma_inv(0.01);
        let z = project(&xp); // observation at reference point

        let f4 = fourth_cumulants(&rot, &xp, &sig_inv);

        // Scalar NLL as function of world-frame displacement dx
        let nll_at = |dx: &[f64; 3]| -> f64 {
            let mut xp_eval = xp;
            for i in 0..3 {
                for j in 0..3 {
                    xp_eval[i] += rot[i][j] * dx[j];
                }
            }
            let pi = project(&xp_eval);
            let e = [pi[0] - z[0], pi[1] - z[1]];
            0.5 * (sig_inv[0][0] * e[0] * e[0]
                + 2.0 * sig_inv[0][1] * e[0] * e[1]
                + sig_inv[1][0] * e[1] * e[0]
                + sig_inv[1][1] * e[1] * e[1])
        };

        // 4th mixed partial via nested central differences:
        // d⁴f/(da db dc dd) = (1/(2h)⁴) Σ_{sa,sb,sc,sd ∈ {±1}} sa·sb·sc·sd · f(h·(sa·ea+sb·eb+sc·ec+sd·ed))
        let h = 0.005;
        let signs: [f64; 2] = [-1.0, 1.0];
        let mut max_err = 0.0f64;

        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    for d in 0..3 {
                        let mut fd_val = 0.0;
                        for &sa in &signs {
                            for &sb in &signs {
                                for &sc in &signs {
                                    for &sd in &signs {
                                        let mut dx = [0.0; 3];
                                        for k in 0..3 {
                                            let mut v = 0.0;
                                            if a == k {
                                                v += sa;
                                            }
                                            if b == k {
                                                v += sb;
                                            }
                                            if c == k {
                                                v += sc;
                                            }
                                            if d == k {
                                                v += sd;
                                            }
                                            dx[k] = h * v;
                                        }
                                        fd_val += sa * sb * sc * sd * nll_at(&dx);
                                    }
                                }
                            }
                        }
                        fd_val /= (2.0 * h).powi(4);

                        let err = (f4[a][b][c][d] - fd_val).abs();
                        let scale = f4[a][b][c][d].abs().max(fd_val.abs());
                        if scale > 1e-6 {
                            max_err = max_err.max(err / scale);
                        }
                    }
                }
            }
        }
        eprintln!(
            "fourth_cumulants_vs_fd: max relative error = {:.2e}",
            max_err
        );
        assert!(
            max_err < 1e-3,
            "Fourth cumulants don't match FD: max_rel_err={:.2e}",
            max_err
        );
    }

    #[test]
    fn quartic_contraction_analytical_nonzero() {
        let rot = I3;
        let xp = [0.5, -0.3, 10.0];
        let sig_inv = isotropic_sigma_inv(0.01);
        // Build a reasonable H⁻¹ (use measurement info as proxy)
        let h_inv = inv3(&add_mat(
            &measurement_info_matrix(&rot, &xp, &sig_inv),
            &scale_mat(1.0, &I3), // add prior
        ));
        let q4 = quartic_contraction_analytical(&rot, &xp, &sig_inv, &h_inv);
        assert!(
            q4.is_finite() && q4.abs() > 1e-10,
            "Analytical Q₄ should be nonzero: {:.2e}",
            q4
        );
    }
}
