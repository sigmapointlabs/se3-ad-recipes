//! Gaussian moment contractions (Isserlis / Wick) for tensor-based
//! uncertainty transport.
//!
//! Raw-tier module: no SemVer stability promise (see [`crate::api`] for
//! the curated tiers); a candidate for `api::expert` promotion once the
//! covariance-transport application stabilizes its conventions.
//!
//! For a smooth map φ: ℝᴺ → ℝᴺ expanded at the origin as
//! `φ(ξ) ≈ φ(0) + J·ξ + ½ H:ξξ + ⅙ C:ξξξ` with `ξ ~ N(0, Σ)`, these
//! contractions give the transported moments complete through O(σ⁴):
//!
//! ```text
//! E[φ(ξ)]   ≈ φ(0) + ½ H:Σ                        second_order_mean_shift_g
//! Cov(φ(ξ)) ≈ J Σ Jᵀ + Δ_QQ + Δ_LC + O(σ⁶)
//!   Δ_QQ[i][j] = ½ tr(Hᵢ Σ Hⱼ Σ)                  isserlis_covariance_correction_g
//!   Δ_LC       = ½ (J Σ Mᵀ + M Σ Jᵀ),  M = C:Σ    linear_cubic_covariance_correction_g
//! ```
//!
//! Everything is dimension-generic (`const N: usize`): the same three
//! functions serve SE(3) chart maps at `N = 6` and SE₂(3) at `N = 9`.
//! Tensor index order is `[out][in]…[in]` with symmetry in the trailing
//! input indices, matching [`crate::jacobians_ad::Tensor666G`] /
//! [`crate::jacobians_ad::Tensor6666G`] and their producers
//! [`crate::jacobians_ad::recentering_hessian_g`] /
//! [`crate::jacobians_ad::recentering_hessian_at_g`].  The cubic tensor
//! need not come from an analytical stack: one `adfn<N>`-seeded evaluation
//! of the general-chart Hessian yields `C[i][p][q][r]` as the tangent of
//! `H[i][p][q]` (the mixed-AD recipe; see the in-tree transport test for
//! the five-line construction).

use crate::autodiff::ad_trait::AD;

/// a·b for N×N `T`-valued matrices (local helper; keeps the module
/// free-standing).
fn matmul_g<T: AD, const N: usize>(a: &[[T; N]; N], b: &[[T; N]; N]) -> [[T; N]; N] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let mut s = T::constant(0.0);
            for k in 0..N {
                s += a[i][k] * b[k][j];
            }
            s
        })
    })
}

/// Second-order mean shift of a zero-mean Gaussian pushed through a chart
/// map with Hessian tensor `h` (index order `[out][in][in]`, e.g. from
/// [`crate::jacobians_ad::recentering_hessian_g`]).  For `ξ ~ N(0, Σ)` and
/// `φ(ξ) ≈ φ(0) + J·ξ + ½ H:ξξ`:
///
/// ```text
/// E[φ(ξ)] ≈ φ(0) + μ,    μ_i = ½ Σ_pq H[i][p][q] · Σ[p][q]
/// ```
pub fn second_order_mean_shift_g<T: AD, const N: usize>(
    h: &[[[T; N]; N]; N],
    sigma: &[[T; N]; N],
) -> [T; N] {
    let half = T::constant(0.5);
    std::array::from_fn(|i| {
        let mut acc = T::constant(0.0);
        for p in 0..N {
            for q in 0..N {
                acc += h[i][p][q] * sigma[p][q];
            }
        }
        half * acc
    })
}

/// Isserlis (Wick) covariance correction — the quadratic×quadratic term.
/// With `y = J·ξ + ½ H:ξξ` and `ξ ~ N(0, Σ)`, the Gaussian fourth-moment
/// identity `E[ξ_a ξ_b ξ_c ξ_d] = Σ_ab Σ_cd + Σ_ac Σ_bd + Σ_ad Σ_bc`
/// reduces the quartic term to
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
pub fn isserlis_covariance_correction_g<T: AD, const N: usize>(
    h: &[[[T; N]; N]; N],
    sigma: &[[T; N]; N],
) -> [[T; N]; N] {
    let z = T::constant(0.0);
    let half = T::constant(0.5);
    // P_i = H_i·Σ;  Δ_ij = ½ tr(P_i·P_j) = ½ Σ_ab P_i[a][b]·P_j[b][a].
    let p: [[[T; N]; N]; N] = std::array::from_fn(|i| matmul_g(&h[i], sigma));
    let mut delta = [[z; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let mut acc = z;
            for a in 0..N {
                for b in 0..N {
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
pub fn linear_cubic_covariance_correction_g<T: AD, const N: usize>(
    jac: &[[T; N]; N],
    cubic: &[[[[T; N]; N]; N]; N],
    sigma: &[[T; N]; N],
) -> [[T; N]; N] {
    let z = T::constant(0.0);
    let half = T::constant(0.5);

    // M[j][p] = Σ_qr C[j][p][q][r]·Σ[q][r]
    let mut m = [[z; N]; N];
    for j in 0..N {
        for p in 0..N {
            let mut acc = z;
            for q in 0..N {
                for r in 0..N {
                    acc += cubic[j][p][q][r] * sigma[q][r];
                }
            }
            m[j][p] = acc;
        }
    }

    // A = J·Σ·Mᵀ;  Δ_LC = ½ (A + Aᵀ).
    let js = matmul_g(jac, sigma);
    let mut delta = [[z; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut a_ij = z;
            let mut a_ji = z;
            for k in 0..N {
                a_ij += js[i][k] * m[j][k];
                a_ji += js[j][k] * m[i][k];
            }
            delta[i][j] = half * (a_ij + a_ji);
        }
    }
    delta
}

// =========================================================================
// Tests — instantiated at N = 6 on the SE(3) recentering map.
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jacobians_ad::{Tensor6666G, recentering_hessian_at_g, recentering_hessian_g};
    use crate::se3_adsafe::{Mat6G, PoseG, Vec6G, mv6_g, se3_jr_inv_g};
    use crate::test_support::Rng;

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

    fn draw_xi(rng: &mut Rng, l: &[[f64; 6]; 6]) -> Vec6G<f64> {
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

        let mu = second_order_mean_shift_g(&h, &sigma);
        let delta = isserlis_covariance_correction_g(&h, &sigma);

        let n = 200_000usize;
        let mut rng = Rng::new(42);
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
        let mu = second_order_mean_shift_g(&h, &sigma);
        let d_qq = isserlis_covariance_correction_g(&h, &sigma);
        let d_lc = linear_cubic_covariance_correction_g(&jac, &cubic, &sigma);

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
