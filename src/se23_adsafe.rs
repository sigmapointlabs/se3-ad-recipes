//! # SE_2(3) — Extended Pose Group, AD-safe.
//!
//! Templated `<T: AD>` SE_2(3) primitives: `ExtendedPoseG`, the 9-dim Vec/Mat
//! aliases the algebra lives on, and exp/log/compose/inverse.  Built on top of
//! [`crate::so3_adsafe`], which supplies `so3_exp_g`, `so3_log_g`,
//! `so3_log_g_quaternion`, `v_matrix_g`, and `v_inv_g`.  The SE_2(3) coupling
//! matrix is the same SE(3) `V` applied independently to both velocity and
//! position — that's what makes SE_2(3) "two SE(3)'s sharing a rotation"
//! rather than a genuinely new group.
//!
//! ## Group structure
//!
//! State ξ = (R, v, p) ∈ SO(3) × ℝ³ × ℝ³, with the 5×5 matrix embedding
//!
//! ```text
//!     | R  v  p |
//! ξ = | 0  1  0 |
//!     | 0  0  1 |
//! ```
//!
//! Composition `(R₁, v₁, p₁) · (R₂, v₂, p₂) = (R₁R₂,  R₁v₂ + v₁,  R₁p₂ + p₁)`:
//! both `v₂` and `p₂` transform identically under the left rotation `R₁`, with
//! zero coupling to each other in the Lie bracket.
//!
//! ## Exponential and logarithm
//!
//! `exp([ω; ν; ρ]) = (R(ω), V(ω)·ν, V(ω)·ρ)` with the same `V` (= SE(3) left
//! Jacobian, `so3_adsafe::v_matrix_g`) acting on both ν and ρ.  `log` returns
//! `ξ = [log(R); V⁻¹·v; V⁻¹·p]`.  Two log variants are exposed:
//! [`ExtendedPoseG::log`] (matrix-based) and [`ExtendedPoseG::log_quat`]
//! (via SU(2)) — the latter for AD paths that get unstable near the antipodal
//! locus on the matrix log.
//!
//! ## References
//!
//! - Barrau & Bonnabel (2017), "The Invariant Extended Kalman Filter as a
//!   Stable Observer", IEEE TAC 62(4): §5 inertial navigation on SE_2(3).
//! - Ge, van Goor, Mahony (2025), "The Geometry of EKF on Manifolds with
//!   Affine Connection", arXiv:2506.05728: §6 IMU case study on SE_2(3).

use crate::autodiff::ad_trait::AD;
use crate::so3_adsafe::{
    Mat3G, Vec3G, mm3_g, mv3_g, so3_exp_g, so3_log_g, so3_log_g_quaternion, transpose3_g, v_inv_g,
    v_matrix_g,
};

// =========================================================================
// AD-generic 9-vector / 9-matrix types and ops
// =========================================================================

/// 9-vector over AD scalar T.
pub type Vec9G<T> = [T; 9];
/// 9×9 matrix over AD scalar T, row-major.
pub type Mat9G<T> = [[T; 9]; 9];

/// Build a 9×9 matrix from a 3×3 grid of 3×3 blocks (row-major).
///
/// Used to assemble the SE_2(3) adjoint and right Jacobians from their
/// constituent 3×3 blocks.  Block layout:
///
/// ```text
/// | a  b  c |
/// | d  e  f |
/// | g  h  i |
/// ```
#[allow(clippy::too_many_arguments)]
pub fn blocks_9x9_g<T: AD>(
    a: &Mat3G<T>,
    b: &Mat3G<T>,
    c: &Mat3G<T>,
    d: &Mat3G<T>,
    e: &Mat3G<T>,
    f: &Mat3G<T>,
    g: &Mat3G<T>,
    h: &Mat3G<T>,
    i: &Mat3G<T>,
) -> Mat9G<T> {
    let z = T::constant(0.0);
    let mut m = [[z; 9]; 9];
    let blocks: [[&Mat3G<T>; 3]; 3] = [[a, b, c], [d, e, f], [g, h, i]];
    for br in 0..3 {
        for bc in 0..3 {
            let blk = blocks[br][bc];
            for r in 0..3 {
                for c2 in 0..3 {
                    m[br * 3 + r][bc * 3 + c2] = blk[r][c2];
                }
            }
        }
    }
    m
}

pub fn mv9_g<T: AD>(m: &Mat9G<T>, v: &Vec9G<T>) -> Vec9G<T> {
    let z = T::constant(0.0);
    let mut r = [z; 9];
    for i in 0..9 {
        let mut s = z;
        for j in 0..9 {
            s += m[i][j] * v[j];
        }
        r[i] = s;
    }
    r
}

pub fn mm9_g<T: AD>(a: &Mat9G<T>, b: &Mat9G<T>) -> Mat9G<T> {
    let z = T::constant(0.0);
    let mut c = [[z; 9]; 9];
    for i in 0..9 {
        for j in 0..9 {
            let mut s = z;
            for k in 0..9 {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

// =========================================================================
// SE_2(3) extended pose
// =========================================================================

/// An extended pose ξ = (R, v, p) ∈ SE_2(3), over AD scalar T.
///
/// Field naming follows the inertial-navigation convention:
///   - `rot`: body-to-world rotation R ∈ SO(3)
///   - `vel`: linear velocity v ∈ ℝ³ (typically world-frame)
///   - `pos`: position p ∈ ℝ³ (typically world-frame)
///
/// The struct does not enforce a frame convention — left/right invariance
/// is determined by how composition is used in the surrounding code.
#[derive(Clone, Copy)]
pub struct ExtendedPoseG<T: AD> {
    pub rot: Mat3G<T>,
    pub vel: Vec3G<T>,
    pub pos: Vec3G<T>,
}

impl<T: AD> ExtendedPoseG<T> {
    /// Group composition: `self · other`.
    ///
    /// `(R_a, v_a, p_a) · (R_b, v_b, p_b) = (R_a R_b,  R_a v_b + v_a,  R_a p_b + p_a)`.
    pub fn compose(&self, other: &ExtendedPoseG<T>) -> ExtendedPoseG<T> {
        let rot = mm3_g(&self.rot, &other.rot);
        let rv = mv3_g(&self.rot, &other.vel);
        let rp = mv3_g(&self.rot, &other.pos);
        let vel = [
            rv[0] + self.vel[0],
            rv[1] + self.vel[1],
            rv[2] + self.vel[2],
        ];
        let pos = [
            rp[0] + self.pos[0],
            rp[1] + self.pos[1],
            rp[2] + self.pos[2],
        ];
        ExtendedPoseG { rot, vel, pos }
    }

    /// SE_2(3) exponential map: `[ω; ν; ρ] → (R(ω), V(ω)·ν, V(ω)·ρ)`.
    ///
    /// Same `V` matrix is applied to both ν and ρ.
    pub fn exp(xi: &Vec9G<T>) -> ExtendedPoseG<T> {
        let omega: Vec3G<T> = [xi[0], xi[1], xi[2]];
        let nu: Vec3G<T> = [xi[3], xi[4], xi[5]];
        let rho: Vec3G<T> = [xi[6], xi[7], xi[8]];
        let rot = so3_exp_g(&omega);
        let v = v_matrix_g(&omega);
        let vel = mv3_g(&v, &nu);
        let pos = mv3_g(&v, &rho);
        ExtendedPoseG { rot, vel, pos }
    }

    /// SE_2(3) logarithmic map (matrix-based SO(3) log):
    /// `(R, v, p) → [log(R); V⁻¹·v; V⁻¹·p]`.
    pub fn log(&self) -> Vec9G<T> {
        let omega = so3_log_g(&self.rot);
        let vi = v_inv_g(&omega);
        let nu = mv3_g(&vi, &self.vel);
        let rho = mv3_g(&vi, &self.pos);
        [
            omega[0], omega[1], omega[2], nu[0], nu[1], nu[2], rho[0], rho[1], rho[2],
        ]
    }

    /// SE_2(3) logarithmic map (quaternion-based SO(3) log).
    ///
    /// Same algebra as [`log`](Self::log) but routes the rotation log
    /// through the SU(2) path (`so3_log_g_quaternion`).  Useful when the
    /// matrix path runs into AD instabilities near the antipodal locus.
    pub fn log_quat(&self) -> Vec9G<T> {
        let omega = so3_log_g_quaternion(&self.rot);
        let vi = v_inv_g(&omega);
        let nu = mv3_g(&vi, &self.vel);
        let rho = mv3_g(&vi, &self.pos);
        [
            omega[0], omega[1], omega[2], nu[0], nu[1], nu[2], rho[0], rho[1], rho[2],
        ]
    }

    /// SE_2(3) inverse: `(R, v, p)⁻¹ = (Rᵀ, −Rᵀv, −Rᵀp)`.
    pub fn inverse(&self) -> ExtendedPoseG<T> {
        let rot = transpose3_g(&self.rot);
        let rt_v = mv3_g(&rot, &self.vel);
        let rt_p = mv3_g(&rot, &self.pos);
        let z = T::constant(0.0);
        let vel = [z - rt_v[0], z - rt_v[1], z - rt_v[2]];
        let pos = [z - rt_p[0], z - rt_p[1], z - rt_p[2]];
        ExtendedPoseG { rot, vel, pos }
    }

    // ─── Channel-specific point actions ─────────────────────────────────
    //
    // SE_2(3) has two natural "point action" channels because the group
    // carries two ℝ³ offsets (position p, velocity v) sharing one rotation R.
    // The two channels are exposed as separate inherent methods rather than
    // routed through the [`crate::act::Act`] trait — that trait is reserved
    // for SE(3) types where "the point action" is unambiguous.

    /// Position-channel point action: `y = R·x + p`.
    ///
    /// The natural transform of a 3D position by an SE_2(3) state.
    #[inline]
    pub fn act_position(&self, x: &Vec3G<T>) -> Vec3G<T> {
        let rx = mv3_g(&self.rot, x);
        [
            rx[0] + self.pos[0],
            rx[1] + self.pos[1],
            rx[2] + self.pos[2],
        ]
    }

    /// Inverse position-channel action: `x = Rᵀ·(y − p)`.
    #[inline]
    pub fn act_position_inverse(&self, x: &Vec3G<T>) -> Vec3G<T> {
        let dx = [x[0] - self.pos[0], x[1] - self.pos[1], x[2] - self.pos[2]];
        mv3_g(&transpose3_g(&self.rot), &dx)
    }

    /// Velocity-channel point action: `y = R·x + v`.
    ///
    /// The natural transform of a 3D velocity by an SE_2(3) state.
    #[inline]
    pub fn act_velocity(&self, x: &Vec3G<T>) -> Vec3G<T> {
        let rx = mv3_g(&self.rot, x);
        [
            rx[0] + self.vel[0],
            rx[1] + self.vel[1],
            rx[2] + self.vel[2],
        ]
    }

    /// Inverse velocity-channel action: `x = Rᵀ·(y − v)`.
    #[inline]
    pub fn act_velocity_inverse(&self, x: &Vec3G<T>) -> Vec3G<T> {
        let dx = [x[0] - self.vel[0], x[1] - self.vel[1], x[2] - self.vel[2]];
        mv3_g(&transpose3_g(&self.rot), &dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{I3, Mat3, Vec3, add_vec, mv, sub_vec};

    type ExtendedPose = ExtendedPoseG<f64>;

    fn approx_eq_vec3(a: &Vec3, b: &Vec3, tol: f64) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < tol)
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

    fn identity_ep() -> ExtendedPose {
        ExtendedPoseG {
            rot: I3,
            vel: [0.0; 3],
            pos: [0.0; 3],
        }
    }

    // The AD-generic struct intentionally mirrors `PoseG<T>`: no convenience
    // methods for action / relative.  Inline them here so the structural
    // tests below still read naturally.
    fn act_on_point(g: &ExtendedPose, x: &Vec3) -> Vec3 {
        add_vec(&mv(&g.rot, x), &g.pos)
    }
    fn act_on_vector(g: &ExtendedPose, w: &Vec3) -> Vec3 {
        mv(&g.rot, w)
    }
    fn relative(a: &ExtendedPose, b: &ExtendedPose) -> ExtendedPose {
        a.inverse().compose(b)
    }

    // ─── Identity / action ──────────────────────────────────────────────

    #[test]
    fn identity_action_on_point() {
        let xi = identity_ep();
        let x = [1.0, 2.0, 3.0];
        assert!(approx_eq_vec3(&act_on_point(&xi, &x), &x, 1e-15));
    }

    #[test]
    fn identity_action_on_vector() {
        let xi = identity_ep();
        let w = [0.5, -0.5, 0.5];
        assert!(approx_eq_vec3(&act_on_vector(&xi, &w), &w, 1e-15));
    }

    // ─── Group axioms ───────────────────────────────────────────────────

    #[test]
    fn compose_inverse_is_identity() {
        let xi_tan = [0.3, -0.5, 0.2, 1.0, -2.0, 0.5, 0.8, 1.2, -0.4];
        let g = ExtendedPose::exp(&xi_tan);
        let id = g.compose(&g.inverse());
        assert!(approx_eq_mat3(&id.rot, &I3, 1e-10));
        assert!(approx_eq_vec3(&id.vel, &[0.0; 3], 1e-10));
        assert!(approx_eq_vec3(&id.pos, &[0.0; 3], 1e-10));
    }

    #[test]
    fn inverse_compose_is_identity() {
        let xi_tan = [0.4, 0.1, -0.3, -1.5, 0.7, 2.0, 0.3, -0.6, 0.9];
        let g = ExtendedPose::exp(&xi_tan);
        let id = g.inverse().compose(&g);
        assert!(approx_eq_mat3(&id.rot, &I3, 1e-10));
        assert!(approx_eq_vec3(&id.vel, &[0.0; 3], 1e-10));
        assert!(approx_eq_vec3(&id.pos, &[0.0; 3], 1e-10));
    }

    #[test]
    fn action_composition() {
        // (g · f) ⋆ x == g ⋆ (f ⋆ x)
        let f = ExtendedPose::exp(&[0.2, 0.3, -0.1, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0]);
        let g = ExtendedPose::exp(&[-0.1, 0.4, 0.2, 0.0, 1.0, -0.5, 0.0, 0.5, 0.5]);
        let x = [1.0, -1.0, 3.0];
        let gf_x = act_on_point(&g.compose(&f), &x);
        let g_fx = act_on_point(&g, &act_on_point(&f, &x));
        assert!(approx_eq_vec3(&gf_x, &g_fx, 1e-10));
    }

    // ─── Exp / log ──────────────────────────────────────────────────────

    #[test]
    fn exp_log_roundtrip() {
        let xi = [0.4, -0.3, 0.6, 1.5, -0.7, 2.1, 0.5, 1.0, -1.5];
        let g = ExtendedPose::exp(&xi);
        let xi_back = g.log();
        for i in 0..9 {
            assert!(
                (xi[i] - xi_back[i]).abs() < 1e-10,
                "component {}: {} vs {}",
                i,
                xi[i],
                xi_back[i]
            );
        }
    }

    #[test]
    fn exp_log_quat_roundtrip() {
        // Same roundtrip but via the SU(2) log path.
        let xi = [0.4, -0.3, 0.6, 1.5, -0.7, 2.1, 0.5, 1.0, -1.5];
        let g = ExtendedPose::exp(&xi);
        let xi_back = g.log_quat();
        for i in 0..9 {
            assert!(
                (xi[i] - xi_back[i]).abs() < 1e-10,
                "component {}: {} vs {}",
                i,
                xi[i],
                xi_back[i]
            );
        }
    }

    #[test]
    fn log_exp_roundtrip_via_finite_element() {
        let a = ExtendedPose::exp(&[0.2, 0.3, -0.1, 0.5, 1.0, -0.5, 0.4, 0.6, 0.8]);
        let b = ExtendedPose::exp(&[-0.3, 0.1, 0.4, 1.0, -0.5, 0.5, -0.2, 0.3, -0.4]);
        let g = a.compose(&b);
        let g_back = ExtendedPose::exp(&g.log());
        assert!(approx_eq_mat3(&g.rot, &g_back.rot, 1e-10));
        assert!(approx_eq_vec3(&g.vel, &g_back.vel, 1e-10));
        assert!(approx_eq_vec3(&g.pos, &g_back.pos, 1e-10));
    }

    // ─── Structural reuse properties ────────────────────────────────────

    #[test]
    fn pure_translation_velocity_position() {
        // ω = 0  ⇒  V = I, so velocity and position pass through unchanged.
        let xi = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let g = ExtendedPose::exp(&xi);
        assert!(approx_eq_mat3(&g.rot, &I3, 1e-15));
        assert!(approx_eq_vec3(&g.vel, &[1.0, 2.0, 3.0], 1e-15));
        assert!(approx_eq_vec3(&g.pos, &[4.0, 5.0, 6.0], 1e-15));
    }

    #[test]
    fn pure_rotation() {
        // ν = ρ = 0  ⇒  vel = pos = 0, rot matches so3_exp_g directly.
        let omega = [0.3, -0.5, 0.7];
        let xi = [omega[0], omega[1], omega[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let g = ExtendedPose::exp(&xi);
        assert!(approx_eq_vec3(&g.vel, &[0.0; 3], 1e-10));
        assert!(approx_eq_vec3(&g.pos, &[0.0; 3], 1e-10));
        let r_so3 = so3_exp_g::<f64>(&omega);
        assert!(approx_eq_mat3(&g.rot, &r_so3, 1e-15));
    }

    #[test]
    fn exp_uses_same_v_for_velocity_and_position() {
        // Key structural property: exp([ω; ν; ρ]) = (R, V·ν, V·ρ) with
        // the *same* V applied to both ν and ρ.
        let omega = [0.5, -0.3, 0.7];
        let nu = [1.0, 2.0, 3.0];
        let rho = [4.0, 5.0, 6.0];
        let xi = [
            omega[0], omega[1], omega[2], nu[0], nu[1], nu[2], rho[0], rho[1], rho[2],
        ];
        let g = ExtendedPose::exp(&xi);
        let v_mat = v_matrix_g::<f64>(&omega);
        assert!(approx_eq_vec3(&g.vel, &mv(&v_mat, &nu), 1e-15));
        assert!(approx_eq_vec3(&g.pos, &mv(&v_mat, &rho), 1e-15));
    }

    #[test]
    fn velocity_position_decoupled_in_compose() {
        // The position component depends only on R and the right operand's
        // position, not on v.  No v ↔ p coupling in compose.
        let r1 = so3_exp_g::<f64>(&[0.2, 0.0, 0.0]);
        let r2 = so3_exp_g::<f64>(&[0.0, 0.3, 0.0]);
        let g_no_pos = ExtendedPoseG {
            rot: r1,
            vel: [1.0, 0.0, 0.0],
            pos: [0.0, 0.0, 0.0],
        };
        let g_with_pos = ExtendedPoseG {
            rot: r1,
            vel: [1.0, 0.0, 0.0],
            pos: [5.0, 0.0, 0.0],
        };
        let h = ExtendedPoseG {
            rot: r2,
            vel: [0.0, 1.0, 0.0],
            pos: [0.0, 2.0, 0.0],
        };
        let lhs = g_no_pos.compose(&h);
        let rhs = g_with_pos.compose(&h);
        assert!(approx_eq_vec3(&lhs.vel, &rhs.vel, 1e-15));
        let diff = sub_vec(&rhs.pos, &lhs.pos);
        assert!(approx_eq_vec3(&diff, &[5.0, 0.0, 0.0], 1e-15));
    }

    // ─── Right update + relative ────────────────────────────────────────

    #[test]
    fn right_update_small_perturbation() {
        let xi = ExtendedPose::exp(&[0.5, -0.3, 0.7, 2.0, 1.0, -1.0, 0.5, -0.5, 1.0]);
        let delta = [1e-8; 9];
        let xi_new = xi.compose(&ExtendedPose::exp(&delta));
        let diff = relative(&xi, &xi_new).log();
        let diff_norm: f64 = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(diff_norm < 1e-6);
    }

    #[test]
    fn relative_pose() {
        let g0 = ExtendedPose::exp(&[0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 0.5, 1.0, 1.5]);
        let g = ExtendedPose::exp(&[0.15, 0.25, 0.35, 1.1, 2.1, 3.1, 0.55, 1.05, 1.55]);
        let delta = relative(&g0, &g);
        let g_recovered = g0.compose(&delta);
        assert!(approx_eq_mat3(&g.rot, &g_recovered.rot, 1e-10));
        assert!(approx_eq_vec3(&g.vel, &g_recovered.vel, 1e-10));
        assert!(approx_eq_vec3(&g.pos, &g_recovered.pos, 1e-10));
    }

    #[test]
    fn omega_matches_so3_log() {
        let omega_in = [0.4, -0.6, 0.3];
        let g = ExtendedPoseG::<f64> {
            rot: so3_exp_g::<f64>(&omega_in),
            vel: [1.0; 3],
            pos: [2.0; 3],
        };
        let omega_out = so3_log_g::<f64>(&g.rot);
        assert!(approx_eq_vec3(&omega_in, &omega_out, 1e-10));
    }

    // ─── Channel-specific point actions (act_position / act_velocity) ──

    /// New inherent `act_position` agrees with the test-private
    /// `act_on_point` helper bit-for-bit (same field order, same ops).
    #[test]
    fn act_position_matches_act_on_point_helper() {
        let g = ExtendedPose::exp(&[0.30, -0.20, 0.40, 1.0, 2.0, 3.0, 0.50, -0.30, 0.70]);
        let test_points: [Vec3; 3] = [[1.0, 2.0, 3.0], [-0.5, 1.5, -2.0], [0.0, 0.0, 1.0]];
        for x in &test_points {
            let y_helper = act_on_point(&g, x);
            let y_method = g.act_position(x);
            assert_eq!(y_helper, y_method);
        }
    }

    /// Sanity: when v ≠ p, the two channel actions must produce different
    /// outputs.  Guards against a copy-paste bug where act_velocity
    /// accidentally references self.pos.
    #[test]
    fn act_position_and_act_velocity_use_distinct_channels() {
        let g = ExtendedPoseG::<f64> {
            rot: I3,
            vel: [10.0, 20.0, 30.0],
            pos: [-1.0, -2.0, -3.0],
        };
        let x = [0.5, 0.5, 0.5];
        let y_pos = g.act_position(&x);
        let y_vel = g.act_velocity(&x);
        // Difference must equal pos − vel exactly (since R = I).
        for i in 0..3 {
            assert!((y_pos[i] - y_vel[i] - (g.pos[i] - g.vel[i])).abs() < 1e-15);
        }
        // And they must not be equal.
        assert!(!approx_eq_vec3(&y_pos, &y_vel, 1e-10));
    }

    #[test]
    fn act_position_roundtrips_through_inverse() {
        let g = ExtendedPose::exp(&[0.30, -0.20, 0.40, 1.0, 2.0, 3.0, 0.50, -0.30, 0.70]);
        let x: Vec3 = [0.8, -1.2, 2.5];
        let y = g.act_position(&x);
        let x_back = g.act_position_inverse(&y);
        for i in 0..3 {
            assert!((x[i] - x_back[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn act_velocity_roundtrips_through_inverse() {
        let g = ExtendedPose::exp(&[0.30, -0.20, 0.40, 1.0, 2.0, 3.0, 0.50, -0.30, 0.70]);
        let x: Vec3 = [0.8, -1.2, 2.5];
        let y = g.act_velocity(&x);
        let x_back = g.act_velocity_inverse(&y);
        for i in 0..3 {
            assert!((x[i] - x_back[i]).abs() < 1e-14);
        }
    }

    /// AD finiteness guard at the chart origin under depth-2 nested AD —
    /// mirrors `act::tests::act_finite_under_d2_at_origin`.
    #[test]
    fn act_channels_finite_under_d2_at_origin() {
        use crate::autodiff::nested_ad::Dual;

        type D2<const N: usize> = Dual<Dual<f64, N>, N>;

        let delta: [D2<9>; 9] = std::array::from_fn(|i| {
            let inner = Dual::<f64, 9>::seed(0.0, i);
            D2::<9>::seed(inner, i)
        });
        let g = ExtendedPoseG::<D2<9>>::exp(&delta);
        let x: [D2<9>; 3] = [
            D2::<9>::constant(1.0),
            D2::<9>::constant(2.0),
            D2::<9>::constant(3.0),
        ];

        let outputs = [
            g.act_position(&x),
            g.act_position_inverse(&x),
            g.act_velocity(&x),
            g.act_velocity_inverse(&x),
        ];

        for (k, y) in outputs.iter().enumerate() {
            for i in 0..3 {
                assert!(y[i].value.value.is_finite(), "method {k} primal NaN at {i}");
                for j in 0..9 {
                    assert!(
                        y[i].value.tangent[j].is_finite(),
                        "method {k} 1st-tangent NaN at ({i},{j})"
                    );
                    assert!(
                        y[i].tangent[j].value.is_finite(),
                        "method {k} 2nd-tangent value NaN at ({i},{j})"
                    );
                    for kk in 0..9 {
                        assert!(
                            y[i].tangent[j].tangent[kk].is_finite(),
                            "method {k} Hessian NaN at ({i},{j},{kk})"
                        );
                    }
                }
            }
        }
    }
}
