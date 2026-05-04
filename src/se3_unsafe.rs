//! # SE(3) — f64-only, AD-unsafe.
//!
//! Conventional θ-form Pose for plain `f64` evaluation.  Uses the
//! second-order Taylor branches of [`crate::so3_unsafe`] and is **not** safe
//! to differentiate through with any AD scalar type — the small-angle branches
//! and singular factor pairs follow the §IV.A/§IV.B failure modes of the
//! companion paper.  For AD use, see [`crate::se3_adsafe`].
//!
//! A pose T ∈ SE(3) is represented as (R, p) where R ∈ SO(3), p ∈ ℝ³.
//! Group composition (Eq. se3comp):
//!   T₁·T₂ = (R₁R₂, R₁p₂ + p₁)
//!
//! Exponential coordinates ξ = \[ω; t\] ∈ ℝ⁶ (rotation-first):
//!   Exp(ξ) = (R(ω), V(ω)·t)
//!
//! where V(ω) = I + B·\[ω\]× + C·\[ω\]×² is the left Jacobian (Eq. V).

use crate::so3_unsafe;
use crate::*;

/// A rigid body pose f = (R, T) ∈ SE(3).
#[derive(Debug, Clone, Copy)]
pub struct Pose {
    /// Rotation matrix R ∈ SO(3).
    pub rot: Mat3,
    /// Translation vector T ∈ ℝ³.
    pub trans: Vec3,
}

impl Pose {
    /// Construct from rotation matrix and translation.
    pub fn new(rot: Mat3, trans: Vec3) -> Self {
        Pose { rot, trans }
    }

    /// Identity pose: (I, 0).
    pub fn identity() -> Self {
        Pose {
            rot: I3,
            trans: [0.0; 3],
        }
    }

    /// Exponential map: ξ = \[ω; t\] ∈ ℝ⁶ → SE(3).
    ///
    /// Exp(ξ) = (R(ω), V(ω)·t)
    ///
    /// where R is the Rodrigues rotation and V is the coupling matrix (= Jl).
    /// Paper Eq. (se3exp).
    pub fn exp(xi: &Vec6) -> Self {
        let omega = [xi[0], xi[1], xi[2]];
        let t = [xi[3], xi[4], xi[5]];
        let rot = so3_unsafe::exp(&omega);
        let v = so3_unsafe::v_matrix(&omega);
        let trans = mv(&v, &t);
        Pose { rot, trans }
    }

    /// Logarithmic map: SE(3) → ξ = \[ω; t\] ∈ ℝ⁶.
    ///
    /// ξ = (log(R), V⁻¹(ω)·p)
    pub fn log(&self) -> Vec6 {
        let omega = so3_unsafe::log(&self.rot);
        let vi = so3_unsafe::v_inv(&omega);
        let t = mv(&vi, &self.trans);
        [omega[0], omega[1], omega[2], t[0], t[1], t[2]]
    }

    /// Group composition: self · other.
    ///
    /// (Rₐ, Tₐ)·(Rᵦ, Tᵦ) = (Rₐ Rᵦ, Rₐ Tᵦ + Tₐ)
    pub fn compose(&self, other: &Pose) -> Pose {
        let rot = mm(&self.rot, &other.rot);
        let trans = add_vec(&mv(&self.rot, &other.trans), &self.trans);
        Pose { rot, trans }
    }

    /// Group inverse: f⁻¹ = (Rᵀ, -Rᵀ T).
    pub fn inverse(&self) -> Pose {
        let rt = transpose(&self.rot);
        let trans = scale_vec(-1.0, &mv(&rt, &self.trans));
        Pose { rot: rt, trans }
    }

    /// Action on a 3D point: f ⋆ x = R·x + T.
    pub fn act(&self, x: &Vec3) -> Vec3 {
        add_vec(&mv(&self.rot, x), &self.trans)
    }

    /// Relative pose: self⁻¹ · other.
    ///
    /// Useful for computing the deviation Δf = f₀⁻¹·f.
    pub fn relative(&self, other: &Pose) -> Pose {
        self.inverse().compose(other)
    }

    /// Rodrigues vector of the rotational part.
    pub fn omega(&self) -> Vec3 {
        so3_unsafe::log(&self.rot)
    }

    /// Extract the 4×4 homogeneous matrix representation.
    pub fn to_matrix(&self) -> [[f64; 4]; 4] {
        let r = &self.rot;
        let t = &self.trans;
        [
            [r[0][0], r[0][1], r[0][2], t[0]],
            [r[1][0], r[1][1], r[1][2], t[1]],
            [r[2][0], r[2][1], r[2][2], t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

/// Compose a finite pose with a small right perturbation in exponential coords:
///   f_new = f · exp(δξ)
///
/// This is the compositive update used in Gauss-Newton on SE(3).
pub fn right_update(f: &Pose, delta_xi: &Vec6) -> Pose {
    let df = Pose::exp(delta_xi);
    f.compose(&df)
}

/// Compose a small left perturbation with a finite pose:
///   f_new = exp(δξ) · f
pub fn left_update(delta_xi: &Vec6, f: &Pose) -> Pose {
    let df = Pose::exp(delta_xi);
    df.compose(f)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_identity_action() {
        let f = Pose::identity();
        let x = [1.0, 2.0, 3.0];
        assert!(approx_eq_vec3(&f.act(&x), &x, 1e-15));
    }

    #[test]
    fn test_compose_inverse_is_identity() {
        let f = Pose::exp(&[0.3, -0.5, 0.2, 1.0, -2.0, 0.5]);
        let id = f.compose(&f.inverse());
        assert!(approx_eq_mat3(&id.rot, &I3, 1e-10));
        assert!(approx_eq_vec3(&id.trans, &[0.0, 0.0, 0.0], 1e-10));
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let xi = [0.4, -0.3, 0.6, 1.5, -0.7, 2.1];
        let f = Pose::exp(&xi);
        let xi_back = f.log();
        for i in 0..6 {
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
    fn test_action_composition() {
        // (g·f) ⋆ x == g ⋆ (f ⋆ x)
        let f = Pose::exp(&[0.2, 0.3, -0.1, 1.0, 0.0, 0.0]);
        let g = Pose::exp(&[-0.1, 0.4, 0.2, 0.0, 1.0, -0.5]);
        let x = [1.0, -1.0, 3.0];
        let gf_x = g.compose(&f).act(&x);
        let g_fx = g.act(&f.act(&x));
        assert!(approx_eq_vec3(&gf_x, &g_fx, 1e-10));
    }

    #[test]
    fn test_right_update_small_perturbation() {
        let f = Pose::exp(&[0.5, -0.3, 0.7, 2.0, 1.0, -1.0]);
        let delta = [1e-8, -1e-8, 1e-8, 1e-8, 1e-8, -1e-8];
        let f_new = right_update(&f, &delta);
        // Should be very close to f
        let diff = f.relative(&f_new).log();
        let diff_norm: f64 = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(diff_norm < 1e-6);
    }

    #[test]
    fn test_relative_pose() {
        let f0 = Pose::exp(&[0.1, 0.2, 0.3, 1.0, 2.0, 3.0]);
        let f = Pose::exp(&[0.15, 0.25, 0.35, 1.1, 2.1, 3.1]);
        let delta = f0.relative(&f);
        // f0 · delta should recover f
        let f_recovered = f0.compose(&delta);
        assert!(approx_eq_mat3(&f.rot, &f_recovered.rot, 1e-10));
        assert!(approx_eq_vec3(&f.trans, &f_recovered.trans, 1e-10));
    }

    #[test]
    fn test_pure_translation() {
        let xi = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        let f = Pose::exp(&xi);
        assert!(approx_eq_mat3(&f.rot, &I3, 1e-15));
        assert!(approx_eq_vec3(&f.trans, &[1.0, 2.0, 3.0], 1e-15));
    }

    #[test]
    fn test_pure_rotation() {
        let xi = [0.3, -0.5, 0.7, 0.0, 0.0, 0.0];
        let f = Pose::exp(&xi);
        assert!(approx_eq_vec3(&f.trans, &[0.0, 0.0, 0.0], 1e-10));
    }
}
