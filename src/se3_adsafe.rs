//! # SE(3) — AD-safe, generic over the scalar type.
//!
//! Templated `<T: AD>` SE(3) primitives — `PoseG`, exp/log/compose/inverse,
//! adjoint, the coupling matrix `Q̃_r`, and the SE(3) right Jacobian
//! and its inverse.  Built on top of [`crate::so3_adsafe`], which supplies
//! the s-parameterized scalar basis and fused atoms (β̄, D̃·ω_m).
//!
//! All members are generic over `T: AD`, so the same implementation
//! evaluates with `T = f64`, `T = adfn<6>`, `T = D2<6>`, `T = D3<6>`, …
//! up to whatever depth is needed.  No removable singularities are
//! exposed here: the only one in this layer (the axial `(ωᵀt) β/s²`
//! term of `Q̃_r`) is replaced by `(ωᵀt) · β̄(s)` from
//! [`crate::so3_adsafe::scalar_beta_over_s_g`].

use crate::autodiff::ad_trait::AD;
use crate::so3_adsafe::{
    Mat3G, Vec3G, add_mat3_g, dot3_g, hat_g, jr_g, jr_inv_g, mm3_g, mv3_g, scalar_beta_over_s_g,
    scalar_d_s, scale_mat3_g, so3_exp_g, so3_log_g, theta_sq_from_omega, transpose3_g, v_inv_g,
    v_matrix_g, z3_g,
};

// =========================================================================
// AD-generic 6-vector / 6-matrix types and ops
// =========================================================================

/// 6-vector over AD scalar T.
pub type Vec6G<T> = [T; 6];
/// 6×6 matrix over AD scalar T, row-major.
pub type Mat6G<T> = [[T; 6]; 6];

/// Build a 6×6 matrix from four 3×3 blocks: [[A B] [C D]].
pub fn blocks_6x6_g<T: AD>(a: &Mat3G<T>, b: &Mat3G<T>, c: &Mat3G<T>, d: &Mat3G<T>) -> Mat6G<T> {
    let z = T::constant(0.0);
    let mut m = [[z; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[i][j];
            m[i][j + 3] = b[i][j];
            m[i + 3][j] = c[i][j];
            m[i + 3][j + 3] = d[i][j];
        }
    }
    m
}

pub fn mv6_g<T: AD>(m: &Mat6G<T>, v: &Vec6G<T>) -> Vec6G<T> {
    let z = T::constant(0.0);
    let mut r = [z; 6];
    for i in 0..6 {
        let mut s = z;
        for j in 0..6 {
            s += m[i][j] * v[j];
        }
        r[i] = s;
    }
    r
}

pub fn mm6_g<T: AD>(a: &Mat6G<T>, b: &Mat6G<T>) -> Mat6G<T> {
    let z = T::constant(0.0);
    let mut c = [[z; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut s = z;
            for k in 0..6 {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

// =========================================================================
// SE(3) pose
// =========================================================================

/// An SE(3) pose over AD scalar T: (R, p) where R ∈ SO(3), p ∈ ℝ³.
#[derive(Clone, Copy)]
pub struct PoseG<T: AD> {
    pub rot: Mat3G<T>,
    pub trans: Vec3G<T>,
}

impl<T: AD> PoseG<T> {
    pub fn compose(&self, other: &PoseG<T>) -> PoseG<T> {
        let rot = mm3_g(&self.rot, &other.rot);
        let rp = mv3_g(&self.rot, &other.trans);
        let trans = [
            rp[0] + self.trans[0],
            rp[1] + self.trans[1],
            rp[2] + self.trans[2],
        ];
        PoseG { rot, trans }
    }

    /// SE(3) exponential map: ξ = [ω; t] → (R(ω), V(ω)·t)
    pub fn exp(xi: &Vec6G<T>) -> PoseG<T> {
        let omega: Vec3G<T> = [xi[0], xi[1], xi[2]];
        let t: Vec3G<T> = [xi[3], xi[4], xi[5]];
        let rot = so3_exp_g(&omega);
        let v = v_matrix_g(&omega);
        let trans = mv3_g(&v, &t);
        PoseG { rot, trans }
    }

    /// SE(3) logarithmic map: (R, p) → [ω; t] = [log(R); V⁻¹(ω)·p]
    pub fn log(&self) -> Vec6G<T> {
        let omega = so3_log_g(&self.rot);
        let vi = v_inv_g(&omega);
        let t = mv3_g(&vi, &self.trans);
        [omega[0], omega[1], omega[2], t[0], t[1], t[2]]
    }

    /// SE(3) inverse: (R, p)⁻¹ = (Rᵀ, −Rᵀ·p)
    pub fn inverse(&self) -> PoseG<T> {
        let rot = transpose3_g(&self.rot);
        let neg_rt_p = mv3_g(&rot, &self.trans);
        let z = T::constant(0.0);
        let trans = [z - neg_rt_p[0], z - neg_rt_p[1], z - neg_rt_p[2]];
        PoseG { rot, trans }
    }
}

/// Convert an f64 Pose to a `PoseG<T>` (constant, no tangent).
pub fn pose_to_g<T: AD>(pose: &crate::se3_unsafe::Pose) -> PoseG<T> {
    let mut rot: Mat3G<T> = [[T::constant(0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rot[i][j] = T::constant(pose.rot[i][j]);
        }
    }
    let trans: Vec3G<T> = [
        T::constant(pose.trans[0]),
        T::constant(pose.trans[1]),
        T::constant(pose.trans[2]),
    ];
    PoseG { rot, trans }
}

// =========================================================================
// Adjoint and SE(3) right Jacobian / its inverse
// =========================================================================

/// AD-generic adjoint representation Ad(R, t) ∈ ℝ⁶ˣ⁶.
///
/// Ad(R, t) = \[\[R, 0\], \[\[t\]× R, R\]\]
///
/// Purely algebraic — no trig, no singularities.
pub fn adjoint_g<T: AD>(rot: &Mat3G<T>, trans: &Vec3G<T>) -> Mat6G<T> {
    let hat_t = hat_g(trans);
    let hat_t_r = mm3_g(&hat_t, rot);
    let z3 = z3_g::<T>();
    blocks_6x6_g(rot, &z3, &hat_t_r, rot)
}

/// Q̃_r(ω, t): inverse-side coupling block for the SE(3) right Jacobian
/// (lower-left of `(J_r^{SE(3)})⁻¹`, paper Eq. 5).  The forward Q_r is
/// recovered as `−Jr · Q̃_r · Jr`.
///
/// Implements
///   Q̃_r = ½\[t\]× + D(WT + TW) + (ωᵀt) · β̄(s) · \[ω\]×²
/// where the axial scalar uses the fused β̄(s) atom from
/// [`crate::so3_adsafe`].
pub fn q_tilde_r_g<T: AD>(omega: &Vec3G<T>, t: &Vec3G<T>) -> Mat3G<T> {
    let (s, theta) = theta_sq_from_omega(omega);
    let hat_w = hat_g(omega);
    let hat_w_sq = mm3_g(&hat_w, &hat_w);
    let hat_t = hat_g(t);

    let d = scalar_d_s(s, theta);
    let wt = mm3_g(&hat_w, &hat_t);
    let tw = mm3_g(&hat_t, &hat_w);
    let sym = add_mat3_g(&wt, &tw);

    // axial term: (ωᵀt) · β̄(s) · [ω]×², using the fused β̄(s) so
    // depth-k nested AD never sees a removable 0/0 at s = 0.
    let axial_scalar = dot3_g(omega, t) * scalar_beta_over_s_g(s, theta);

    add_mat3_g(
        &add_mat3_g(
            &scale_mat3_g(T::constant(0.5), &hat_t),
            &scale_mat3_g(d, &sym),
        ),
        &scale_mat3_g(axial_scalar, &hat_w_sq),
    )
}

/// (Jr^{SE(3)})⁻¹(ξ) where ξ = [ω; t].  Paper Eq. (4) inverse side.
pub fn se3_jr_inv_g<T: AD>(xi: &Vec6G<T>) -> Mat6G<T> {
    let omega: Vec3G<T> = [xi[0], xi[1], xi[2]];
    let t: Vec3G<T> = [xi[3], xi[4], xi[5]];
    let jri = jr_inv_g(&omega);
    let q_tilde = q_tilde_r_g(&omega, &t);
    blocks_6x6_g(&jri, &z3_g(), &q_tilde, &jri)
}

/// Jr^{SE(3)}(ξ) = \[\[Jr, 0\], \[−Jr·Q̃_r·Jr, Jr\]\].
pub fn se3_jr_g<T: AD>(xi: &Vec6G<T>) -> Mat6G<T> {
    let omega: Vec3G<T> = [xi[0], xi[1], xi[2]];
    let t: Vec3G<T> = [xi[3], xi[4], xi[5]];
    let jr_val = jr_g(&omega);
    let q_tilde = q_tilde_r_g(&omega, &t);
    let ll = scale_mat3_g(
        T::constant(-1.0),
        &mm3_g(&jr_val, &mm3_g(&q_tilde, &jr_val)),
    );
    blocks_6x6_g(&jr_val, &z3_g(), &ll, &jr_val)
}
