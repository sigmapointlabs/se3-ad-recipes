//! # SE(3) — quaternion-storage AD-safe pose.
//!
//! Templated `<T: AD>` SE(3) primitives over a **quaternion + translation**
//! storage scheme: 7 scalars (`q0`, `qv₃`, `t₃`) instead of the 12 stored by
//! [`crate::se3_adsafe::PoseG`] (a 3×3 R + 3-vector t).
//!
//! ## Why a second representation
//!
//! [`PoseG`] is the canonical recipe form — every §V/§VI derivative tensor
//! in [`crate::jacobians_ad`] is written against `R` and `t`, so anything
//! that consumes the recipe stays on `PoseG`.  `PoseQ` is the storage form
//! you'd reach for in a pose-graph optimiser, an integrator that composes
//! many poses, or interop with libraries (manif, GTSAM, Sophus) that
//! standardise on quaternions.
//!
//! Storage is the only durable win.  Composition cost is comparable to
//! `PoseG` — the translation transform `R(q)·v` via the algebraic identity
//! `v + 2 q0 (qv × v) + 2 qv × (qv × v)` is ~24 flops vs. the 27 of
//! `mm3_g`'s matrix-vector — and `to_pose_g` materialises R anyway when
//! needed.
//!
//! ## AD safety
//!
//! Two scalars in this module need careful handling under nested AD:
//! `cos(θ/2)` and `sin(θ/2)/θ` on the **exp** side, and the log factor
//! `2·atan2(√s_q, q0)/√s_q` on the **log** side.  All three sit on a
//! removable singularity at the chart origin (θ = 0 ⇔ qv = 0 ⇔ s_q = 0)
//! that AD cannot navigate symbolically; each is therefore split into
//! a Taylor branch below `TAYLOR_THRESHOLD_S = 1e-4` and a closed form
//! above.
//!
//! The two cases have **different AD-safety arguments** that are worth
//! keeping straight:
//!
//! * **Below threshold (Taylor branch).**  Each scalar is a polynomial
//!   in `s = θ²` (or `s_q = qv·qv`), so AD safety is unconditional up to
//!   the polynomial degree.  The Step-1 atoms carry degree 4 in `s`, which
//!   is depth 4 in `s` = depth 8 in `θ` of headroom — safely past the
//!   recipe's depth-3 working margin.  This is the only branch traversed
//!   at the seed point δ = 0 of an NLL Hessian / cubic evaluation, which
//!   is what `so3_log_quaternion_d3_log_exp_is_identity_at_origin` checks
//!   to depth 3.
//!
//! * **Above threshold (closed form).**  Each scalar is built from
//!   primitives (`sqrt`, `atan2`, `cos`, `sin`, `/`) that are **C^∞ on
//!   the open domain** `s > 0`, so AD safety follows from smoothness —
//!   no specific depth bound is needed.  Finite-precision conditioning
//!   does degrade with AD depth: the dominant offender is `sqrt`, whose
//!   k-th derivative scales as `s^{-(2k-1)/2}`.  At the threshold s = 1e-4
//!   and depth 3 the relative size of the third derivative of the full
//!   log factor `2·atan2(√s_q, q0)/√s_q` is ~1e5 (measured), and after
//!   chain-ruling through `s_q(δ)` an end-to-end Hessian / cubic
//!   evaluation typically loses 4–8 of f64's 16 digits — adequate for
//!   nll_bench-class problems, tight enough to know about.  Production
//!   callers reach this branch when they evaluate **away from** the seed
//!   origin (e.g. linearising the prior at a finite ξ̄ ≠ 0); accuracy
//!   there is governed by this conditioning estimate, not by any test.
//!
//! What the existing depth-3 test does **not** establish is that the
//! closed-form branch is depth-3 safe at arbitrary above-threshold
//! points — that test seeds at the origin where only the Taylor branch
//! fires.  The closed-form branch's safety is by-construction (smoothness),
//! not by-test.
//!
//! ## Discipline
//!
//! Like `PoseG`, this type does **not** renormalise on every operation.
//! Group identities (q0² + qv·qv = 1, R Rᵀ = I) drift the same way under
//! finite-precision composition; users who care project back periodically
//! (or call `from_pose_g(self.to_pose_g())`, which re-extracts via
//! Shepperd).  This matches the `PoseG` convention so the two forms have
//! the same numerical character.

use crate::autodiff::ad_trait::AD;
use crate::se3_adsafe::PoseG;
use crate::so3_adsafe::{
    Mat3G, Vec3G, cross3_g, dot3_g, mat3_to_quat_shepperd_g, mv3_g,
    scalar_cos_half_s, scalar_half_sinc_half_s, theta_sq_from_omega,
    v_inv_g, v_matrix_g, TAYLOR_THRESHOLD_S,
};

// `cross3_g`, `mat3_to_quat_shepperd_g`, and `TAYLOR_THRESHOLD_S` need to be
// `pub` (or `pub(crate)`) in `so3_adsafe` for this module to use them.
// `mat3_to_quat_shepperd_g` is currently a private helper of
// `so3_log_g_quaternion`; promote it to `pub(crate)`.  `TAYLOR_THRESHOLD_S`
// is currently private; promote it likewise.

// =========================================================================
// PoseQ — quaternion-storage SE(3) pose
// =========================================================================

/// SE(3) pose stored as scalar quaternion (q0, qv) plus translation.
///
/// Hamilton convention: `q0² + qv·qv = 1` (enforced at construction by
/// `exp`; preserved by `compose`/`inverse` as algebraic identities, not
/// renormalised per call).
#[derive(Clone, Copy)]
pub struct PoseQ<T: AD> {
    pub q0: T,
    pub qv: Vec3G<T>,
    pub trans: Vec3G<T>,
}

pub type Vec6G<T> = [T; 6];

impl<T: AD> PoseQ<T> {
    /// Identity pose: q = (1, 0⃗), t = 0⃗.
    pub fn identity() -> Self {
        Self {
            q0: T::constant(1.0),
            qv: [T::constant(0.0); 3],
            trans: [T::constant(0.0); 3],
        }
    }

    /// SE(3) exponential: ξ = [ω, t] → (q(ω), V(ω)·t).
    ///
    /// Quaternion built directly from ω via the Step-1 atoms — no matrix
    /// R is materialised on this side.  Translation reuses `V(ω)` from
    /// [`crate::so3_adsafe`] verbatim.
    pub fn exp(xi: &Vec6G<T>) -> Self {
        let omega: Vec3G<T> = [xi[0], xi[1], xi[2]];
        let t: Vec3G<T> = [xi[3], xi[4], xi[5]];
        let (s, theta) = theta_sq_from_omega(&omega);
        let q0 = scalar_cos_half_s(s, theta);
        let half_sinc = scalar_half_sinc_half_s(s, theta);
        let qv = [omega[0] * half_sinc, omega[1] * half_sinc, omega[2] * half_sinc];
        let v = v_matrix_g(&omega);
        let trans = mv3_g(&v, &t);
        Self { q0, qv, trans }
    }

    /// SE(3) logarithm: (q, p) → ξ = [ω, t].
    ///
    /// 1. Canonicalise q0 ≥ 0.
    /// 2. ω = factor · qv with factor = 2·asin(√s_q)/√s_q (Taylor below
    ///    threshold; 2·atan2(√s_q, q0)/√s_q above).  This is the same
    ///    factor scalar proved AD-safe by
    ///    `so3_log_quaternion_d3_log_exp_is_identity_at_origin`.
    /// 3. t = V⁻¹(ω) · trans.
    pub fn log(&self) -> Vec6G<T> {
        let (q0, qv) = if self.q0.to_constant() < 0.0 {
            let neg = T::constant(-1.0);
            (neg * self.q0, [neg * self.qv[0], neg * self.qv[1], neg * self.qv[2]])
        } else {
            (self.q0, self.qv)
        };
        let s_q = dot3_g(&qv, &qv);

        // 2·asin(x)/x = 2·(1 + x²/6 + 3x⁴/40 + 15x⁶/336 + 105x⁸/3456 + …)
        // In s_q = x²:  2 + s_q/3 + 3 s_q²/20 + 5 s_q³/56 + 35 s_q⁴/576 + …
        let factor = if s_q.to_constant() < TAYLOR_THRESHOLD_S {
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
        let omega: Vec3G<T> = [factor * qv[0], factor * qv[1], factor * qv[2]];

        let vi = v_inv_g(&omega);
        let t = mv3_g(&vi, &self.trans);
        [omega[0], omega[1], omega[2], t[0], t[1], t[2]]
    }

    /// Compose: self · other.
    ///
    /// Quaternion product (Hamilton):  `q = (a₀b₀ − aᵥ·bᵥ, a₀bᵥ + b₀aᵥ + aᵥ × bᵥ)`.
    /// Translation:  `t = R(self.q) · other.trans + self.trans`, with R(q)·v
    /// computed via the algebraic identity that avoids materialising R.
    pub fn compose(&self, other: &PoseQ<T>) -> PoseQ<T> {
        let q0 = self.q0 * other.q0 - dot3_g(&self.qv, &other.qv);
        let cab = cross3_g(&self.qv, &other.qv);
        let qv = [
            self.q0 * other.qv[0] + other.q0 * self.qv[0] + cab[0],
            self.q0 * other.qv[1] + other.q0 * self.qv[1] + cab[1],
            self.q0 * other.qv[2] + other.q0 * self.qv[2] + cab[2],
        ];
        let rotated = quat_rotate_vec(self.q0, &self.qv, &other.trans);
        let trans = [
            rotated[0] + self.trans[0],
            rotated[1] + self.trans[1],
            rotated[2] + self.trans[2],
        ];
        PoseQ { q0, qv, trans }
    }

    /// Inverse: (q, t)⁻¹ = (q⁻¹, −R(q⁻¹) · t)  with  q⁻¹ = (q0, −qv).
    pub fn inverse(&self) -> PoseQ<T> {
        let neg = T::constant(-1.0);
        let q0_inv = self.q0;
        let qv_inv = [neg * self.qv[0], neg * self.qv[1], neg * self.qv[2]];
        let neg_t = [neg * self.trans[0], neg * self.trans[1], neg * self.trans[2]];
        let trans = quat_rotate_vec(q0_inv, &qv_inv, &neg_t);
        PoseQ { q0: q0_inv, qv: qv_inv, trans }
    }

    /// Bridge to [`PoseG`].  Builds R from the quaternion via the standard
    /// quadratic identity.
    pub fn to_pose_g(&self) -> PoseG<T> {
        PoseG { rot: quat_to_rotmat(self.q0, &self.qv), trans: self.trans }
    }

    /// Bridge from [`PoseG`].  Extracts q via Shepperd, canonicalises q0 ≥ 0
    /// so successive log/compose chains agree with `PoseQ::log`'s canonical
    /// branch.
    pub fn from_pose_g(p: &PoseG<T>) -> Self {
        let (q0, qv) = mat3_to_quat_shepperd_g(&p.rot);
        let (q0, qv) = if q0.to_constant() < 0.0 {
            let neg = T::constant(-1.0);
            (neg * q0, [neg * qv[0], neg * qv[1], neg * qv[2]])
        } else {
            (q0, qv)
        };
        Self { q0, qv, trans: p.trans }
    }
}

// =========================================================================
// Free helpers: quaternion-vector ops shared by compose/inverse/to_pose_g
// =========================================================================

/// `R(q) · v` without materialising R, via the identity
/// `v + 2 q0 (qv × v) + 2 qv × (qv × v)`.  ~24 flops.
#[inline]
pub fn quat_rotate_vec<T: AD>(q0: T, qv: &Vec3G<T>, v: &Vec3G<T>) -> Vec3G<T> {
    let two = T::constant(2.0);
    let qxv = cross3_g(qv, v);
    let qxqxv = cross3_g(qv, &qxv);
    [
        v[0] + two * q0 * qxv[0] + two * qxqxv[0],
        v[1] + two * q0 * qxv[1] + two * qxqxv[1],
        v[2] + two * q0 * qxv[2] + two * qxqxv[2],
    ]
}

/// Standard quaternion → rotation matrix.  Only used by `to_pose_g`.
#[inline]
pub fn quat_to_rotmat<T: AD>(q0: T, qv: &Vec3G<T>) -> Mat3G<T> {
    let one = T::constant(1.0);
    let two = T::constant(2.0);
    let xx = qv[0] * qv[0];
    let yy = qv[1] * qv[1];
    let zz = qv[2] * qv[2];
    let xy = qv[0] * qv[1];
    let xz = qv[0] * qv[2];
    let yz = qv[1] * qv[2];
    let wx = q0 * qv[0];
    let wy = q0 * qv[1];
    let wz = q0 * qv[2];
    [
        [one - two * (yy + zz), two * (xy - wz), two * (xz + wy)],
        [two * (xy + wz), one - two * (xx + zz), two * (yz - wx)],
        [two * (xz - wy), two * (yz + wx), one - two * (xx + yy)],
    ]
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::nested_ad::Dual;
    use crate::se3_adsafe::PoseG;
    use crate::so3_adsafe::Vec3G;

    type D2<const N: usize> = Dual<Dual<f64, N>, N>;

    fn approx_eq_mat3<T: AD>(a: &Mat3G<T>, b: &Mat3G<T>, tol: f64) -> bool {
        for i in 0..3 {
            for j in 0..3 {
                if (a[i][j].to_constant() - b[i][j].to_constant()).abs() > tol {
                    return false;
                }
            }
        }
        true
    }
    fn approx_eq_vec3<T: AD>(a: &Vec3G<T>, b: &Vec3G<T>, tol: f64) -> bool {
        (0..3).all(|i| (a[i].to_constant() - b[i].to_constant()).abs() < tol)
    }

    #[test]
    fn identity_compose_is_no_op() {
        let p = PoseQ::<f64>::exp(&[0.3, -0.2, 0.4, 1.0, 2.0, -0.5]);
        let id = PoseQ::<f64>::identity();
        let l = p.compose(&id);
        let r = id.compose(&p);
        assert!((p.q0 - l.q0).abs() < 1e-15);
        assert!(approx_eq_vec3(&p.qv, &l.qv, 1e-15));
        assert!(approx_eq_vec3(&p.trans, &l.trans, 1e-15));
        assert!((p.q0 - r.q0).abs() < 1e-15);
        assert!(approx_eq_vec3(&p.qv, &r.qv, 1e-15));
        assert!(approx_eq_vec3(&p.trans, &r.trans, 1e-15));
    }

    #[test]
    fn compose_inverse_is_identity() {
        let p = PoseQ::<f64>::exp(&[0.3, -0.5, 0.2, 1.0, -2.0, 0.5]);
        let id = p.compose(&p.inverse());
        assert!((id.q0.abs() - 1.0).abs() < 1e-12, "id.q0 = {}", id.q0);
        assert!(approx_eq_vec3(&id.qv, &[0.0, 0.0, 0.0], 1e-12));
        assert!(approx_eq_vec3(&id.trans, &[0.0, 0.0, 0.0], 1e-12));
    }

    #[test]
    fn unit_quaternion_after_exp() {
        for &xi in &[
            [0.0_f64; 6],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [1e-9, 0.0, 0.0, 1.0, 0.0, 0.0],
            [3.0, 0.1, -0.05, -0.5, 1.2, 0.3],
        ] {
            let p = PoseQ::<f64>::exp(&xi);
            let n2 = p.q0 * p.q0 + p.qv[0] * p.qv[0] + p.qv[1] * p.qv[1] + p.qv[2] * p.qv[2];
            assert!((n2 - 1.0).abs() < 1e-12, "xi={xi:?}: |q|² = {n2}");
        }
    }

    #[test]
    fn exp_log_roundtrip() {
        for &xi in &[
            [0.4_f64, -0.3, 0.6, 1.5, -0.7, 2.1],
            [1e-9, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.5, -0.7, 2.1],
            [2.5, 0.1, -0.05, 0.0, 0.0, 0.0],
        ] {
            let p = PoseQ::<f64>::exp(&xi);
            let xi_back = p.log();
            for i in 0..6 {
                assert!((xi[i] - xi_back[i]).abs() < 1e-10,
                    "xi={xi:?} component {i}: {} vs {}", xi[i], xi_back[i]);
            }
        }
    }

    #[test]
    fn poseq_exp_matches_poseg_exp_f64() {
        for &xi in &[
            [0.0_f64; 6],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [1e-9, 0.0, 0.0, 1.0, 0.0, 0.0],
            [3.0, 0.1, -0.05, -0.5, 1.2, 0.3],
        ] {
            let pq = PoseQ::<f64>::exp(&xi);
            let pg = PoseG::<f64>::exp(&xi);
            let pq_as_g = pq.to_pose_g();
            assert!(approx_eq_mat3(&pq_as_g.rot, &pg.rot, 1e-12),
                "xi={xi:?}: rot mismatch");
            assert!(approx_eq_vec3(&pq_as_g.trans, &pg.trans, 1e-12),
                "xi={xi:?}: trans mismatch");
        }
    }

    #[test]
    fn poseq_compose_matches_poseg_compose_f64() {
        let xi_a = [0.3_f64, -0.2, 0.5, 1.0, -1.5, 0.5];
        let xi_b = [-0.1, 0.4, 0.2, 0.3, 0.7, -0.4];
        let pa_q = PoseQ::<f64>::exp(&xi_a);
        let pb_q = PoseQ::<f64>::exp(&xi_b);
        let pa_g = PoseG::<f64>::exp(&xi_a);
        let pb_g = PoseG::<f64>::exp(&xi_b);
        let c_q_as_g = pa_q.compose(&pb_q).to_pose_g();
        let c_g = pa_g.compose(&pb_g);
        assert!(approx_eq_mat3(&c_q_as_g.rot, &c_g.rot, 1e-12), "rot mismatch");
        assert!(approx_eq_vec3(&c_q_as_g.trans, &c_g.trans, 1e-12), "trans mismatch");
    }

    #[test]
    fn poseq_inverse_matches_poseg_inverse_f64() {
        let xi = [0.3_f64, -0.2, 0.5, 1.0, -1.5, 0.5];
        let pq = PoseQ::<f64>::exp(&xi);
        let pg = PoseG::<f64>::exp(&xi);
        let inv_q_as_g = pq.inverse().to_pose_g();
        let inv_g = pg.inverse();
        assert!(approx_eq_mat3(&inv_q_as_g.rot, &inv_g.rot, 1e-12));
        assert!(approx_eq_vec3(&inv_q_as_g.trans, &inv_g.trans, 1e-12));
    }

    #[test]
    fn from_pose_g_to_pose_g_roundtrip() {
        for &xi in &[
            [0.3_f64, -0.2, 0.5, 1.0, -1.5, 0.5],
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            [2.0, 0.1, -0.05, 0.5, 0.5, 0.5],
        ] {
            let pg = PoseG::<f64>::exp(&xi);
            let pq = PoseQ::<f64>::from_pose_g(&pg);
            let pg_back = pq.to_pose_g();
            assert!(approx_eq_mat3(&pg.rot, &pg_back.rot, 1e-12),
                "xi={xi:?}: rot bridge roundtrip");
            assert!(approx_eq_vec3(&pg.trans, &pg_back.trans, 1e-12),
                "xi={xi:?}: trans bridge roundtrip");
        }
    }

    /// AD depth-2 cross-check: PoseQ exp+compose Hessians match PoseG.
    /// Equality at value, first-tangent, AND second-tangent levels = the
    /// quaternion route is AD-safe through depth 2 for the operations
    /// `nll_bench` would invoke if wired with PoseQ.
    #[test]
    fn poseq_compose_d2_matches_poseg_d2() {
        let base_xi: [f64; 6] = [0.3, -0.2, 0.5, 1.0, -1.5, 0.5];
        let seed_xi: [f64; 6] = [0.05, 0.0, -0.02, 0.1, 0.0, 0.05];

        let delta_q: Vec6G<D2<6>> = std::array::from_fn(|i| {
            let inner = Dual::<f64, 6>::seed(seed_xi[i], i);
            D2::<6>::seed(inner, i)
        });
        let base_q: Vec6G<D2<6>> = std::array::from_fn(|i| D2::<6>::constant(base_xi[i]));

        let pa_q = PoseQ::<D2<6>>::exp(&base_q);
        let pb_q = PoseQ::<D2<6>>::exp(&delta_q);
        let composed_q_as_g = pa_q.compose(&pb_q).to_pose_g();

        let pa_g = PoseG::<D2<6>>::exp(&base_q);
        let pb_g = PoseG::<D2<6>>::exp(&delta_q);
        let composed_g = pa_g.compose(&pb_g);

        let cmp = |a: D2<6>, b: D2<6>, label: &str| {
            assert!((a.value.value - b.value.value).abs() < 1e-12,
                "{label} value: q={} g={}", a.value.value, b.value.value);
            for i in 0..6 {
                let dq = a.value.tangent[i];
                let dg = b.value.tangent[i];
                assert!((dq - dg).abs() < 1e-10, "{label} ∂[{i}]: q={dq} g={dg}");
            }
            for i in 0..6 { for j in 0..6 {
                let hq = a.tangent[i].tangent[j];
                let hg = b.tangent[i].tangent[j];
                assert!((hq - hg).abs() < 1e-9, "{label} ∂²[{i},{j}]: q={hq} g={hg}");
            }}
        };

        for i in 0..3 { for j in 0..3 {
            cmp(composed_q_as_g.rot[i][j], composed_g.rot[i][j],
                &format!("rot[{i}][{j}]"));
        }}
        for i in 0..3 {
            cmp(composed_q_as_g.trans[i], composed_g.trans[i],
                &format!("trans[{i}]"));
        }
    }

    /// AD depth-2 lift of so3_log_quaternion_d3_log_exp_is_identity_at_origin
    /// up to SE(3): Log(Exp(δ)) ≡ δ at δ = 0 means value 0, ∇ = I, ∇² = 0
    /// component-wise.
    #[test]
    fn poseq_log_exp_is_identity_at_origin_d2() {
        let delta: Vec6G<D2<6>> = std::array::from_fn(|i| {
            let inner = Dual::<f64, 6>::seed(0.0, i);
            D2::<6>::seed(inner, i)
        });
        let p = PoseQ::<D2<6>>::exp(&delta);
        let xi_back = p.log();

        for i in 0..6 {
            assert!(xi_back[i].value.value.abs() < 1e-13,
                "value[{i}] = {}", xi_back[i].value.value);
            for j in 0..6 {
                let v = xi_back[i].value.tangent[j];
                let exp = if i == j { 1.0 } else { 0.0 };
                assert!((v - exp).abs() < 1e-11,
                    "∂xi_back[{i}]/∂δ[{j}] = {v}, expected {exp}");
            }
            for j in 0..6 { for k in 0..6 {
                let h = xi_back[i].tangent[j].tangent[k];
                assert!(h.abs() < 1e-9,
                    "∂²xi_back[{i}]/∂δ[{j}]∂δ[{k}] = {h}");
            }}
        }
    }
}