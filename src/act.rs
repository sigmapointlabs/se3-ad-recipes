//! Uniform SE(3) point-action API across the crate's three pose representations.
//!
//! Implements [`Act<T>`] for [`crate::se3_adsafe::PoseG<T>`],
//! [`crate::se3_quat_adsafe::PoseQ<T>`], and [`crate::se3_unsafe::Pose`] so
//! call sites read uniformly regardless of storage:
//!
//! ```ignore
//! let y = pose.act(&x);            // R·x + t
//! let x = pose.act_inverse(&y);    // Rᵀ·(y − t)
//! ```

use crate::autodiff::ad_trait::AD;
use crate::so3_adsafe::Vec3G;

/// SE(3) point action: `y = R·x + t` and its inverse `R^T · (x − t)`.
pub trait Act<T: AD> {
    /// Forward point action `y = R·x + t`.
    fn act(&self, x: &Vec3G<T>) -> Vec3G<T>;

    /// Inverse point action `x = R^T · (y − t)`.
    fn act_inverse(&self, x: &Vec3G<T>) -> Vec3G<T>;
}

// =========================================================================
// PoseG (rotation matrix + translation)
// =========================================================================

use crate::se3_adsafe::PoseG;
use crate::so3_adsafe::{mv3_g, transpose3_g};

impl<T: AD> Act<T> for PoseG<T> {
    #[inline]
    fn act(&self, x: &Vec3G<T>) -> Vec3G<T> {
        let rx = mv3_g(&self.rot, x);
        [
            rx[0] + self.trans[0],
            rx[1] + self.trans[1],
            rx[2] + self.trans[2],
        ]
    }

    #[inline]
    fn act_inverse(&self, x: &Vec3G<T>) -> Vec3G<T> {
        let dx = [
            x[0] - self.trans[0],
            x[1] - self.trans[1],
            x[2] - self.trans[2],
        ];
        mv3_g(&transpose3_g(&self.rot), &dx)
    }
}

// =========================================================================
// PoseQ (quaternion + translation)
// =========================================================================

use crate::se3_quat_adsafe::{PoseQ, quat_rotate_vec};
use crate::so3_adsafe::cross3_g;

impl<T: AD> Act<T> for PoseQ<T> {
    #[inline]
    fn act(&self, x: &Vec3G<T>) -> Vec3G<T> {
        let rx = quat_rotate_vec(self.q0, &self.qv, x);
        [
            rx[0] + self.trans[0],
            rx[1] + self.trans[1],
            rx[2] + self.trans[2],
        ]
    }

    /// Inverse rotation by conjugate q* = (q0, −qv).  Inlined to avoid
    /// allocating a negated qv vector; the second cross product is
    /// unchanged because the two sign flips on `qv` cancel.
    #[inline]
    fn act_inverse(&self, x: &Vec3G<T>) -> Vec3G<T> {
        let dx = [
            x[0] - self.trans[0],
            x[1] - self.trans[1],
            x[2] - self.trans[2],
        ];
        let two = T::constant(2.0);
        let qxv = cross3_g(&self.qv, &dx);
        let qxqxv = cross3_g(&self.qv, &qxv);
        let s = two * self.q0;
        [
            dx[0] - s * qxv[0] + two * qxqxv[0],
            dx[1] - s * qxv[1] + two * qxqxv[1],
            dx[2] - s * qxv[2] + two * qxqxv[2],
        ]
    }
}

// =========================================================================
// se3_unsafe::Pose (f64-only)
// =========================================================================

use crate::se3_unsafe::Pose;
use crate::{Vec3, mv, transpose};

impl Act<f64> for Pose {
    /// Forward to the inherent method so the two cannot drift apart.
    #[inline]
    fn act(&self, x: &Vec3) -> Vec3 {
        Pose::act(self, x)
    }

    #[inline]
    fn act_inverse(&self, x: &Vec3) -> Vec3 {
        let dx = [
            x[0] - self.trans[0],
            x[1] - self.trans[1],
            x[2] - self.trans[2],
        ];
        mv(&transpose(&self.rot), &dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec6;
    use crate::se3_adsafe::PoseG;
    use crate::se3_quat_adsafe::PoseQ;

    /// Pick a non-trivial pose, build it both ways, check `act` agrees.
    #[test]
    fn poseg_and_poseq_act_agree() {
        let xi: Vec6 = [0.30, -0.20, 0.40, 0.50, -0.30, 0.70];
        let pg = PoseG::<f64>::exp(&xi);
        let pq = PoseQ::<f64>::from_pose_g(&pg);

        let test_points: [[f64; 3]; 3] = [[1.0, 2.0, 3.0], [-0.5, 1.5, -2.0], [0.0, 0.0, 1.0]];

        for x in &test_points {
            let y_g = pg.act(x);
            let y_q = pq.act(x);
            for i in 0..3 {
                assert!(
                    (y_g[i] - y_q[i]).abs() < 1e-12,
                    "act mismatch at component {i}: PoseG={} PoseQ={}",
                    y_g[i],
                    y_q[i]
                );
            }
        }
    }

    #[test]
    fn pose_inherent_act_matches_trait_act() {
        use crate::Vec3;
        use crate::se3_unsafe::Pose;

        let xi: Vec6 = [0.30, -0.20, 0.40, 0.50, -0.30, 0.70];
        let pose = Pose::exp(&xi);
        let test_points: [Vec3; 3] = [[1.0, 2.0, 3.0], [-0.5, 1.5, -2.0], [0.0, 0.0, 1.0]];

        for x in &test_points {
            let y_inherent = Pose::act(&pose, x);
            let y_trait = <Pose as Act<f64>>::act(&pose, x);
            assert_eq!(
                y_inherent, y_trait,
                "inherent vs trait act diverged at {x:?}"
            );
        }
    }

    #[test]
    fn poseg_act_roundtrips_through_inverse() {
        let xi: Vec6 = [0.30, -0.20, 0.40, 0.50, -0.30, 0.70];
        let pg = PoseG::<f64>::exp(&xi);
        let x: [f64; 3] = [0.8, -1.2, 2.5];
        let y = pg.act(&x);
        let x_back = pg.act_inverse(&y);
        for i in 0..3 {
            assert!(
                (x[i] - x_back[i]).abs() < 1e-14,
                "PoseG round-trip diverged at {i}: {} vs {}",
                x[i],
                x_back[i]
            );
        }
    }

    #[test]
    fn poseq_act_roundtrips_through_inverse() {
        let xi: Vec6 = [0.30, -0.20, 0.40, 0.50, -0.30, 0.70];
        let pq = PoseQ::<f64>::from_pose_g(&PoseG::<f64>::exp(&xi));
        let x: [f64; 3] = [0.8, -1.2, 2.5];
        let y = pq.act(&x);
        let x_back = pq.act_inverse(&y);
        for i in 0..3 {
            assert!(
                (x[i] - x_back[i]).abs() < 1e-14,
                "PoseQ round-trip diverged at {i}: {} vs {}",
                x[i],
                x_back[i]
            );
        }
    }

    #[test]
    fn pose_unsafe_act_roundtrips_through_inverse() {
        use crate::Vec3;
        use crate::se3_unsafe::Pose;

        let xi: Vec6 = [0.30, -0.20, 0.40, 0.50, -0.30, 0.70];
        let pose = Pose::exp(&xi);
        let x: Vec3 = [0.8, -1.2, 2.5];
        let y = <Pose as Act<f64>>::act(&pose, &x);
        let x_back = <Pose as Act<f64>>::act_inverse(&pose, &y);
        for i in 0..3 {
            assert!(
                (x[i] - x_back[i]).abs() < 1e-14,
                "Pose round-trip diverged at {i}: {} vs {}",
                x[i],
                x_back[i]
            );
        }
    }

    #[test]
    fn act_finite_under_d2_at_origin() {
        use crate::autodiff::nested_ad::Dual;

        type D2<const N: usize> = Dual<Dual<f64, N>, N>;

        // Seed δ at the origin with D2<6> directions on both AD levels —
        // mirrors the seeding used by nll_bench::hessian_d2.
        let delta: [D2<6>; 6] = std::array::from_fn(|i| {
            let inner = Dual::<f64, 6>::seed(0.0, i);
            D2::<6>::seed(inner, i)
        });

        let pg = PoseG::<D2<6>>::exp(&delta);
        let pq = PoseQ::<D2<6>>::exp(&delta);

        let x: [D2<6>; 3] = [
            D2::<6>::constant(1.0),
            D2::<6>::constant(2.0),
            D2::<6>::constant(3.0),
        ];

        let y_g = pg.act(&x);
        let y_q = pq.act(&x);

        // Walk every primal + tangent + tangent-of-tangent scalar in the
        // output and assert it is finite. A NaN here would mean an unsafe
        // branch leaked into one of the impls.
        for i in 0..3 {
            assert!(y_g[i].value.value.is_finite(), "PoseG primal NaN at {i}");
            assert!(y_q[i].value.value.is_finite(), "PoseQ primal NaN at {i}");
            for j in 0..6 {
                assert!(
                    y_g[i].value.tangent[j].is_finite(),
                    "PoseG 1st-tangent NaN at ({i},{j})"
                );
                assert!(
                    y_q[i].value.tangent[j].is_finite(),
                    "PoseQ 1st-tangent NaN at ({i},{j})"
                );
                assert!(
                    y_g[i].tangent[j].value.is_finite(),
                    "PoseG 2nd-tangent value NaN at ({i},{j})"
                );
                assert!(
                    y_q[i].tangent[j].value.is_finite(),
                    "PoseQ 2nd-tangent value NaN at ({i},{j})"
                );
                for k in 0..6 {
                    assert!(
                        y_g[i].tangent[j].tangent[k].is_finite(),
                        "PoseG Hessian NaN at ({i},{j},{k})"
                    );
                    assert!(
                        y_q[i].tangent[j].tangent[k].is_finite(),
                        "PoseQ Hessian NaN at ({i},{j},{k})"
                    );
                }
            }
        }
    }

    #[test]
    fn poseg_act_matches_projective_transform_point_g() {
        use crate::projective::transform_point_g;

        let xi: Vec6 = [0.30, -0.20, 0.40, 0.50, -0.30, 0.70];
        let pg = PoseG::<f64>::exp(&xi);

        let test_points: [[f64; 3]; 3] = [[1.0, 2.0, 3.0], [-0.5, 1.5, -2.0], [0.0, 0.0, 1.0]];

        for x in &test_points {
            let y_trait = pg.act(x);
            let y_free = transform_point_g::<f64>(&pg.rot, &pg.trans, x);
            // Both compute mv3_g(&R, &x) + t in the same field order, so
            // they must agree bit-for-bit.
            assert_eq!(
                y_trait, y_free,
                "PoseG::act diverged from transform_point_g at {x:?}"
            );
        }
    }
}
