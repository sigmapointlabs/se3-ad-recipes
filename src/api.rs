//! Curated application API for SE(3) derivative recipes.
//!
//! Raw modules remain available for paper reproducibility, but new application
//! code should use this module. Names here favor explicit conventions over the
//! compact notation used in the paper.
//!
//! # Stability tiers
//!
//! The crate exposes three nested tiers of API stability. Pick the highest
//! tier that meets your needs.
//!
//! 1. **Application tier — [`pose`], [`extended_pose`], [`quaternion_pose`],
//!    [`rotation`], [`types`].** Opaque group types (`Pose`, `ExtendedPose`,
//!    `QuatPose`) with descriptive method names. `f64` only. **SemVer-stable.**
//!    Breaking changes require a major release, and the intent is to keep
//!    those exceptionally rare and well-motivated.
//! 2. **Expert tier — [`expert`].** AD-generic operations organized by
//!    group (`expert::so3`, `expert::se3`, `expert::se23`,
//!    `expert::quat_se3`, `expert::projective`, `expert::linalg`,
//!    `expert::types`, `expert::ad`). Generic over `T: AD`. **Standard
//!    semantic versioning applies:** additions ship in minor releases;
//!    renames and removals require a major release. Semantics are
//!    paper-aligned (AD-safe fused scalar basis from §V.D, smooth at the
//!    SO(3) origin); the specific small-angle / antipodal switch
//!    thresholds and Taylor depths are implementation details and may be
//!    tightened in minor releases when the new threshold remains
//!    numerically AD-safe at every supported derivative depth. The
//!    naming convention is clean and dimension-agnostic — no `_g`
//!    suffix, no dimension suffixes — and is intended to outlive the
//!    paper's evolving notation.
//! 3. **Raw modules.** `so3_adsafe`, `se3_adsafe`, `se23_adsafe`,
//!    `se3_quat_adsafe`, `jacobians_*`, `projective`, `autodiff`. Hidden
//!    from rustdoc, `pub` for source compatibility. **No stability
//!    promise** — these track paper notation and may rename / move /
//!    disappear between minor releases. Use only as an escape hatch for
//!    operations not yet exposed at the expert tier.
//!
//! # AD-generic code
//!
//! The application tier is `f64`-only by design. AD-generic code — forward,
//! reverse, or nested-dual — must use the [`expert`] tier (or, for
//! uncovered helpers, the raw modules). This is the supported pattern: the
//! application tier owns the *types* and *API contract*; the expert tier
//! owns the *AD-generic implementations*.
//!
//! # Quickstart
//!
//! Build a pose from a twist, act on a point, and round-trip through `log`:
//!
//! ```
//! use se3_ad_recipes::prelude::*;
//!
//! let twist: Twist = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0];
//! let pose = Pose::exp(&twist);
//!
//! let point: Point3 = [0.5, 0.0, 0.0];
//! let moved = pose.act(&point);
//!
//! let recovered = pose.log();
//! for (a, b) in recovered.iter().zip(twist.iter()) {
//!     assert!((a - b).abs() < 1e-12);
//! }
//!
//! // Right-perturbation in the SE(3) tangent space.
//! let delta: Twist = [0.0, 0.0, 0.01, 0.0, 0.0, 0.0];
//! let perturbed = se3_right_update(&pose, &delta);
//! assert_ne!(perturbed, pose);
//! # let _ = moved;
//! ```

fn negate<const N: usize>(value: &[f64; N]) -> [f64; N] {
    std::array::from_fn(|i| -value[i])
}

fn all_finite<const N: usize>(value: &[f64; N]) -> bool {
    value.iter().all(|component| component.is_finite())
}

/// Input-shape validation failure shared by checked constructors across the
/// application tier.
///
/// Wrapped by [`pose::PoseError::Input`] and [`quaternion_pose::QuatPoseError::Input`]
/// so domain-specific validators (orthogonality, unit-norm, ...) and generic
/// input checks (finiteness, tolerance shape) compose without each enum
/// re-stating the same variants.
///
/// Marked `#[non_exhaustive]`: future input-shape checks (e.g. NaN-in-quaternion
/// special cases) may add variants in a minor release without bumping major.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum InputError {
    /// A checked constructor received a non-finite matrix or vector component.
    NonFiniteInput {
        /// Name of the argument containing `NaN` or infinity.
        field: &'static str,
    },
    /// The supplied tolerance was negative or non-finite.
    InvalidTolerance {
        /// Invalid tolerance value.
        tolerance: f64,
    },
}

impl core::fmt::Display for InputError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteInput { field } => write!(f, "{field} contains a non-finite value"),
            Self::InvalidTolerance { tolerance } => write!(
                f,
                "tolerance must be finite and non-negative, got {tolerance}"
            ),
        }
    }
}

impl std::error::Error for InputError {}

fn check_tolerance(tolerance: f64) -> Result<(), InputError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        Err(InputError::InvalidTolerance { tolerance })
    } else {
        Ok(())
    }
}

fn check_finite_vec<const N: usize>(v: &[f64; N], field: &'static str) -> Result<(), InputError> {
    if all_finite(v) {
        Ok(())
    } else {
        Err(InputError::NonFiniteInput { field })
    }
}

fn check_finite_mat3(m: &types::Mat3, field: &'static str) -> Result<(), InputError> {
    if m.iter().all(all_finite) {
        Ok(())
    } else {
        Err(InputError::NonFiniteInput { field })
    }
}

/// Common fixed-size types and semantic aliases.
pub mod types {
    /// 3×3 matrix, row-major.
    pub type Mat3 = [[f64; 3]; 3];

    /// 6×6 matrix, row-major.
    pub type Mat6 = [[f64; 6]; 6];

    /// 9×9 matrix, row-major.
    pub type Mat9 = [[f64; 9]; 9];

    /// Three-dimensional vector.
    pub type Vec3 = [f64; 3];

    /// Six-dimensional vector.
    pub type Vec6 = [f64; 6];

    /// Nine-dimensional vector.
    pub type Vec9 = [f64; 9];

    /// Rotation-first SE(3) tangent vector `[omega_x, omega_y, omega_z, t_x, t_y, t_z]`.
    pub type Twist = Vec6;

    /// A point or translation vector in three dimensions.
    pub type Point3 = Vec3;

    /// 3×6 matrix, e.g. the Jacobian of a 3D point with respect to a 6D twist.
    pub type Mat3x6 = [[f64; 6]; 3];

    /// 3×9 matrix, e.g. the Jacobian of a 3D point with respect to a 9D
    /// SE_2(3) tangent.
    pub type Mat3x9 = [[f64; 9]; 3];
}

/// SO(3) operations with descriptive Jacobian names.
pub mod rotation {
    use super::negate;
    use crate::so3_adsafe::{hat_g, jr_g, jr_inv_g, so3_exp_g, so3_log_g};
    use crate::{Mat3, Vec3};

    /// Exponential map from a rotation vector to a rotation matrix.
    #[must_use]
    pub fn exp(omega: &Vec3) -> Mat3 {
        so3_exp_g(omega)
    }

    /// Principal logarithm from a rotation matrix to a rotation vector.
    #[must_use]
    pub fn log(rotation: &Mat3) -> Vec3 {
        so3_log_g(rotation)
    }

    /// Hat map from a rotation vector to its skew-symmetric matrix.
    #[must_use]
    pub fn hat(omega: &Vec3) -> Mat3 {
        hat_g(omega)
    }

    /// Vee map from a skew-symmetric matrix to a rotation vector.
    #[must_use]
    pub fn vee(matrix: &Mat3) -> Vec3 {
        [matrix[2][1], matrix[0][2], matrix[1][0]]
    }

    /// SO(3) right Jacobian.
    #[must_use]
    pub fn right_jacobian(omega: &Vec3) -> Mat3 {
        jr_g(omega)
    }

    /// Inverse SO(3) right Jacobian.
    #[must_use]
    pub fn right_jacobian_inverse(omega: &Vec3) -> Mat3 {
        jr_inv_g(omega)
    }

    /// SO(3) left Jacobian, equal to `J_r(-omega)`.
    #[must_use]
    pub fn left_jacobian(omega: &Vec3) -> Mat3 {
        jr_g(&negate(omega))
    }

    /// Inverse SO(3) left Jacobian, equal to `J_r^-1(-omega)`.
    #[must_use]
    pub fn left_jacobian_inverse(omega: &Vec3) -> Mat3 {
        jr_inv_g(&negate(omega))
    }

    /// Apply a right perturbation: `R * Exp(delta)`.
    #[must_use]
    pub fn right_update(rotation: &Mat3, delta: &Vec3) -> Mat3 {
        crate::mm(rotation, &so3_exp_g(delta))
    }

    /// Apply a left perturbation: `Exp(delta) * R`.
    #[must_use]
    pub fn left_update(delta: &Vec3, rotation: &Mat3) -> Mat3 {
        crate::mm(&so3_exp_g(delta), rotation)
    }
}

/// SE(3) poses and operations using the rotation-first `[omega; t]` convention.
pub mod pose {
    use super::{negate, types::Mat3x6};
    use crate::act::Act;
    use crate::se3_adsafe::{PoseG, adjoint_g, se3_jr_g, se3_jr_inv_g};
    use crate::so3_adsafe::hat_g;
    use crate::{I3, Mat3, Mat6, Vec3, Vec6, det3, frob_diff, mm, scale_mat, transpose};

    /// Default tolerance used by [`Pose::from_rotation_translation`] when
    /// validating that the supplied 3×3 matrix is a proper rotation. Chosen
    /// to accept matrices that round-trip through standard `f64` group
    /// operations without renormalization, while rejecting obvious garbage.
    pub const DEFAULT_ROTATION_TOLERANCE: f64 = 1e-6;

    /// Reason an external `(R, t)` pair could not be accepted as a [`Pose`].
    ///
    /// Marked `#[non_exhaustive]`: future rotation-validation checks may add
    /// variants in a minor release without bumping major.
    #[derive(Clone, Copy, Debug, PartialEq)]
    #[non_exhaustive]
    pub enum PoseError {
        /// Generic input-shape failure: non-finite component or invalid tolerance.
        Input(super::InputError),
        /// `Rᵀ R` differed from the identity by more than the requested
        /// tolerance (Frobenius norm).
        NotOrthogonal {
            /// `‖Rᵀ R − I‖_F` for the supplied matrix.
            orthogonality_residual: f64,
            /// The tolerance the residual was compared against.
            tolerance: f64,
        },
        /// `det(R)` was not within tolerance of `+1`; the matrix may be a
        /// reflection (`det ≈ −1`) or singular.
        NotProperRotation {
            /// `det(R)` for the supplied matrix.
            determinant: f64,
            /// The tolerance `|det − 1|` was compared against.
            tolerance: f64,
        },
    }

    impl From<super::InputError> for PoseError {
        fn from(e: super::InputError) -> Self {
            Self::Input(e)
        }
    }

    impl core::fmt::Display for PoseError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::Input(e) => write!(f, "pose input: {e}"),
                Self::NotOrthogonal {
                    orthogonality_residual,
                    tolerance,
                } => write!(
                    f,
                    "rotation matrix not orthogonal: ‖RᵀR − I‖_F = {orthogonality_residual:e} > {tolerance:e}"
                ),
                Self::NotProperRotation {
                    determinant,
                    tolerance,
                } => write!(
                    f,
                    "rotation matrix not a proper rotation: det = {determinant} (|det − 1| > {tolerance:e})"
                ),
            }
        }
    }

    impl std::error::Error for PoseError {}

    pub(crate) fn validate_rotation(rot: &Mat3, tolerance: f64) -> Result<(), PoseError> {
        super::check_tolerance(tolerance)?;
        super::check_finite_mat3(rot, "rotation")?;

        let residual = frob_diff(&mm(&transpose(rot), rot), &I3);
        if residual > tolerance {
            return Err(PoseError::NotOrthogonal {
                orthogonality_residual: residual,
                tolerance,
            });
        }
        let determinant = det3(rot);
        if (determinant - 1.0).abs() > tolerance {
            return Err(PoseError::NotProperRotation {
                determinant,
                tolerance,
            });
        }
        Ok(())
    }

    /// Opaque rigid-body rotation + translation in SE(3).
    ///
    /// The internal representation is exactly the pair `(R ∈ SO(3), t ∈ ℝ³)`
    /// stored as a `[[f64; 3]; 3]` and a `[f64; 3]`. Construct from external data via
    /// [`Pose::from_rotation_translation`] (validated) or
    /// [`Pose::from_parts_unchecked`] (caller asserts the invariant).
    ///
    /// `PartialEq` compares the underlying `(R, t)` arrays bitwise; for
    /// numerically-tolerant equality use the Frobenius / L2 differences on
    /// `pose.rotation()` and `pose.translation()`.
    ///
    /// ```
    /// use se3_ad_recipes::api::pose::Pose;
    ///
    /// let pose = Pose::exp(&[0.0, 0.0, 0.5, 1.0, 2.0, -0.5]);
    /// let inverse = pose.inverse();
    /// let identity = pose.compose(&inverse);
    ///
    /// // Composition with the inverse returns identity (to numerical tolerance).
    /// let log = identity.log();
    /// for component in log {
    ///     assert!(component.abs() < 1e-12);
    /// }
    /// ```
    #[derive(Clone, Copy)]
    #[must_use = "Pose is a pure value; ignoring it drops the computed transform"]
    pub struct Pose {
        inner: PoseG<f64>,
    }

    impl Pose {
        /// Identity pose.
        pub const fn identity() -> Self {
            Self {
                inner: PoseG {
                    rot: I3,
                    trans: [0.0; 3],
                },
            }
        }

        /// Construct a pose from a rotation matrix and translation, checking
        /// that `R` is a proper rotation within
        /// [`DEFAULT_ROTATION_TOLERANCE`].
        ///
        /// Returns [`PoseError::NotOrthogonal`] if `‖Rᵀ R − I‖_F > tol`, and
        /// [`PoseError::NotProperRotation`] if `|det R − 1| > tol` (the
        /// latter catches reflections, where `det R ≈ −1`). Non-finite input
        /// and invalid tolerances are rejected before these checks.
        ///
        /// ```
        /// use se3_ad_recipes::api::pose::{Pose, PoseError};
        ///
        /// // Identity rotation, unit translation along x: accepted.
        /// let identity_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        /// let pose = Pose::from_rotation_translation(&identity_rot, &[1.0, 0.0, 0.0]).unwrap();
        /// assert_eq!(pose.translation(), &[1.0, 0.0, 0.0]);
        ///
        /// // Reflection (det = -1): rejected.
        /// let reflection = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
        /// let err = Pose::from_rotation_translation(&reflection, &[0.0; 3]).unwrap_err();
        /// assert!(matches!(err, PoseError::NotProperRotation { .. }));
        /// ```
        pub fn from_rotation_translation(rot: &Mat3, trans: &Vec3) -> Result<Self, PoseError> {
            Self::from_rotation_translation_with_tolerance(rot, trans, DEFAULT_ROTATION_TOLERANCE)
        }

        /// As [`Pose::from_rotation_translation`] but with a caller-supplied
        /// tolerance, for ingesting matrices from sources with known
        /// numerical conditioning.
        pub fn from_rotation_translation_with_tolerance(
            rot: &Mat3,
            trans: &Vec3,
            tolerance: f64,
        ) -> Result<Self, PoseError> {
            validate_rotation(rot, tolerance)?;
            super::check_finite_vec(trans, "translation")?;
            Ok(Self {
                inner: PoseG {
                    rot: *rot,
                    trans: *trans,
                },
            })
        }

        /// Construct a pose from raw parts without validation.
        ///
        /// The caller asserts that `rot` is a proper rotation matrix
        /// (`Rᵀ R = I`, `det R = +1`). Passing an arbitrary matrix is safe
        /// (no memory unsafety), but downstream numerical results — Jacobians,
        /// logs, compositions — will be silently wrong. Prefer
        /// [`Pose::from_rotation_translation`] unless you have an
        /// independent guarantee.
        pub fn from_parts_unchecked(rot: Mat3, trans: Vec3) -> Self {
            Self {
                inner: PoseG { rot, trans },
            }
        }

        /// Exponential map from a rotation-first twist.
        pub fn exp(twist: &Vec6) -> Self {
            Self {
                inner: PoseG::exp(twist),
            }
        }

        /// Logarithmic map to a rotation-first twist.
        #[must_use]
        pub fn log(&self) -> Vec6 {
            self.inner.log()
        }

        /// Logarithmic map via the SU(2) (quaternion) path. Numerically
        /// preferred near the antipodal locus, where the matrix log is
        /// inconvenient. Delegates through the quaternion representation;
        /// the rotation algebra is identical to [`log`](Self::log)
        /// elsewhere.
        #[must_use]
        pub fn log_quaternion(&self) -> Vec6 {
            crate::se3_quat_adsafe::PoseQ::from_pose_g(&self.inner).log()
        }

        /// Group composition `self * other`.
        pub fn compose(&self, other: &Self) -> Self {
            Self {
                inner: self.inner.compose(&other.inner),
            }
        }

        /// Group inverse.
        pub fn inverse(&self) -> Self {
            Self {
                inner: self.inner.inverse(),
            }
        }

        /// Apply the pose to a point.
        #[must_use]
        pub fn act(&self, point: &Vec3) -> Vec3 {
            self.inner.act(point)
        }

        /// Apply the inverse pose to a point.
        #[must_use]
        pub fn act_inverse(&self, point: &Vec3) -> Vec3 {
            self.inner.act_inverse(point)
        }

        /// Read-only access to the rotation matrix.
        pub fn rotation(&self) -> &Mat3 {
            &self.inner.rot
        }

        /// Read-only access to the translation vector.
        pub fn translation(&self) -> &Vec3 {
            &self.inner.trans
        }

        /// Adjoint representation: `Ad(pose) → Mat6`.
        #[must_use]
        pub fn adjoint(&self) -> Mat6 {
            adjoint_g(&self.inner.rot, &self.inner.trans)
        }
    }

    impl Default for Pose {
        /// Identity pose.
        fn default() -> Self {
            Self::identity()
        }
    }

    impl core::fmt::Debug for Pose {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("Pose")
                .field("rotation", &self.inner.rot)
                .field("translation", &self.inner.trans)
                .finish()
        }
    }

    impl PartialEq for Pose {
        fn eq(&self, other: &Self) -> bool {
            self.inner.rot == other.inner.rot && self.inner.trans == other.inner.trans
        }
    }

    #[cfg(feature = "serde")]
    #[derive(serde::Serialize, serde::Deserialize)]
    struct PoseRepr {
        rotation: Mat3,
        translation: Vec3,
    }

    #[cfg(feature = "serde")]
    impl serde::Serialize for Pose {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            PoseRepr {
                rotation: self.inner.rot,
                translation: self.inner.trans,
            }
            .serialize(serializer)
        }
    }

    #[cfg(feature = "serde")]
    impl<'de> serde::Deserialize<'de> for Pose {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let repr = PoseRepr::deserialize(deserializer)?;
            Pose::from_rotation_translation(&repr.rotation, &repr.translation)
                .map_err(serde::de::Error::custom)
        }
    }

    /// Apply a right perturbation: `pose * Exp(delta)`.
    pub fn right_update(pose: &Pose, delta: &Vec6) -> Pose {
        pose.compose(&Pose::exp(delta))
    }

    /// Apply a left perturbation: `Exp(delta) * pose`.
    pub fn left_update(delta: &Vec6, pose: &Pose) -> Pose {
        Pose::exp(delta).compose(pose)
    }

    /// SE(3) right Jacobian for a rotation-first twist.
    #[must_use]
    pub fn right_jacobian(twist: &Vec6) -> Mat6 {
        se3_jr_g(twist)
    }

    /// Inverse SE(3) right Jacobian for a rotation-first twist.
    #[must_use]
    pub fn right_jacobian_inverse(twist: &Vec6) -> Mat6 {
        se3_jr_inv_g(twist)
    }

    /// SE(3) left Jacobian, equal to `J_r(-twist)`.
    #[must_use]
    pub fn left_jacobian(twist: &Vec6) -> Mat6 {
        se3_jr_g(&negate(twist))
    }

    /// Inverse SE(3) left Jacobian, equal to `J_r^-1(-twist)`.
    #[must_use]
    pub fn left_jacobian_inverse(twist: &Vec6) -> Mat6 {
        se3_jr_inv_g(&negate(twist))
    }

    /// Jacobian of `pose * Exp(delta)` acting on `point`, evaluated at zero.
    ///
    /// Columns use the rotation-first perturbation convention and equal
    /// `[-R [point]x | R]`.
    ///
    /// Typical use: nonlinear least-squares with the residual `r = pose.act(p) - y`,
    /// where the row of the Jacobian for each landmark is this 3×6 block.
    ///
    /// ```
    /// use se3_ad_recipes::api::pose::{self, Pose};
    ///
    /// let pose = Pose::exp(&[0.2, -0.1, 0.3, 1.0, 0.2, -0.4]);
    /// let landmark = [0.7, -0.5, 2.0];
    /// let jacobian = pose::right_point_action_jacobian(&pose, &landmark);
    ///
    /// // Translation columns (3..6) are just R, regardless of the landmark.
    /// for row in 0..3 {
    ///     for column in 0..3 {
    ///         assert_eq!(jacobian[row][3 + column], pose.rotation()[row][column]);
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn right_point_action_jacobian(pose: &Pose, point: &Vec3) -> Mat3x6 {
        let rotation = pose.rotation();
        let rotational = scale_mat(-1.0, &mm(rotation, &hat_g(point)));
        let mut jacobian = [[0.0; 6]; 3];
        for row in 0..3 {
            jacobian[row][..3].copy_from_slice(&rotational[row]);
            jacobian[row][3..].copy_from_slice(&rotation[row]);
        }
        jacobian
    }
}

/// SE_2(3) extended-pose group for inertial navigation: `(R, velocity, position)`.
pub mod extended_pose {
    use crate::jacobians_se23_adsafe::{
        adjoint_inv_se23_g, adjoint_se23_g, se23_jr_g, se23_jr_inv_g,
    };
    use crate::se23_adsafe::ExtendedPoseG;
    use crate::{I3, Mat3, Mat9, Vec3, Vec9};

    use super::{
        check_finite_vec, negate,
        pose::{DEFAULT_ROTATION_TOLERANCE, PoseError, validate_rotation},
    };

    /// Opaque SE_2(3) state: rotation, velocity, position.
    ///
    /// Internal representation is `(R ∈ SO(3), v ∈ ℝ³, p ∈ ℝ³)`.
    /// The tangent ordering is `[ω; ν; ρ]`
    /// (rotation, then velocity, then position).
    ///
    /// `PartialEq` compares the underlying `(R, v, p)` arrays bitwise; for
    /// numerically-tolerant equality use the Frobenius / L2 differences on
    /// `state.rotation()`, `state.velocity()`, and `state.position()`.
    ///
    /// SE_2(3) carries two natural point-action channels because the state
    /// has two ℝ³ offsets sharing one rotation. Use [`act_position`] to
    /// transform a 3D position by `(R, p)` and [`act_velocity`] to transform
    /// a 3D velocity by `(R, v)`.
    ///
    /// ```
    /// use se3_ad_recipes::api::extended_pose::ExtendedPose;
    ///
    /// // Twist with nonzero ω, ν, ρ.
    /// let state = ExtendedPose::exp(&[0.1, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0]);
    ///
    /// let x = [1.0, 0.0, 0.0];
    /// let world_point = state.act_position(&x);
    /// let world_velocity = state.act_velocity(&x);
    ///
    /// // The two channels produce different outputs (different offsets).
    /// assert_ne!(world_point, world_velocity);
    /// ```
    ///
    /// [`act_position`]: ExtendedPose::act_position
    /// [`act_velocity`]: ExtendedPose::act_velocity
    #[derive(Clone, Copy)]
    #[must_use = "ExtendedPose is a pure value; ignoring it drops the computed state"]
    pub struct ExtendedPose {
        inner: ExtendedPoseG<f64>,
    }

    impl ExtendedPose {
        /// Identity element: `R = I`, `v = 0`, `p = 0`.
        pub const fn identity() -> Self {
            Self {
                inner: ExtendedPoseG {
                    rot: I3,
                    vel: [0.0; 3],
                    pos: [0.0; 3],
                },
            }
        }

        /// Construct from `(R, v, p)` with rotation validated against
        /// [`DEFAULT_ROTATION_TOLERANCE`]. All components must be finite.
        pub fn from_rotation_velocity_position(
            rot: &Mat3,
            vel: &Vec3,
            pos: &Vec3,
        ) -> Result<Self, PoseError> {
            Self::from_rotation_velocity_position_with_tolerance(
                rot,
                vel,
                pos,
                DEFAULT_ROTATION_TOLERANCE,
            )
        }

        /// As [`ExtendedPose::from_rotation_velocity_position`] but with a
        /// caller-supplied rotation tolerance.
        pub fn from_rotation_velocity_position_with_tolerance(
            rot: &Mat3,
            vel: &Vec3,
            pos: &Vec3,
            tolerance: f64,
        ) -> Result<Self, PoseError> {
            validate_rotation(rot, tolerance)?;
            check_finite_vec(vel, "velocity")?;
            check_finite_vec(pos, "position")?;
            Ok(Self {
                inner: ExtendedPoseG {
                    rot: *rot,
                    vel: *vel,
                    pos: *pos,
                },
            })
        }

        /// Construct from raw parts without validation. Caller asserts that
        /// `rot` is a proper rotation matrix.
        pub fn from_parts_unchecked(rot: Mat3, vel: Vec3, pos: Vec3) -> Self {
            Self {
                inner: ExtendedPoseG { rot, vel, pos },
            }
        }

        /// SE_2(3) exponential map from a rotation-velocity-position tangent.
        pub fn exp(xi: &Vec9) -> Self {
            Self {
                inner: ExtendedPoseG::exp(xi),
            }
        }

        /// SE_2(3) logarithmic map (matrix-based SO(3) log).
        #[must_use]
        pub fn log(&self) -> Vec9 {
            self.inner.log()
        }

        /// SE_2(3) logarithmic map (quaternion-based SO(3) log; preferred
        /// near the antipodal locus).
        #[must_use]
        pub fn log_quaternion(&self) -> Vec9 {
            self.inner.log_quat()
        }

        /// Group composition `self · other`.
        pub fn compose(&self, other: &Self) -> Self {
            Self {
                inner: self.inner.compose(&other.inner),
            }
        }

        /// Group inverse.
        pub fn inverse(&self) -> Self {
            Self {
                inner: self.inner.inverse(),
            }
        }

        /// Position-channel action: `y = R · x + p`. Transforms a 3D point.
        #[must_use]
        pub fn act_position(&self, point: &Vec3) -> Vec3 {
            self.inner.act_position(point)
        }

        /// Inverse position-channel action: `x = Rᵀ · (y − p)`.
        #[must_use]
        pub fn act_position_inverse(&self, point: &Vec3) -> Vec3 {
            self.inner.act_position_inverse(point)
        }

        /// Velocity-channel action: `y = R · x + v`. Transforms a 3D velocity.
        #[must_use]
        pub fn act_velocity(&self, point: &Vec3) -> Vec3 {
            self.inner.act_velocity(point)
        }

        /// Inverse velocity-channel action: `x = Rᵀ · (y − v)`.
        #[must_use]
        pub fn act_velocity_inverse(&self, point: &Vec3) -> Vec3 {
            self.inner.act_velocity_inverse(point)
        }

        /// Read-only access to the rotation matrix.
        pub fn rotation(&self) -> &Mat3 {
            &self.inner.rot
        }

        /// Read-only access to the velocity vector.
        pub fn velocity(&self) -> &Vec3 {
            &self.inner.vel
        }

        /// Read-only access to the position vector.
        pub fn position(&self) -> &Vec3 {
            &self.inner.pos
        }

        /// SE_2(3) adjoint representation: `Ad(self) → Mat9`.
        #[must_use]
        pub fn adjoint(&self) -> Mat9 {
            adjoint_se23_g(&self.inner.rot, &self.inner.vel, &self.inner.pos)
        }

        /// SE_2(3) inverse adjoint representation.
        #[must_use]
        pub fn adjoint_inverse(&self) -> Mat9 {
            adjoint_inv_se23_g(&self.inner.rot, &self.inner.vel, &self.inner.pos)
        }
    }

    impl Default for ExtendedPose {
        fn default() -> Self {
            Self::identity()
        }
    }

    impl core::fmt::Debug for ExtendedPose {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("ExtendedPose")
                .field("rotation", &self.inner.rot)
                .field("velocity", &self.inner.vel)
                .field("position", &self.inner.pos)
                .finish()
        }
    }

    impl PartialEq for ExtendedPose {
        fn eq(&self, other: &Self) -> bool {
            self.inner.rot == other.inner.rot
                && self.inner.vel == other.inner.vel
                && self.inner.pos == other.inner.pos
        }
    }

    #[cfg(feature = "serde")]
    #[derive(serde::Serialize, serde::Deserialize)]
    struct ExtendedPoseRepr {
        rotation: Mat3,
        velocity: Vec3,
        position: Vec3,
    }

    #[cfg(feature = "serde")]
    impl serde::Serialize for ExtendedPose {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            ExtendedPoseRepr {
                rotation: self.inner.rot,
                velocity: self.inner.vel,
                position: self.inner.pos,
            }
            .serialize(serializer)
        }
    }

    #[cfg(feature = "serde")]
    impl<'de> serde::Deserialize<'de> for ExtendedPose {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let repr = ExtendedPoseRepr::deserialize(deserializer)?;
            ExtendedPose::from_rotation_velocity_position(
                &repr.rotation,
                &repr.velocity,
                &repr.position,
            )
            .map_err(serde::de::Error::custom)
        }
    }

    /// Right perturbation: `pose · Exp(delta)`.
    pub fn right_update(pose: &ExtendedPose, delta: &Vec9) -> ExtendedPose {
        pose.compose(&ExtendedPose::exp(delta))
    }

    /// Left perturbation: `Exp(delta) · pose`.
    pub fn left_update(delta: &Vec9, pose: &ExtendedPose) -> ExtendedPose {
        ExtendedPose::exp(delta).compose(pose)
    }

    /// SE_2(3) right Jacobian for a rotation-velocity-position tangent.
    #[must_use]
    pub fn right_jacobian(xi: &Vec9) -> Mat9 {
        se23_jr_g(xi)
    }

    /// Inverse SE_2(3) right Jacobian.
    #[must_use]
    pub fn right_jacobian_inverse(xi: &Vec9) -> Mat9 {
        se23_jr_inv_g(xi)
    }

    /// SE_2(3) left Jacobian, equal to `J_r(-xi)`.
    #[must_use]
    pub fn left_jacobian(xi: &Vec9) -> Mat9 {
        se23_jr_g(&negate(xi))
    }

    /// Inverse SE_2(3) left Jacobian.
    #[must_use]
    pub fn left_jacobian_inverse(xi: &Vec9) -> Mat9 {
        se23_jr_inv_g(&negate(xi))
    }

    /// Jacobian of `pose · Exp(delta).act_position(point)` evaluated at delta = 0.
    ///
    /// Columns are blocked as `[∂/∂ω | ∂/∂ν | ∂/∂ρ]` per the SE_2(3) tangent
    /// ordering and equal `[-R [point]x | 0 | R]`: a right-perturbation in the
    /// velocity channel does not move the position-channel output.
    #[must_use]
    pub fn right_point_action_position_jacobian(
        pose: &ExtendedPose,
        point: &Vec3,
    ) -> super::types::Mat3x9 {
        position_channel_jacobian(pose.rotation(), point)
    }

    /// Jacobian of `pose · Exp(delta).act_velocity(point)` evaluated at delta = 0.
    ///
    /// Columns are blocked as `[∂/∂ω | ∂/∂ν | ∂/∂ρ]` and equal
    /// `[-R [point]x | R | 0]`: a right-perturbation in the position channel
    /// does not move the velocity-channel output.
    #[must_use]
    pub fn right_point_action_velocity_jacobian(
        pose: &ExtendedPose,
        point: &Vec3,
    ) -> super::types::Mat3x9 {
        velocity_channel_jacobian(pose.rotation(), point)
    }

    fn position_channel_jacobian(rotation: &Mat3, point: &Vec3) -> super::types::Mat3x9 {
        let mut jacobian = [[0.0; 9]; 3];
        write_rotation_columns(&mut jacobian, rotation, point);
        for row in 0..3 {
            jacobian[row][6..].copy_from_slice(&rotation[row]);
        }
        jacobian
    }

    fn velocity_channel_jacobian(rotation: &Mat3, point: &Vec3) -> super::types::Mat3x9 {
        let mut jacobian = [[0.0; 9]; 3];
        write_rotation_columns(&mut jacobian, rotation, point);
        for row in 0..3 {
            jacobian[row][3..6].copy_from_slice(&rotation[row]);
        }
        jacobian
    }

    fn write_rotation_columns(jacobian: &mut super::types::Mat3x9, rotation: &Mat3, point: &Vec3) {
        use crate::so3_adsafe::hat_g;
        use crate::{mm, scale_mat};
        let rotational = scale_mat(-1.0, &mm(rotation, &hat_g(point)));
        for row in 0..3 {
            jacobian[row][..3].copy_from_slice(&rotational[row]);
        }
    }
}

/// Quaternion-storage SE(3): same group as [`pose::Pose`] in different coordinates.
pub mod quaternion_pose {
    use crate::act::Act;
    use crate::se3_quat_adsafe::PoseQ;
    use crate::{Vec3, Vec6};

    use super::{check_finite_vec, check_tolerance, pose::Pose};

    /// Tolerance used by [`QuatPose::from_quaternion_translation`] when
    /// validating unit-norm of the quaternion.
    pub const DEFAULT_QUATERNION_TOLERANCE: f64 = 1e-6;

    /// Reason an external `(q₀, q_v, t)` could not be accepted as a [`QuatPose`].
    ///
    /// Marked `#[non_exhaustive]`: future quaternion-validation checks may
    /// add variants in a minor release without bumping major.
    #[derive(Clone, Copy, Debug, PartialEq)]
    #[non_exhaustive]
    pub enum QuatPoseError {
        /// Generic input-shape failure: non-finite component or invalid tolerance.
        Input(super::InputError),
        /// `q₀² + q_v · q_v` differed from 1 by more than the requested tolerance.
        NotUnitNorm {
            /// `|q₀² + q_v · q_v − 1|` for the supplied quaternion.
            norm_residual: f64,
            /// The tolerance the residual was compared against.
            tolerance: f64,
        },
    }

    impl From<super::InputError> for QuatPoseError {
        fn from(e: super::InputError) -> Self {
            Self::Input(e)
        }
    }

    impl core::fmt::Display for QuatPoseError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::Input(e) => write!(f, "quaternion pose input: {e}"),
                Self::NotUnitNorm {
                    norm_residual,
                    tolerance,
                } => write!(
                    f,
                    "quaternion not unit-norm: |q·q − 1| = {norm_residual:e} > {tolerance:e}"
                ),
            }
        }
    }

    impl std::error::Error for QuatPoseError {}

    /// Opaque rigid-body pose stored as `(q₀, q_v, t)`.
    ///
    /// Hamilton convention; `q₀² + ‖q_v‖² = 1` is preserved as an algebraic
    /// identity by [`exp`](Self::exp), [`compose`](Self::compose), and
    /// [`inverse`](Self::inverse). Useful when matrix-form rotations are
    /// numerically inconvenient (e.g. near the antipodal locus) and as a
    /// compact 7-scalar pose representation.
    ///
    /// `PartialEq` compares the underlying `(q₀, q_v, t)` storage bitwise;
    /// for numerically-tolerant equality (and to handle the `±q` sign
    /// ambiguity in SU(2)), compare against a canonical pose form or use
    /// L2 differences on `qp.q0()`, `qp.qv()`, and `qp.translation()`.
    ///
    /// Bidirectional [`From`] bridges to [`Pose`] let you switch
    /// representations without going through `exp` / `log`:
    ///
    /// ```
    /// use se3_ad_recipes::api::pose::Pose;
    /// use se3_ad_recipes::api::quaternion_pose::QuatPose;
    ///
    /// let pose = Pose::exp(&[0.2, -0.1, 0.3, 1.0, -2.0, 0.5]);
    ///
    /// // Switch to quaternion storage; the rotation algebra is identical.
    /// let quat_pose: QuatPose = (&pose).into();
    /// let round_trip: Pose = (&quat_pose).into();
    ///
    /// // The bridge is bit-equivalent up to float arithmetic noise.
    /// for row in 0..3 {
    ///     for column in 0..3 {
    ///         let diff = pose.rotation()[row][column] - round_trip.rotation()[row][column];
    ///         assert!(diff.abs() < 1e-14);
    ///     }
    /// }
    /// ```
    #[derive(Clone, Copy)]
    #[must_use = "QuatPose is a pure value; ignoring it drops the computed transform"]
    pub struct QuatPose {
        inner: PoseQ<f64>,
    }

    impl QuatPose {
        /// Identity pose: `q = (1, 0)`, `t = 0`.
        pub const fn identity() -> Self {
            Self {
                inner: PoseQ {
                    q0: 1.0,
                    qv: [0.0; 3],
                    trans: [0.0; 3],
                },
            }
        }

        /// Construct from `(q₀, q_v, t)` validating unit-norm of the
        /// quaternion within [`DEFAULT_QUATERNION_TOLERANCE`]. All components
        /// must be finite.
        pub fn from_quaternion_translation(
            q0: f64,
            qv: &Vec3,
            trans: &Vec3,
        ) -> Result<Self, QuatPoseError> {
            Self::from_quaternion_translation_with_tolerance(
                q0,
                qv,
                trans,
                DEFAULT_QUATERNION_TOLERANCE,
            )
        }

        /// As [`QuatPose::from_quaternion_translation`] but with a
        /// caller-supplied tolerance.
        pub fn from_quaternion_translation_with_tolerance(
            q0: f64,
            qv: &Vec3,
            trans: &Vec3,
            tolerance: f64,
        ) -> Result<Self, QuatPoseError> {
            check_tolerance(tolerance)?;
            if !q0.is_finite() {
                return Err(super::InputError::NonFiniteInput {
                    field: "quaternion",
                }
                .into());
            }
            check_finite_vec(qv, "quaternion")?;
            check_finite_vec(trans, "translation")?;
            let n = q0 * q0 + qv[0] * qv[0] + qv[1] * qv[1] + qv[2] * qv[2];
            let residual = (n - 1.0).abs();
            if residual > tolerance {
                return Err(QuatPoseError::NotUnitNorm {
                    norm_residual: residual,
                    tolerance,
                });
            }
            Ok(Self {
                inner: PoseQ {
                    q0,
                    qv: *qv,
                    trans: *trans,
                },
            })
        }

        /// Construct from raw parts without validation. Caller asserts the
        /// quaternion is unit-norm.
        pub fn from_parts_unchecked(q0: f64, qv: Vec3, trans: Vec3) -> Self {
            Self {
                inner: PoseQ { q0, qv, trans },
            }
        }

        /// Exponential map from a rotation-first twist (same convention as
        /// [`Pose::exp`]).
        pub fn exp(twist: &Vec6) -> Self {
            Self {
                inner: PoseQ::exp(twist),
            }
        }

        /// Logarithmic map to a rotation-first twist.
        #[must_use]
        pub fn log(&self) -> Vec6 {
            self.inner.log()
        }

        /// Group composition `self · other`.
        pub fn compose(&self, other: &Self) -> Self {
            Self {
                inner: self.inner.compose(&other.inner),
            }
        }

        /// Group inverse.
        pub fn inverse(&self) -> Self {
            Self {
                inner: self.inner.inverse(),
            }
        }

        /// Apply the pose to a point: `R(q) · point + t`.
        #[must_use]
        pub fn act(&self, point: &Vec3) -> Vec3 {
            self.inner.act(point)
        }

        /// Apply the inverse pose to a point: `R(q)ᵀ · (point - t)`.
        #[must_use]
        pub fn act_inverse(&self, point: &Vec3) -> Vec3 {
            self.inner.act_inverse(point)
        }

        /// Convert to the matrix-storage [`Pose`] form.
        pub fn to_pose(&self) -> Pose {
            let g = self.inner.to_pose_g();
            Pose::from_parts_unchecked(g.rot, g.trans)
        }

        /// Quaternion scalar component.
        pub fn q0(&self) -> f64 {
            self.inner.q0
        }

        /// Quaternion vector component.
        pub fn qv(&self) -> &Vec3 {
            &self.inner.qv
        }

        /// Read-only access to the translation vector.
        pub fn translation(&self) -> &Vec3 {
            &self.inner.trans
        }

        /// Adjoint representation: `Ad(self) → [f64; 6×6]`. Delegates
        /// through the matrix-storage form.
        #[must_use]
        pub fn adjoint(&self) -> crate::Mat6 {
            self.to_pose().adjoint()
        }
    }

    impl Default for QuatPose {
        fn default() -> Self {
            Self::identity()
        }
    }

    impl core::fmt::Debug for QuatPose {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("QuatPose")
                .field("q0", &self.inner.q0)
                .field("qv", &self.inner.qv)
                .field("translation", &self.inner.trans)
                .finish()
        }
    }

    impl PartialEq for QuatPose {
        fn eq(&self, other: &Self) -> bool {
            self.inner.q0 == other.inner.q0
                && self.inner.qv == other.inner.qv
                && self.inner.trans == other.inner.trans
        }
    }

    #[cfg(feature = "serde")]
    #[derive(serde::Serialize, serde::Deserialize)]
    struct QuatPoseRepr {
        q0: f64,
        qv: Vec3,
        translation: Vec3,
    }

    #[cfg(feature = "serde")]
    impl serde::Serialize for QuatPose {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            QuatPoseRepr {
                q0: self.inner.q0,
                qv: self.inner.qv,
                translation: self.inner.trans,
            }
            .serialize(serializer)
        }
    }

    #[cfg(feature = "serde")]
    impl<'de> serde::Deserialize<'de> for QuatPose {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let repr = QuatPoseRepr::deserialize(deserializer)?;
            QuatPose::from_quaternion_translation(repr.q0, &repr.qv, &repr.translation)
                .map_err(serde::de::Error::custom)
        }
    }

    impl From<&Pose> for QuatPose {
        fn from(pose: &Pose) -> Self {
            let g = crate::se3_adsafe::PoseG {
                rot: *pose.rotation(),
                trans: *pose.translation(),
            };
            Self {
                inner: PoseQ::from_pose_g(&g),
            }
        }
    }

    impl From<Pose> for QuatPose {
        fn from(pose: Pose) -> Self {
            (&pose).into()
        }
    }

    impl From<&QuatPose> for Pose {
        fn from(qp: &QuatPose) -> Self {
            qp.to_pose()
        }
    }

    impl From<QuatPose> for Pose {
        fn from(qp: QuatPose) -> Self {
            (&qp).into()
        }
    }

    /// Apply a right perturbation: `pose · Exp(delta)`.
    pub fn right_update(pose: &QuatPose, delta: &Vec6) -> QuatPose {
        pose.compose(&QuatPose::exp(delta))
    }

    /// Apply a left perturbation: `Exp(delta) · pose`.
    pub fn left_update(delta: &Vec6, pose: &QuatPose) -> QuatPose {
        QuatPose::exp(delta).compose(pose)
    }

    /// Jacobian of `pose · Exp(delta).act(point)` evaluated at delta = 0.
    ///
    /// Routes through the matrix-storage representation: the SE(3) Jacobian
    /// depends only on `R` and `point`, so this is bit-equivalent to
    /// [`super::pose::right_point_action_jacobian`] applied to
    /// [`QuatPose::to_pose`]'s output.
    #[must_use]
    pub fn right_point_action_jacobian(pose: &QuatPose, point: &Vec3) -> super::types::Mat3x6 {
        super::pose::right_point_action_jacobian(&pose.to_pose(), point)
    }
}

/// AD-generic API, organized by group.
///
/// Standard semantic versioning applies — additions ship in minor
/// releases; renames or removals require a major release. Names are
/// dimension-agnostic and free of implementation suffixes, and are
/// intended to outlive the paper's evolving notation. Semantics follow
/// the paper's AD-safe fused scalar basis (smooth at the SO(3) origin);
/// specific small-angle / antipodal thresholds and Taylor depths are
/// implementation details and may be tightened in minor releases when
/// they remain numerically AD-safe at every supported derivative depth.
pub mod expert {
    /// Generic point-action trait. Implemented by `expert::se3::PoseG` and
    /// the per-channel methods on `expert::se23::ExtendedPoseG`.
    pub use crate::act::Act;

    /// AD-generic fixed-size matrix, vector, and tensor type aliases.
    ///
    /// The matrix and vector aliases are the load-bearing primitives every
    /// expert-tier operation flows through — `pub type` aliases for
    /// row-major `[[T; N]; N]` matrices and `[T; N]` vectors, so AD-generic
    /// code can construct values with array literals and pattern-match on
    /// indices without importing anything else. The semantic tensor
    /// aliases (`JacobianDerivative6`, `Hessian6`) document operations'
    /// return shapes at the type level.
    pub mod types {
        pub use crate::se3_adsafe::{Mat6G, Vec6G};
        pub use crate::se23_adsafe::{Mat9G, Vec9G};
        pub use crate::so3_adsafe::{Mat3G, Vec3G};

        /// SE(3) Jacobian derivative tensor: six 6×6 slices indexed by
        /// the tangent direction `m ∈ {0..6}`. Slice `m` is `∂Jr/∂ξ_m`
        /// (or `∂Jr⁻¹/∂ξ_m`, depending on the producing function).
        pub use crate::jacobians_ad::JacobianDerivative6G as JacobianDerivative6;

        /// 6×6×6 cubic tensor over the AD scalar `T`. Used as the SE(3)
        /// chart-Hessian return type.
        pub use crate::jacobians_ad::Tensor666G as Hessian6;
    }

    /// AD-generic linear algebra: dimension-generic core ops plus the
    /// block-assembly helpers needed to compose Lie-group Jacobians.
    pub mod linalg {
        use super::types::{Mat3G, Mat6G, Mat9G};
        use crate::autodiff::ad_trait::AD;

        /// N×N matrix multiplication: `c[i][j] = Σ_k a[i][k] · b[k][j]`.
        ///
        /// Const-generic over `N`, so the same call works for 3×3, 6×6,
        /// or 9×9 matrices — the dimension is inferred from the input
        /// array types.
        ///
        /// ```
        /// use se3_ad_recipes::api::expert::linalg::matmul;
        ///
        /// let identity = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        /// let scale = [[2.0_f64, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]];
        /// let product = matmul(&identity, &scale);
        /// assert_eq!(product, scale);
        /// ```
        pub fn matmul<T: AD, const N: usize>(a: &[[T; N]; N], b: &[[T; N]; N]) -> [[T; N]; N] {
            let z = T::constant(0.0);
            let mut c = [[z; N]; N];
            for i in 0..N {
                for j in 0..N {
                    let mut s = z;
                    for k in 0..N {
                        s += a[i][k] * b[k][j];
                    }
                    c[i][j] = s;
                }
            }
            c
        }

        /// N×N matrix times N-vector: `r[i] = Σ_j m[i][j] · v[j]`.
        pub fn matvec<T: AD, const N: usize>(m: &[[T; N]; N], v: &[T; N]) -> [T; N] {
            let z = T::constant(0.0);
            let mut r = [z; N];
            for i in 0..N {
                let mut s = z;
                for j in 0..N {
                    s += m[i][j] * v[j];
                }
                r[i] = s;
            }
            r
        }

        /// N×N matrix transpose.
        pub fn transpose<T: AD, const N: usize>(m: &[[T; N]; N]) -> [[T; N]; N] {
            let z = T::constant(0.0);
            let mut t = [[z; N]; N];
            for i in 0..N {
                for j in 0..N {
                    t[i][j] = m[j][i];
                }
            }
            t
        }

        /// Assemble a 6×6 matrix from a 2×2 grid of 3×3 blocks.
        ///
        /// ```text
        /// blocks = [[tl, tr],          6×6 result =  | tl  tr |
        ///           [bl, br]]                        | bl  br |
        /// ```
        ///
        /// Used to compose SE(3) adjoints, right Jacobians, and their
        /// derivative tensors from constituent 3×3 SO(3) pieces.
        pub fn block_2x2<T: AD>(blocks: [[Mat3G<T>; 2]; 2]) -> Mat6G<T> {
            let z = T::constant(0.0);
            let mut m = [[z; 6]; 6];
            for br in 0..2 {
                for bc in 0..2 {
                    let block = blocks[br][bc];
                    for r in 0..3 {
                        for c in 0..3 {
                            m[br * 3 + r][bc * 3 + c] = block[r][c];
                        }
                    }
                }
            }
            m
        }

        /// Assemble a 9×9 matrix from a 3×3 grid of 3×3 blocks.
        ///
        /// Used to compose SE_2(3) adjoints and right Jacobians.
        pub fn block_3x3<T: AD>(blocks: [[Mat3G<T>; 3]; 3]) -> Mat9G<T> {
            let z = T::constant(0.0);
            let mut m = [[z; 9]; 9];
            for br in 0..3 {
                for bc in 0..3 {
                    let block = blocks[br][bc];
                    for r in 0..3 {
                        for c in 0..3 {
                            m[br * 3 + r][bc * 3 + c] = block[r][c];
                        }
                    }
                }
            }
            m
        }
    }

    /// SO(3) — rotation matrix group.
    ///
    /// AD-generic SO(3) operations exposed under clean names (no `_g`
    /// suffix). The per-paper scalar basis atoms (A, B, C, D, β, β̄,
    /// `theta_sq_from_omega`, etc.) are intentionally *not* re-exported
    /// here — users constructing new analytic SE(3) derivatives following
    /// the paper's §V.D recipe should reach into the raw module
    /// `crate::so3_adsafe::*` directly. Those names follow paper notation
    /// and have no stability guarantee (raw-module tier).
    pub mod so3 {
        /// AD-generic SO(3) exponential map: rotation vector → rotation matrix.
        ///
        /// ```
        /// use se3_ad_recipes::api::expert::so3;
        ///
        /// // exp(0) = identity.
        /// let r = so3::exp::<f64>(&[0.0, 0.0, 0.0]);
        /// assert_eq!(r, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        ///
        /// // log(exp(omega)) ≈ omega.
        /// let omega = [0.2, -0.1, 0.3];
        /// let recovered = so3::log::<f64>(&so3::exp::<f64>(&omega));
        /// for (a, b) in recovered.iter().zip(omega.iter()) {
        ///     assert!((a - b).abs() < 1e-12);
        /// }
        /// ```
        pub use crate::so3_adsafe::so3_exp_g as exp;

        /// AD-generic SO(3) principal logarithm: rotation matrix → rotation vector.
        pub use crate::so3_adsafe::so3_log_g as log;

        /// AD-generic SO(3) logarithm via the SU(2) (quaternion) path.
        /// Preferred near the antipodal locus where the matrix log is
        /// numerically inconvenient.
        pub use crate::so3_adsafe::so3_log_g_quaternion as log_quaternion;

        /// AD-generic hat map: rotation vector → 3×3 skew-symmetric matrix.
        pub use crate::so3_adsafe::hat_g as hat;

        /// AD-generic SO(3) generator matrices `E_m`, m ∈ {0, 1, 2}.
        /// Returns the m-th of the three basis hats (`∂R/∂ω_m` at the
        /// identity). Useful when composing analytic derivatives.
        pub use crate::so3_adsafe::hat_basis_g as hat_basis;

        /// AD-generic SO(3) right Jacobian.
        pub use crate::so3_adsafe::jr_g as right_jacobian;

        /// AD-generic SO(3) inverse right Jacobian.
        pub use crate::so3_adsafe::jr_inv_g as right_jacobian_inverse;

        /// AD-generic V matrix: `∫₀¹ R(sω) ds = I + B(θ) [ω]_× + C(θ) [ω]_×²`.
        /// Used to build the SE(3) right Jacobian and the SE_2(3) exp.
        pub use crate::so3_adsafe::v_matrix_g as v_matrix;

        /// AD-generic V⁻¹ matrix.
        pub use crate::so3_adsafe::v_inv_g as v_inverse;

        /// AD-generic Shepperd's algorithm: rotation matrix → unit
        /// quaternion `(q₀, q_v)`. Used by quaternion-storage SE(3).
        pub use crate::so3_adsafe::mat3_to_quat_shepperd_g as mat3_to_quaternion;
    }

    /// SE(3) — rotation-matrix-storage rigid-body group.
    ///
    /// AD-generic operations exposed under clean, dimension-agnostic
    /// names. Internal helpers (per-slice partials, scalar basis atoms)
    /// are intentionally *not* re-exported — reach for the high-level
    /// entry points `right_jacobian_derivative` and
    /// `right_jacobian_directional_derivative` instead of composing slabs
    /// by hand.
    pub mod se3 {
        /// AD-generic backing struct for the application-tier
        /// `api::pose::Pose`: `{ rot: Mat3G<T>, trans: Vec3G<T> }` with
        /// `compose`, `inverse`, `exp`, `log`, and `Act` point-action
        /// implementations.
        pub use crate::se3_adsafe::PoseG;

        /// AD-generic SE(3) right Jacobian `Jr(ξ) → Mat6G<T>`.
        pub use crate::se3_adsafe::se3_jr_g as right_jacobian;

        /// AD-generic SE(3) inverse right Jacobian `Jr(ξ)⁻¹ → Mat6G<T>`.
        pub use crate::se3_adsafe::se3_jr_inv_g as right_jacobian_inverse;

        /// AD-generic SE(3) right-Jacobian derivative tensor `∂Jr/∂ξ`.
        ///
        /// Returns six 6×6 slices (one per tangent direction) packed as
        /// `[Mat6G<T>; 6]`.
        pub use crate::jacobians_ad::se3_jr_derivative_g as right_jacobian_derivative;

        /// AD-generic SE(3) inverse-right-Jacobian derivative tensor
        /// `∂Jr⁻¹/∂ξ`. Same `[Mat6G<T>; 6]` shape as
        /// [`right_jacobian_derivative`].
        pub use crate::jacobians_ad::se3_jr_inv_derivative_g as right_jacobian_inverse_derivative;

        /// Directional derivative of the SE(3) right Jacobian:
        /// `Σ_m direction[m] · ∂Jr/∂ξ_m`, returned as a single 6×6 matrix.
        ///
        /// Convenience entry point for single-direction work (Hessian-vector
        /// products in iterative solvers); when several directions are
        /// needed in a hot loop, prefer caching the full
        /// `right_jacobian_derivative` tensor.
        ///
        /// ```
        /// use se3_ad_recipes::api::expert::se3;
        ///
        /// let xi = [0.1, -0.2, 0.05, 1.0, -0.5, 0.3];
        /// let direction = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        ///
        /// // Directional derivative against e_1 equals the second slice
        /// // of the full ∂Jr/∂ξ tensor.
        /// let slice = se3::right_jacobian_directional_derivative::<f64>(&xi, &direction);
        /// let tensor = se3::right_jacobian_derivative::<f64>(&xi);
        /// for row in 0..6 {
        ///     for column in 0..6 {
        ///         assert!((slice[row][column] - tensor[1][row][column]).abs() < 1e-14);
        ///     }
        /// }
        /// ```
        pub use crate::jacobians_ad::se3_jr_directional_derivative_g as right_jacobian_directional_derivative;

        /// Directional derivative of the SE(3) inverse right Jacobian.
        /// See `right_jacobian_directional_derivative` for the use case.
        pub use crate::jacobians_ad::se3_jr_inv_directional_derivative_g as right_jacobian_inverse_directional_derivative;

        /// AD-generic SE(3) adjoint matrix `Ad(pose) → Mat6G<T>`. Takes a
        /// pose value rather than separated `(rotation, translation)` —
        /// the raw-parts form lives in the doc-hidden raw module.
        pub fn adjoint<T: crate::autodiff::ad_trait::AD>(
            pose: &PoseG<T>,
        ) -> crate::se3_adsafe::Mat6G<T> {
            crate::se3_adsafe::adjoint_g(&pose.rot, &pose.trans)
        }

        /// AD-generic SE(3) translational coupling matrix `Q̃_r(ω, t) →
        /// Mat3G<T>`. Exposed for callers writing their own SE(3)
        /// chart-Hessian derivations. Takes tangent-space `(ω, t)`
        /// (this is a tangent operator, not a group operator).
        pub use crate::se3_adsafe::q_tilde_r_g as q_tilde_r;
    }

    /// SE_2(3) — extended pose group for inertial navigation.
    ///
    /// AD-generic operations exposed under the same clean naming
    /// convention as [`se3`]. Adjoint variants take an `ExtendedPoseG`
    /// value rather than separated `(rotation, velocity, position)` —
    /// the raw-parts forms live in the doc-hidden raw module.
    pub mod se23 {
        /// AD-generic backing struct for the application-tier
        /// `api::extended_pose::ExtendedPose`:
        /// `{ rot: Mat3G<T>, vel: Vec3G<T>, pos: Vec3G<T> }`.
        pub use crate::se23_adsafe::ExtendedPoseG;

        /// AD-generic SE_2(3) right Jacobian `Jr(xi) → Mat9G<T>`.
        pub use crate::jacobians_se23_adsafe::se23_jr_g as right_jacobian;

        /// AD-generic SE_2(3) inverse right Jacobian.
        pub use crate::jacobians_se23_adsafe::se23_jr_inv_g as right_jacobian_inverse;

        /// AD-generic SE_2(3) adjoint matrix `Ad(pose) → Mat9G<T>`.
        pub fn adjoint<T: crate::autodiff::ad_trait::AD>(
            pose: &ExtendedPoseG<T>,
        ) -> crate::se23_adsafe::Mat9G<T> {
            crate::jacobians_se23_adsafe::adjoint_se23_g(&pose.rot, &pose.vel, &pose.pos)
        }

        /// AD-generic SE_2(3) inverse adjoint matrix.
        pub fn adjoint_inverse<T: crate::autodiff::ad_trait::AD>(
            pose: &ExtendedPoseG<T>,
        ) -> crate::se23_adsafe::Mat9G<T> {
            crate::jacobians_se23_adsafe::adjoint_inv_se23_g(&pose.rot, &pose.vel, &pose.pos)
        }
    }

    /// Quaternion-storage SE(3) — same group as [`se3`] in different coordinates.
    pub mod quat_se3 {
        /// AD-generic backing struct for the application-tier
        /// `api::quaternion_pose::QuatPose`:
        /// `{ q0: T, qv: Vec3G<T>, trans: Vec3G<T> }`.
        pub use crate::se3_quat_adsafe::PoseQ;

        /// Build a rotation matrix from a unit quaternion `(q₀, q_v)`.
        pub use crate::se3_quat_adsafe::quat_to_rotmat as to_rotation_matrix;

        /// Apply a unit quaternion `(q₀, q_v)` to a 3-vector without
        /// materializing the rotation matrix.
        pub use crate::se3_quat_adsafe::quat_rotate_vec as rotate_vector;
    }

    /// Pinhole-projective operations.
    ///
    /// The top-level surface here is the AD-generic geometric layer —
    /// projection, its derivatives, the raw-storage point transform, and
    /// the `[I | -hat(x)]` stacking matrix. Statistical derivatives
    /// (likelihood, information matrix, cumulant tensors) live in the
    /// `statistics` submodule and are currently `f64` only. Neither
    /// layer is in the default prelude — advanced surface for users
    /// assembling reprojection-residual cost functions.
    pub mod projective {
        /// Pinhole projection: `[x, y, z] → [x/z, y/z]`.
        ///
        /// ```
        /// use se3_ad_recipes::api::expert::projective::project;
        ///
        /// let pixel = project::<f64>(&[2.0, 4.0, 2.0]);
        /// assert!((pixel[0] - 1.0).abs() < 1e-14);
        /// assert!((pixel[1] - 2.0).abs() < 1e-14);
        /// ```
        pub use crate::projective::project_g as project;

        /// Jacobian of the projection: 2×3 matrix `∂π/∂x` packed as `[[T; 3]; 2]`.
        pub use crate::projective::project_jacobian_g as project_jacobian;

        /// Hessian of the projection: returns two 3×3 matrices, one per
        /// pixel coordinate, as `(Mat3G<T>, Mat3G<T>)`.
        pub use crate::projective::project_hessian_g as project_hessian;

        /// Raw `(R, t)` point transform `x ↦ R · x + t`. The free-function
        /// equivalent of [`crate::api::pose::Pose::act`] without
        /// constructing a `Pose`.
        pub use crate::projective::transform_point_g as transform_point;

        /// Stacking matrix `J_×(x) = [I_3 | -hat(x)]` returned as `[[T; 6]; 3]`,
        /// the kinematic structure that appears in the right-perturbation
        /// Jacobian of point actions.
        pub use crate::projective::j_cross_g as j_cross;

        /// Statistical derivatives of the projective likelihood:
        /// information matrix, third cumulants, and a scalar fourth-order
        /// saddlepoint correction. Currently `f64` only — the cumulant
        /// tensors don't carry an AD-generic implementation yet.
        pub mod statistics {
            /// Apply intrinsic calibration to a pixel observation and its
            /// 2×2 measurement precision matrix.
            ///
            /// `z → K⁻¹ z`, `Σ⁻¹ → Kᵀ Σ⁻¹ K`. Pre-processing step before
            /// the rest of the statistical pipeline operates in normalized
            /// camera coordinates.
            pub use crate::projective::apply_calibration as calibrate;

            /// Reprojection error `z − π(x')` in normalized camera coordinates.
            pub use crate::projective::reprojection_error;

            /// Negative log-likelihood of a Gaussian projective
            /// measurement: `½ · (z − π)ᵀ Σ⁻¹ (z − π)`.
            pub use crate::projective::neg_log_likelihood;

            /// Fisher information matrix in landmark-frame coordinates,
            /// `Rᵀ · (Pᵀ Σ⁻¹ P) · R`, where `P = ∂π/∂x'`.
            pub use crate::projective::measurement_info_matrix as information_matrix;

            /// Third cumulants of the projective likelihood rotated to
            /// landmark frame. Returns a 3×3×3 tensor of leading-order
            /// asymmetry contributions used by the saddlepoint expansion.
            pub use crate::projective::third_cumulants;

            /// Fourth-order quartic correction for the saddlepoint /
            /// Edgeworth expansion of the projective likelihood.
            ///
            /// Returns the scalar contraction of the fourth-cumulant
            /// tensor with the inverse Hessian, the leading 1/N correction
            /// to the Gaussian approximation around the mode.
            pub use crate::projective::quartic_contraction_analytical as quartic_correction;
        }
    }

    /// Automatic differentiation framework: forward, reverse, and nested duals.
    ///
    /// Explicit allow-list of stable AD backends. The `autodiff` raw module
    /// contains additional internal types and helpers that are deliberately
    /// not re-exported — only the entries below are part of the expert-tier
    /// contract.
    ///
    /// Pick the backend by computation order:
    /// - **First-order forward**: use `adfn` (fixed-size tangent `N`).
    /// - **Higher-order forward**: use `D2` for Hessians and `D3` for
    ///   third-order tensors. Both alias chains of `Dual`.
    /// - **Forward-over-reverse**: use `adr_n6` with the `tape_n6_clear`
    ///   / `tape_n6_backward` / `tape_n6_backward_into` tape ops.
    ///
    /// Note: `Dual<f64, N>` is functionally equivalent to `adfn<N>`; the
    /// flat `D1<N>` alias is intentionally not exported to keep a single
    /// blessed first-order forward type.
    pub mod ad {
        /// Core trait every numerical routine in the expert tier is
        /// generic over. Implemented by `f64` (plain evaluation), every
        /// `adfn<N>`, every `Dual<T, N>`, and `adr_n6`.
        pub use crate::autodiff::ad_trait::AD;

        /// Forward-mode AD scalar with compile-time tangent dimension `N`.
        /// One forward sweep yields the gradient with respect to `N` seeded
        /// inputs simultaneously.
        pub use crate::autodiff::forward_ad::adfn;

        /// Nestable forward-mode dual: `Dual<T, N>` carries a value of
        /// type `T` plus an `N`-tangent of `T`s. Nest to reach arbitrary
        /// derivative order. See `D2` / `D3` for the common cases.
        pub use crate::autodiff::nested_ad::Dual;

        /// Two-level nested dual: `Dual<Dual<f64, N>, N>`. One evaluation
        /// computes value, gradient, and full N×N Hessian.
        pub use crate::autodiff::nested_ad::D2;

        /// Three-level nested dual for third-order tensors.
        pub use crate::autodiff::nested_ad::D3;

        /// Reverse-mode AD scalar whose partials are themselves `adfn<6>`.
        /// One forward + one reverse pass yields a full Hessian (automatic
        /// forward-over-reverse).
        pub use crate::autodiff::reverse_ad_n6::adr_n6;

        /// Reset the global tape used by `adr_n6`. Call before each
        /// independent computation.
        pub use crate::autodiff::reverse_ad_n6::tape_n6_clear;

        /// Run the reverse sweep for one scalar output (by tape index)
        /// and return its adjoint vector.
        pub use crate::autodiff::reverse_ad_n6::tape_n6_backward;

        /// Same as `tape_n6_backward` but writes into a caller-supplied
        /// buffer, avoiding allocation in tight loops.
        pub use crate::autodiff::reverse_ad_n6::tape_n6_backward_into;
    }
}

#[cfg(test)]
mod tests {
    use super::InputError;
    use super::extended_pose::{self, ExtendedPose};
    use super::pose;
    use super::pose::{Pose, PoseError};
    use super::quaternion_pose::{QuatPose, QuatPoseError};
    use crate::{I3, frob_diff, l2_diff};

    #[test]
    fn facade_pose_roundtrip_and_action() {
        let twist = [0.2, -0.1, 0.3, 1.0, -2.0, 0.5];
        let pose = Pose::exp(&twist);
        assert!(l2_diff(&pose.log(), &twist) < 1e-10);

        let point = [0.4, 0.5, -0.2];
        assert!(l2_diff(&pose.act_inverse(&pose.act(&point)), &point) < 1e-10);
    }

    #[test]
    fn identity_constructors_are_const_evaluable() {
        const POSE: Pose = Pose::identity();
        const QUAT_POSE: QuatPose = QuatPose::identity();
        const EXTENDED_POSE: ExtendedPose = ExtendedPose::identity();
        assert_eq!(POSE, Pose::identity());
        assert_eq!(QUAT_POSE, QuatPose::identity());
        assert_eq!(EXTENDED_POSE, ExtendedPose::identity());
    }

    #[test]
    fn pose_standard_traits() {
        let d = Pose::default();
        assert_eq!(d, Pose::identity());
        assert_eq!(d.rotation(), &I3);
        assert_eq!(d.translation(), &[0.0; 3]);
        let s = format!("{d:?}");
        assert!(s.contains("Pose"));
        assert!(s.contains("rotation"));
        assert!(s.contains("translation"));
        let a = Pose::exp(&[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = Pose::exp(&[0.1, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_ne!(a, b);
    }

    #[test]
    fn from_rotation_translation_accepts_exp_output() {
        let p = Pose::exp(&[0.2, -0.1, 0.3, 1.0, -2.0, 0.5]);
        let round = Pose::from_rotation_translation(p.rotation(), p.translation()).unwrap();
        assert_eq!(round, p);
    }

    #[test]
    fn from_rotation_translation_rejects_non_orthogonal() {
        let bad = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]];
        match Pose::from_rotation_translation(&bad, &[0.0; 3]) {
            Err(PoseError::NotOrthogonal {
                orthogonality_residual,
                ..
            }) => assert!(orthogonality_residual > 0.1),
            other => panic!("expected NotOrthogonal, got {other:?}"),
        }
    }

    #[test]
    fn from_rotation_translation_rejects_reflection() {
        let reflection = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
        match Pose::from_rotation_translation(&reflection, &[0.0; 3]) {
            Err(PoseError::NotProperRotation { determinant, .. }) => {
                assert!((determinant + 1.0).abs() < 1e-12)
            }
            other => panic!("expected NotProperRotation, got {other:?}"),
        }
    }

    #[test]
    fn from_parts_unchecked_skips_validation() {
        let bad = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let p = Pose::from_parts_unchecked(bad, [1.0, 2.0, 3.0]);
        assert_eq!(p.rotation(), &bad);
        assert_eq!(p.translation(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn pose_error_implements_display_and_error_trait() {
        let err = Pose::from_rotation_translation(
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]],
            &[0.0; 3],
        )
        .unwrap_err();
        let _: &dyn std::error::Error = &err;
        assert!(!format!("{err}").is_empty());
    }

    #[test]
    fn checked_pose_constructors_reject_non_finite_data_and_invalid_tolerance() {
        let mut rotation = I3;
        rotation[0][0] = f64::NAN;
        assert!(matches!(
            Pose::from_rotation_translation(&rotation, &[0.0; 3]),
            Err(PoseError::Input(InputError::NonFiniteInput {
                field: "rotation"
            }))
        ));
        assert!(matches!(
            Pose::from_rotation_translation(&I3, &[f64::INFINITY, 0.0, 0.0]),
            Err(PoseError::Input(InputError::NonFiniteInput {
                field: "translation"
            }))
        ));
        for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                Pose::from_rotation_translation_with_tolerance(&I3, &[0.0; 3], tolerance),
                Err(PoseError::Input(InputError::InvalidTolerance { .. }))
            ));
        }
    }

    #[test]
    fn extended_pose_round_trip_and_action_channels() {
        let xi = [0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 1.0, -2.0, 0.5];
        let g = ExtendedPose::exp(&xi);
        assert!(l2_diff(&g.log(), &xi) < 1e-10);

        let x = [0.7, -0.5, 2.0];
        assert!(l2_diff(&g.act_position_inverse(&g.act_position(&x)), &x) < 1e-10);
        assert!(l2_diff(&g.act_velocity_inverse(&g.act_velocity(&x)), &x) < 1e-10);
    }

    #[test]
    fn extended_pose_default_identity_and_partial_eq() {
        let id = ExtendedPose::default();
        assert_eq!(id, ExtendedPose::identity());
        assert_eq!(id.rotation(), &I3);
        assert_eq!(id.velocity(), &[0.0; 3]);
        assert_eq!(id.position(), &[0.0; 3]);
    }

    #[test]
    fn extended_pose_from_parts_validates_rotation() {
        let reflection = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
        match ExtendedPose::from_rotation_velocity_position(
            &reflection,
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
        ) {
            Err(PoseError::NotProperRotation { determinant, .. }) => {
                assert!((determinant + 1.0).abs() < 1e-12)
            }
            other => panic!("expected NotProperRotation, got {other:?}"),
        }
    }

    #[test]
    fn checked_extended_pose_rejects_non_finite_channels() {
        assert!(matches!(
            ExtendedPose::from_rotation_velocity_position(&I3, &[f64::NAN, 0.0, 0.0], &[0.0; 3]),
            Err(PoseError::Input(InputError::NonFiniteInput {
                field: "velocity"
            }))
        ));
        assert!(matches!(
            ExtendedPose::from_rotation_velocity_position(
                &I3,
                &[0.0; 3],
                &[0.0, f64::INFINITY, 0.0]
            ),
            Err(PoseError::Input(InputError::NonFiniteInput {
                field: "position"
            }))
        ));
    }

    #[test]
    fn quat_pose_round_trip() {
        let twist = [0.2, -0.1, 0.3, 1.0, -2.0, 0.5];
        let q = QuatPose::exp(&twist);
        assert!(l2_diff(&q.log(), &twist) < 1e-10);
    }

    #[test]
    fn pose_log_quaternion_round_trips_and_matches_matrix_log_generically() {
        let twist = [0.2, -0.1, 0.3, 1.0, -2.0, 0.5];
        let pose = Pose::exp(&twist);
        // Round-trip through the quaternion log path.
        assert!(l2_diff(&pose.log_quaternion(), &twist) < 1e-10);
        // Away from the antipodal locus, matrix log and quaternion log agree.
        assert!(l2_diff(&pose.log_quaternion(), &pose.log()) < 1e-12);
    }

    #[test]
    fn quat_pose_bridges_with_pose_match_numerically() {
        let twist = [0.2, -0.1, 0.3, 1.0, -2.0, 0.5];
        let p = Pose::exp(&twist);
        let q: QuatPose = p.into();
        let p_round: Pose = q.into();
        assert!(frob_diff(p_round.rotation(), p.rotation()) < 1e-14);
        assert!(l2_diff(p_round.translation(), p.translation()) < 1e-14);
    }

    #[test]
    fn pose_quat_bridges_accept_references() {
        let p = Pose::exp(&[0.2, -0.1, 0.3, 1.0, -2.0, 0.5]);
        let q_from_ref: QuatPose = (&p).into();
        let q_from_val: QuatPose = p.into();
        assert_eq!(q_from_ref, q_from_val);
        let p_from_ref: Pose = (&q_from_ref).into();
        let p_from_val: Pose = q_from_ref.into();
        assert_eq!(p_from_ref, p_from_val);
    }

    #[test]
    fn quat_pose_unit_norm_check_rejects_garbage() {
        match QuatPose::from_quaternion_translation(0.5, &[0.5, 0.5, 0.5], &[0.0; 3]) {
            Ok(_) => {} // 0.25 + 0.75 = 1.0 — actually unit norm
            Err(e) => panic!("expected ok for unit quaternion, got {e:?}"),
        }
        match QuatPose::from_quaternion_translation(2.0, &[0.0, 0.0, 0.0], &[0.0; 3]) {
            Err(QuatPoseError::NotUnitNorm { norm_residual, .. }) => {
                assert!(norm_residual > 1.0)
            }
            other => panic!("expected NotUnitNorm, got {other:?}"),
        }
    }

    #[test]
    fn checked_quat_pose_rejects_non_finite_data_and_invalid_tolerance() {
        assert!(matches!(
            QuatPose::from_quaternion_translation(f64::NAN, &[0.0; 3], &[0.0; 3]),
            Err(QuatPoseError::Input(InputError::NonFiniteInput {
                field: "quaternion"
            }))
        ));
        assert!(matches!(
            QuatPose::from_quaternion_translation(1.0, &[0.0; 3], &[0.0, 0.0, f64::NAN]),
            Err(QuatPoseError::Input(InputError::NonFiniteInput {
                field: "translation"
            }))
        ));
        assert!(matches!(
            QuatPose::from_quaternion_translation_with_tolerance(
                1.0,
                &[0.0; 3],
                &[0.0; 3],
                f64::NAN
            ),
            Err(QuatPoseError::Input(InputError::InvalidTolerance { .. }))
        ));
    }

    #[test]
    fn quat_pose_actions_and_updates_match_matrix_pose() {
        let twist = [0.2, -0.1, 0.3, 1.0, -2.0, 0.5];
        let delta = [0.01, -0.02, 0.03, 0.1, 0.2, -0.1];
        let point = [0.7, -0.5, 2.0];
        let quaternion_pose = QuatPose::exp(&twist);
        let matrix_pose: Pose = quaternion_pose.into();

        assert!(l2_diff(&quaternion_pose.act(&point), &matrix_pose.act(&point)) < 1e-14);
        assert!(
            l2_diff(
                &quaternion_pose.act_inverse(&point),
                &matrix_pose.act_inverse(&point)
            ) < 1e-14
        );

        let quaternion_right = super::quaternion_pose::right_update(&quaternion_pose, &delta);
        let matrix_right = pose::right_update(&matrix_pose, &delta);
        let quaternion_left = super::quaternion_pose::left_update(&delta, &quaternion_pose);
        let matrix_left = pose::left_update(&delta, &matrix_pose);
        let right_as_matrix: Pose = quaternion_right.into();
        let left_as_matrix: Pose = quaternion_left.into();
        assert!(frob_diff(right_as_matrix.rotation(), matrix_right.rotation()) < 1e-14);
        assert!(l2_diff(right_as_matrix.translation(), matrix_right.translation()) < 1e-14);
        assert!(frob_diff(left_as_matrix.rotation(), matrix_left.rotation()) < 1e-14);
        assert!(l2_diff(left_as_matrix.translation(), matrix_left.translation()) < 1e-14);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn pose_json_round_trip_validates_rotation() {
        // JSON's decimal float format loses up to ~1 ULP per scalar; the
        // round-trip is numerically equivalent rather than bit-identical.
        let pose = Pose::exp(&[0.2, -0.1, 0.3, 1.0, -2.0, 0.5]);
        let json = serde_json::to_string(&pose).unwrap();
        let recovered: Pose = serde_json::from_str(&json).unwrap();
        assert!(frob_diff(recovered.rotation(), pose.rotation()) < 1e-14);
        assert!(l2_diff(recovered.translation(), pose.translation()) < 1e-14);

        let bad = r#"{"rotation":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.5]],"translation":[0.0,0.0,0.0]}"#;
        let err = serde_json::from_str::<Pose>(bad).unwrap_err();
        assert!(err.to_string().contains("orthogonal") || err.to_string().contains("rotation"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn extended_pose_json_round_trip_validates_rotation() {
        let g = ExtendedPose::exp(&[0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 1.0, -2.0, 0.5]);
        let json = serde_json::to_string(&g).unwrap();
        let recovered: ExtendedPose = serde_json::from_str(&json).unwrap();
        assert!(frob_diff(recovered.rotation(), g.rotation()) < 1e-14);
        assert!(l2_diff(recovered.velocity(), g.velocity()) < 1e-14);
        assert!(l2_diff(recovered.position(), g.position()) < 1e-14);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn quat_pose_json_round_trip_validates_unit_norm() {
        let q = QuatPose::exp(&[0.2, -0.1, 0.3, 1.0, -2.0, 0.5]);
        let json = serde_json::to_string(&q).unwrap();
        let recovered: QuatPose = serde_json::from_str(&json).unwrap();
        assert!((recovered.q0() - q.q0()).abs() < 1e-14);
        assert!(l2_diff(recovered.qv(), q.qv()) < 1e-14);
        assert!(l2_diff(recovered.translation(), q.translation()) < 1e-14);

        let bad = r#"{"q0":2.0,"qv":[0.0,0.0,0.0],"translation":[0.0,0.0,0.0]}"#;
        let err = serde_json::from_str::<QuatPose>(bad).unwrap_err();
        assert!(err.to_string().contains("unit-norm") || err.to_string().contains("quaternion"));
    }

    #[test]
    fn right_point_action_jacobian_matches_finite_difference() {
        let pose = Pose::exp(&[0.2, -0.1, 0.3, 1.0, 0.2, -0.4]);
        let point = [0.7, -0.5, 2.0];
        let jacobian = pose::right_point_action_jacobian(&pose, &point);
        let base = pose.act(&point);
        let h = 1e-7;
        for column in 0..6 {
            let mut delta = [0.0; 6];
            delta[column] = h;
            let moved = pose::right_update(&pose, &delta).act(&point);
            for row in 0..3 {
                assert!(((moved[row] - base[row]) / h - jacobian[row][column]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn quat_pose_right_point_action_jacobian_matches_matrix_pose() {
        use super::quaternion_pose;
        let twist = [0.2, -0.1, 0.3, 1.0, 0.2, -0.4];
        let point = [0.7, -0.5, 2.0];
        let p = Pose::exp(&twist);
        let q = QuatPose::exp(&twist);
        let jac_pose = pose::right_point_action_jacobian(&p, &point);
        let jac_quat = quaternion_pose::right_point_action_jacobian(&q, &point);
        for row in 0..3 {
            for column in 0..6 {
                assert!((jac_pose[row][column] - jac_quat[row][column]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn extended_pose_position_channel_jacobian_matches_finite_difference() {
        let xi = [0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 1.0, -2.0, 0.5];
        let g = ExtendedPose::exp(&xi);
        let point = [0.7, -0.5, 2.0];
        let jacobian = extended_pose::right_point_action_position_jacobian(&g, &point);
        let base = g.act_position(&point);
        let h = 1e-7;
        for column in 0..9 {
            let mut delta = [0.0; 9];
            delta[column] = h;
            let moved = extended_pose::right_update(&g, &delta).act_position(&point);
            for row in 0..3 {
                assert!(((moved[row] - base[row]) / h - jacobian[row][column]).abs() < 1e-6);
            }
        }
        // Velocity columns (3..6) are zero in the position channel.
        for row in 0..3 {
            for column in 3..6 {
                assert_eq!(jacobian[row][column], 0.0);
            }
        }
    }

    #[test]
    fn extended_pose_velocity_channel_jacobian_matches_finite_difference() {
        let xi = [0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 1.0, -2.0, 0.5];
        let g = ExtendedPose::exp(&xi);
        let point = [0.7, -0.5, 2.0];
        let jacobian = extended_pose::right_point_action_velocity_jacobian(&g, &point);
        let base = g.act_velocity(&point);
        let h = 1e-7;
        for column in 0..9 {
            let mut delta = [0.0; 9];
            delta[column] = h;
            let moved = extended_pose::right_update(&g, &delta).act_velocity(&point);
            for row in 0..3 {
                assert!(((moved[row] - base[row]) / h - jacobian[row][column]).abs() < 1e-6);
            }
        }
        // Position columns (6..9) are zero in the velocity channel.
        for row in 0..3 {
            for column in 6..9 {
                assert_eq!(jacobian[row][column], 0.0);
            }
        }
    }
}
