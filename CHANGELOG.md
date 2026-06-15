# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-06-15

First semver-stable release. Introduces the curated application API
(`api`, `prelude`) and its AD-generic counterpart (`api::expert`), and
recasts the crate's public surface around opaque group types. The raw
paper-aligned modules remain `pub` for source compatibility but are
hidden from generated documentation.

**Stability commitment.** Both the application tier and the expert tier
are SemVer-stable: breaking changes require a major release, additions
ship in minor releases. The application tier's contract is the stricter
of the two — the intent is to keep breaking changes there exceptionally
rare and well-motivated. Raw modules carry no stability promise and
serve only as an escape hatch.

### Added

#### Curated application API (`f64`-only, opaque)

- `api::pose::Pose` — opaque SE(3) pose wrapping `(R, t)`. Constructors
  `Pose::identity` (`const fn`), `Pose::exp`, `Pose::from_rotation_translation`
  (validated against orthogonality + `det R = +1`),
  `Pose::from_rotation_translation_with_tolerance`, `Pose::from_parts_unchecked`.
  Inherent methods `log`, `compose`, `inverse`, `act`, `act_inverse`,
  `adjoint`, `rotation`, `translation`. Free functions
  `pose::right_jacobian` / `..._inverse`, `pose::left_jacobian` / `..._inverse`,
  `pose::right_update` / `pose::left_update`, `pose::right_point_action_jacobian`.
  Derives `Clone`, `Copy`, `Debug`, `Default` (= identity), `PartialEq`.
- `api::extended_pose::ExtendedPose` — opaque SE_2(3) state `(R, v, p)`
  for inertial navigation. Mirrors `Pose` plus channel-specific actions
  `act_position` / `act_velocity` (+ inverses), channel-specific
  point-action Jacobians, `log_quaternion` for antipodal-safe SO(3) log,
  `adjoint` and `adjoint_inverse` methods.
- `api::quaternion_pose::QuatPose` — opaque quaternion-storage SE(3) pose,
  useful near the antipodal locus and as a compact 7-scalar
  representation. Bidirectional `From<&Pose>` / `From<&QuatPose>` bridges.
  `adjoint` method delegates through `to_pose`.
- `api::rotation` — SO(3) operations (`exp`, `log`, `hat`, `vee`,
  `right_jacobian` / `..._inverse`, `left_jacobian` / `..._inverse`,
  `right_update` / `left_update`).
- `api::types` — semantic aliases (`Twist`, `Point3`) plus row-major
  `Mat3` / `Mat6` / `Mat9` / `Mat3x6` / `Mat3x9` / `Vec3` / `Vec6` / `Vec9`.
- `api::InputError` — shared input-shape errors (`NonFiniteInput`,
  `InvalidTolerance`) wrapped by `PoseError::Input` and `QuatPoseError::Input`.
- All three group error enums (`PoseError`, `QuatPoseError`, `InputError`)
  are `#[non_exhaustive]`, so future variants ship in minor releases.

#### Expert tier (`api::expert`, AD-generic)

- `api::expert::types` — `Mat3G<T>` / `Mat6G<T>` / `Mat9G<T>` / `Vec3G<T>`
  / `Vec6G<T>` / `Vec9G<T>` first-class aliases, plus the semantic
  tensor aliases `JacobianDerivative6 = [Mat6G<T>; 6]` and
  `Hessian6 = [[[T; 6]; 6]; 6]` so function signatures document the
  meaning, not just the shape.
- `api::expert::linalg` — const-generic `matmul`, `matvec`, `transpose`
  and the dim-specific block assemblers `block_2x2` (4 × 3×3 blocks →
  `Mat6G<T>`) / `block_3x3` (9 × 3×3 blocks → `Mat9G<T>`). Clean
  dimension-agnostic names; no `_g` or `_6x6` suffix.
- `api::expert::so3` — `exp`, `log`, `log_quaternion`, `hat`, `hat_basis`,
  `right_jacobian` / `..._inverse`, `v_matrix`, `v_inverse`,
  `mat3_to_quaternion`. The per-paper scalar basis atoms remain in the
  raw `so3_adsafe` module for advanced users writing their own analytic
  derivatives but are not part of the stable expert surface.
- `api::expert::se3` — clean-named operations plus the new high-level
  derivative entry points: `right_jacobian_derivative` (returns the full
  `JacobianDerivative6<T>` tensor) and
  `right_jacobian_directional_derivative` (one contracted 6×6 slice,
  useful for Hessian-vector products). Companion
  `right_jacobian_inverse_derivative` and
  `right_jacobian_inverse_directional_derivative`. `adjoint` takes a
  `&PoseG<T>` pose value rather than separated `(rotation, translation)`.
- `api::expert::se23` — SE_2(3) operations with the same clean naming:
  `right_jacobian`, `right_jacobian_inverse`, `adjoint`,
  `adjoint_inverse`. The adjoint variants take an `&ExtendedPoseG<T>`
  pose value.
- `api::expert::quat_se3` — `PoseQ`, `to_rotation_matrix` (was
  `quat_to_rotmat`), `rotate_vector` (was `quat_rotate_vec`).
- `api::expert::projective` — geometric layer: `project`,
  `project_jacobian`, `project_hessian`, `transform_point`, `j_cross`.
- `api::expert::projective::statistics` — statistical derivatives,
  currently `f64` only: `calibrate` (was `apply_calibration`),
  `reprojection_error`, `neg_log_likelihood`, `information_matrix` (was
  `measurement_info_matrix`), `third_cumulants`, `quartic_correction`
  (was `quartic_contraction_analytical`). Not part of the default
  prelude.
- `api::expert::ad` — documented allow-list of AD backends: `AD` trait,
  `adfn<N>` (first-order forward), `Dual` / `D2` / `D3` (nested forward
  for Hessians and third-order tensors), `adr_n6` plus
  `tape_n6_clear` / `tape_n6_backward` / `tape_n6_backward_into`
  (forward-over-reverse).
- `api::expert::Act` — generic point-action trait.

#### Convenience prelude

- `se3_ad_recipes::prelude` exports the three opaque pose types, the
  error enums, the type aliases, and the common standalone operations
  (perturbation updates, point-action Jacobians) — sized to the
  most-frequent NLS-loop call sites without flooding the namespace.

#### Optional `serde` feature

- `serde` Cargo feature, off by default, adds `Serialize` + `Deserialize`
  impls on `Pose`, `ExtendedPose`, `QuatPose`. On-disk shape is
  `{rotation, translation}` / `{rotation, velocity, position}` /
  `{q0, qv, translation}` — opaque to the inner representations and
  stable across releases. Deserialize routes through the validated
  constructors, so on-disk data cannot produce an invalid pose.

#### Other

- `rust-version = "1.85"` declared in `Cargo.toml`, matching `edition = "2024"`.
- `Pose`, `ExtendedPose`, `QuatPose` marked `#[must_use]`; standalone
  value-returning functions (Jacobians, adjoints, log, act) marked
  `#[must_use]` explicitly.
- `Pose::identity`, `ExtendedPose::identity`, `QuatPose::identity` are
  `const fn`.
- `bench-support` Cargo feature now gates `nll_bench`, removing it from
  the default `pub` surface of library consumers. The criterion bench
  requires `--features bench-support`.

### Changed

- Raw paper-aligned modules (`so3_*`, `se3_*`, `se23_adsafe`,
  `se3_quat_adsafe`, `projective`, `jacobians_*`, `autodiff`, ...) marked
  `#[doc(hidden)]`. They remain `pub` for source compatibility but no
  longer appear in generated documentation.
- The application tier owns the canonical fixed-size type aliases;
  `crate::Mat3` / etc. are now `#[doc(hidden)]` re-exports from
  `api::types`.

### Removed

- No public-surface removals. The raw paper-aligned API is unchanged.
  Redundant method+free-function pairs introduced earlier in this
  release cycle (`pose::exp` + `Pose::exp` etc.) are absent from the
  released surface; the method form is canonical.

## [0.1.0]

### Added

- Initial release. Eight different ways to compute the SE(3) NLL Hessian
  (Table I), the AD framework (`adfn`, `Dual`, `adr_n6`), the AD-safe
  fused scalar basis described in the paper, and the SE(3) / SE_2(3) /
  quaternion-storage primitives that underpin the curated API.
