# se3-ad-recipes

[![Rust CI](https://github.com/sigmapointlabs/se3-ad-recipes/actions/workflows/rust-tests.yml/badge.svg)](https://github.com/sigmapointlabs/se3-ad-recipes/actions/workflows/rust-tests.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust edition: 2024](https://img.shields.io/badge/rust-2024-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/index.html)
[![Release](https://img.shields.io/github/v/release/sigmapointlabs/se3-ad-recipes)](https://github.com/sigmapointlabs/se3-ad-recipes/releases)

Companion code for the arXiv preprint *"Exact Higher-Order Derivatives for SE(3) via Analytical/AD Methods."*

This repository implements eight ways to compute the same 6×6 SE(3) negative-log-likelihood Hessian and reproduces the benchmark table from the paper.

## Quickstart

```rust
use se3_ad_recipes::prelude::*;

// Build an SE(3) pose from a rotation-first twist [ω; t].
let pose = Pose::exp(&[0.0, 0.0, 0.1, 1.0, 0.0, 0.0]);

// Apply it to a 3D point.
let world = pose.act(&[0.5, 0.0, 0.0]);

// Right-perturb in the tangent space (common in NLS solvers).
let updated = se3_right_update(&pose, &[0.0, 0.0, 0.01, 0.0, 0.0, 0.0]);

// 3x6 Jacobian for the point-action residual r = pose.act(p) - y.
let jac = right_point_action_jacobian(&pose, &[0.5, 0.0, 0.0]);
```

`Pose` is opaque — invariants `Rᵀ R = I`, `det R = +1` cannot be violated by the API. Use `Pose::from_rotation_translation(&R, &t)` to ingest external data with validation, or `Pose::from_parts_unchecked` when you have an independent guarantee. SE_2(3) (`ExtendedPose`, extended pose group for inertial navigation) and quaternion-storage SE(3) (`QuatPose`, useful near the antipodal locus) follow the same shape. AD-generic code uses the `expert` tier — see the `api` module rustdoc for the full stability story.

## Tour

### Building a pose

`Pose::exp` is the usual path — pass a rotation-first twist `[ω; t]`:

```rust
use se3_ad_recipes::prelude::*;

let pose = Pose::exp(&[0.0, 0.0, 0.5, 1.0, 2.0, -0.5]);
```

Ingesting external `(R, t)` data validates orthogonality and `det R = +1` (and rejects non-finite components):

```rust
use se3_ad_recipes::prelude::*;

let r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
let pose = Pose::from_rotation_translation(&r, &[1.0, 0.0, 0.0]).unwrap();
```

When you have an independent guarantee (e.g. a rotation just produced by another group op), `Pose::from_parts_unchecked` skips the check. Both `Pose::identity` and `Pose::default` are `const fn`, so they work in `const` contexts.

### Group operations

`compose`, `inverse`, `act`, `act_inverse` are methods on `Pose`. They never observe the underlying `(R, t)` representation:

```rust
use se3_ad_recipes::prelude::*;

let a = Pose::exp(&[0.0, 0.0, 0.1, 1.0, 0.0, 0.0]);
let b = Pose::exp(&[0.0, 0.0, 0.2, 0.0, 1.0, 0.0]);

let ab = a.compose(&b);
let a_inv = a.inverse();

let p_world = ab.act(&[0.5, 0.0, 0.0]);
let p_back = ab.act_inverse(&p_world);  // round-trip
```

### Tangent space and perturbation updates

`Pose::log` returns the rotation-first twist `[ω; t]`. Free functions `se3_right_update` and `se3_left_update` apply `Exp(δ)` on the right or left — the canonical update in nonlinear least-squares solvers:

```rust
use se3_ad_recipes::prelude::*;

let pose = Pose::exp(&[0.0, 0.0, 0.5, 1.0, 2.0, -0.5]);

// Right perturbation: pose * Exp(δ)
let delta: Twist = [0.0, 0.0, 0.01, 0.0, 0.0, 0.0];
let updated = se3_right_update(&pose, &delta);

// Recover the underlying twist
let twist_back = updated.log();
```

### Jacobians for nonlinear least squares

The point-action Jacobian is the right-perturbation derivative of `pose.act(p)` evaluated at `δ = 0` — a 3×6 matrix `[ -R[p]× | R ]`:

```rust
use se3_ad_recipes::prelude::*;

let pose = Pose::exp(&[0.2, -0.1, 0.3, 1.0, 0.2, -0.4]);
let landmark = [0.7, -0.5, 2.0];

let jac: Mat3x6 = right_point_action_jacobian(&pose, &landmark);

// Build a residual row r = pose.act(p) - measurement; pair it with `jac` as the
// Jacobian row in your Gauss-Newton / Levenberg-Marquardt linear system.
let r: Point3 = pose.act(&landmark);
```

For Hessian-aware work, `api::expert::se3::right_jacobian_derivative` returns the full `[Mat6G<T>; 6]` derivative tensor; the directional-derivative variant returns one 6×6 slice for Hessian-vector products.

### Inertial navigation: `ExtendedPose` (SE_2(3))

`ExtendedPose` extends SE(3) with a velocity channel. The tangent is `[ω; ν; ρ]` (rotation, velocity, position). Two natural point actions — `act_position` transforms a 3D position by `(R, p)`, `act_velocity` transforms a 3D velocity by `(R, v)`:

```rust
use se3_ad_recipes::prelude::*;

// [ω; ν; ρ]
let state = ExtendedPose::exp(&[0.1, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0]);

let x = [1.0, 0.0, 0.0];
let world_point = state.act_position(&x);
let world_velocity = state.act_velocity(&x);
```

Channel-specific Jacobians come from `se23_right_point_action_position_jacobian` and `se23_right_point_action_velocity_jacobian`.

### Quaternion-storage SE(3): `QuatPose`

`QuatPose` carries the same SE(3) group element as `Pose` but stores `(q₀, q_v, t)` in Hamilton convention. Useful near the antipodal locus (where the matrix log is awkward) and as a compact 7-scalar representation. Conversion via `From` materializes the alternate representation — Shepperd's algorithm going `Pose → QuatPose`, the quadratic identity going back — and copies the storage:

```rust
use se3_ad_recipes::prelude::*;

let pose = Pose::exp(&[0.2, -0.1, 0.3, 1.0, -2.0, 0.5]);

// Switch representations without going through exp / log.
let quat: QuatPose = (&pose).into();
let back: Pose = (&quat).into();

// Same group element either way.
assert!((back.translation()[0] - pose.translation()[0]).abs() < 1e-14);
```

### AD-generic code via `api::expert`

The application tier above is `f64`-only by design. For forward, reverse, or nested-dual AD, use `api::expert`. All Lie-group operations there are generic over the `AD` scalar trait:

```rust
use se3_ad_recipes::api::expert::se3;

// Compute the SE(3) right Jacobian and its directional derivative.
let xi = [0.1, -0.2, 0.05, 1.0, -0.5, 0.3];
let direction = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
let jr = se3::right_jacobian::<f64>(&xi);
let djr_dir = se3::right_jacobian_directional_derivative::<f64>(&xi, &direction);
```

To run the same code under AD, swap `f64` for `expert::ad::adfn<N>` (first-order forward), `expert::ad::D2<N>` (Hessian-ready), `expert::ad::D3<N>` (third-order), or `expert::ad::adr_n6` (forward-over-reverse with the `tape_n6_*` ops). The const-generic `expert::linalg::{matmul, matvec, transpose}` and the block assemblers `block_2x2` / `block_3x3` let you compose Lie-group Jacobians without dipping into raw modules.

### Optional `serde` integration

Enable the `serde` Cargo feature to get `Serialize` / `Deserialize` impls on all three pose types. The on-disk shape is `{rotation, translation}` / `{rotation, velocity, position}` / `{q0, qv, translation}` — opaque to the inner representation. Deserialize routes through the validated constructors, so on-disk data cannot produce an invalid pose:

```toml
[dependencies]
se3-ad-recipes = { version = "1", features = ["serde"] }
```

## The benchmark

The benchmark compares:

  1. Finite differences on the cost value.
  2. Finite differences on a forward-AD gradient.
  3. Forward-of-forward nested AD with a naïve θ-Taylor scalar basis
     (the §IV.A polynomial-depletion trap).
  4. Forward-of-forward nested AD with the production fused-scalar basis.
  5. Forward AD seeded into a hand-rolled analytical gradient that
     uses the *unfused* `D'(θ)·ω/θ` factor (the §IV.B singular-pair
     trap; produces NaN at depth 0).
  6. Finite differences on the analytical gradient.
  7. **Forward AD seeded into the analytical gradient with fused
     scalars — "the recipe".**
  8. Automatic forward-over-reverse via a custom `adr_n6` reverse tape
     whose partials are `adfn<6>` (no analytical gradient required).

Default features have zero runtime dependencies; the optional `serde` feature pulls in `serde` as a runtime dependency. `criterion` and `serde_json` are dev-deps.

## Table I

The block below is auto-verified by
[`nll_tests::readme_table_i_is_up_to_date`] on every `cargo test`.
The accuracy column is deterministic to the printed precision; LOC
counts user-written analytical lines, **excluding the AD scalar type
implementation itself**. The **Time** column is informational only —
it is machine-dependent and is not gated by the test (the verification
strips it before comparison); refresh it explicitly with the
`UPDATE_README=1` command shown below, ideally under `--release`.

[`nll_tests::readme_table_i_is_up_to_date`]: src/nll_tests.rs

<!-- HESSIAN_TABLE_START -->
| # | Method | LOC | Rel. err. vs. oracle | Time |
|---|---|---|---|---|
| 1 | FD of value (no AD) | 43 | 6.65e-3 | 337 μs |
| 2 | FD of AD-gradient (baseline) | 30 | 9.18e-7 | 752 μs |
| 6 | FD of analytical gradient (fused basis) | 104 | 9.18e-7 | 154 μs |
| 3 | Nested AD, naïve basis | 118 | 3.34e-18 | 956 μs |
| 4 | Nested AD, fused basis (oracle) | 30 | 0 (reference) | 995 μs |
| 5 | Seeded AD of analytical gradient, naïve basis | (102) | NaN (depth-0 §IV.B trap) | — |
| 7 | Seeded AD of analytical gradient, fused basis (recipe) | 102 | 1.30e-16 | 168 μs |
| 8 | Auto FoR (`UnsafeCell` tape, no analytical grad) | 30 | 2.54e-16 | 412 μs |
<!-- HESSIAN_TABLE_END -->

To populate or refresh after a code change:

```bash
UPDATE_README=1 cargo test --release --lib readme_table_i_is_up_to_date
```

## Bench

```bash
cargo bench --features bench-support --bench nll_hessian -- --quick  # ~90 s
cargo bench --features bench-support --bench nll_hessian             # full, ~3 min
```

## Public API stability

The `api` module exposes three nested tiers. The **application tier**
(`api::pose`, `api::extended_pose`, `api::quaternion_pose`, `api::rotation`,
`api::types`) is SemVer-stable — breaking changes require a major release,
and the intent is to keep those exceptionally rare. The **expert tier**
(`api::expert::{types, linalg, so3, se3, se23, quat_se3, projective, ad}`)
exposes AD-generic operations under clean, dimension-agnostic names and
follows standard SemVer (additions in minor releases; renames or removals
require a major release). The **raw modules** (`so3_adsafe`, `se3_adsafe`,
...) are kept `pub` for paper reproducibility with no stability promise.
See the `api` module's rustdoc for the full contract. Benchmark internals
require the `bench-support` feature.

## License

Dual licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
