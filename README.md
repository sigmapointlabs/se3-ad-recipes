# se3-ad-recipes

[![Rust CI](https://github.com/sigmapointlabs/se3-ad-recipes/actions/workflows/rust-tests.yml/badge.svg)](https://github.com/sigmapointlabs/se3-ad-recipes/actions/workflows/rust-tests.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust edition: 2024](https://img.shields.io/badge/rust-2024-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/index.html)
[![Release](https://img.shields.io/github/v/release/sigmapointlabs/se3-ad-recipes)](https://github.com/sigmapointlabs/se3-ad-recipes/releases)

Companion code for the arXiv preprint *"Exact Higher-Order Derivatives for SE(3) via Analytical/AD Methods."*

This repository implements eight ways to compute the same 6×6 SE(3) negative-log-likelihood Hessian and reproduces the benchmark table from the paper.

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

Zero runtime dependencies; criterion is the only dev-dep.

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
| 1 | FD of value (no AD) | 43 | 6.65e-3 | 344 μs |
| 2 | FD of AD-gradient (baseline) | 30 | 9.18e-7 | 742 μs |
| 6 | FD of analytical gradient (fused basis) | 104 | 9.18e-7 | 158 μs |
| 3 | Nested AD, naïve basis | 118 | 3.34e-18 | 981 μs |
| 4 | Nested AD, fused basis (oracle) | 30 | 0 (reference) | 1.02 ms |
| 5 | Seeded AD of analytical gradient, naïve basis | (102) | NaN (depth-0 §IV.B trap) | — |
| 7 | Seeded AD of analytical gradient, fused basis (recipe) | 102 | 1.30e-16 | 166 μs |
| 8 | Auto FoR (`UnsafeCell` tape, no analytical grad) | 30 | 2.54e-16 | 410 μs |
<!-- HESSIAN_TABLE_END -->

To populate or refresh after a code change:

```bash
UPDATE_README=1 cargo test --release --lib readme_table_i_is_up_to_date
```

## Bench

```bash
cargo bench --bench nll_hessian -- --quick      # ~90 s
cargo bench --bench nll_hessian                 # full criterion run, ~3 min
```

## License

Dual licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
