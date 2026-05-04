//! Automatic-differentiation primitives used by the SE(3) NLL Hessian bench.
//!
//! Four AD scalar types reachable from `nll_bench`:
//! - `ad_trait` — the `AD` trait every scalar type implements
//! - `forward_ad` — forward-mode AD with compile-time tangent dim N
//! - `nested_ad` — `Dual<T,N>` for forward-of-forward (Hessians via D2)
//! - `reverse_ad_n6` — reverse-mode tape with `adfn<6>` partials, the
//!   inner half of the automatic forward-over-reverse path

pub mod ad_trait;
pub mod forward_ad;
pub mod nested_ad;
pub mod reverse_ad_n6;
