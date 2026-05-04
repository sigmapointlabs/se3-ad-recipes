//! Core AD trait and f64 implementation.
//!
//! Defines the minimal `AD` trait required by this crate's generic
//! differentiation routines. Replaces the external `ad_trait` crate.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Trait for scalar types that support automatic differentiation.
///
/// Implemented for `f64` (plain evaluation), `adfn<N>` (forward-mode),
/// and `adr` (reverse-mode tape).
pub trait AD:
    Copy
    + Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + 'static
{
    /// Create a constant (derivative-free) value.
    fn constant(v: f64) -> Self;

    /// Extract the primal (f64) value, discarding any derivative info.
    fn to_constant(&self) -> f64;

    /// Convert this value to another AD type (preserving only the primal).
    fn to_other_ad_type<T2: AD>(&self) -> T2 {
        T2::constant(self.to_constant())
    }

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn sqrt(self) -> Self;
    fn ln(self) -> Self;
    fn abs(self) -> Self;
    fn acos(self) -> Self;
    fn atan2(self, x: Self) -> Self;
}

// ═══════════════════════════════════════════════════════════════════════════
// f64 implementation
// ═══════════════════════════════════════════════════════════════════════════

impl AD for f64 {
    #[inline]
    fn constant(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_constant(&self) -> f64 {
        *self
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        f64::atan2(self, x)
    }
}
