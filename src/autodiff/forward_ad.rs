//! Forward-mode automatic differentiation with compile-time tangent dimension.
//!
//! `adfn<N>` carries a primal value and an N-dimensional tangent vector.
//! Arithmetic and transcendental operations propagate tangents via the
//! chain rule, giving exact first-order derivatives.

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::ad_trait::AD;

/// Forward-mode AD scalar with `N` tangent components.
///
/// Used as `adfn<6>` in this crate for SE(3) Lie-algebra perturbations.
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct adfn<const N: usize> {
    value: f64,
    tangent: [f64; N],
}

impl<const N: usize> adfn<N> {
    /// Create a new AD variable with given value and tangent.
    #[inline]
    pub fn new(value: f64, tangent: [f64; N]) -> Self {
        Self { value, tangent }
    }

    /// Create a constant (zero tangent).
    #[inline]
    pub fn new_constant(value: f64) -> Self {
        Self {
            value,
            tangent: [0.0; N],
        }
    }

    /// Access the tangent vector.
    #[inline]
    pub fn tangent(&self) -> [f64; N] {
        self.tangent
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AD trait implementation
// ═══════════════════════════════════════════════════════════════════════════

impl<const N: usize> AD for adfn<N> {
    #[inline]
    fn constant(v: f64) -> Self {
        Self::new_constant(v)
    }

    #[inline]
    fn to_constant(&self) -> f64 {
        self.value
    }

    #[inline]
    fn sin(self) -> Self {
        // d/dx sin(x) = cos(x)
        let c = self.value.cos();
        let mut t = [0.0; N];
        for i in 0..N {
            t[i] = c * self.tangent[i];
        }
        Self {
            value: self.value.sin(),
            tangent: t,
        }
    }

    #[inline]
    fn cos(self) -> Self {
        // d/dx cos(x) = -sin(x)
        let s = -self.value.sin();
        let mut t = [0.0; N];
        for i in 0..N {
            t[i] = s * self.tangent[i];
        }
        Self {
            value: self.value.cos(),
            tangent: t,
        }
    }

    #[inline]
    fn sqrt(self) -> Self {
        // d/dx sqrt(x) = 1/(2·sqrt(x))
        let sv = self.value.sqrt();
        let d = if sv.abs() > 0.0 { 0.5 / sv } else { 0.0 };
        let mut t = [0.0; N];
        for i in 0..N {
            t[i] = d * self.tangent[i];
        }
        Self {
            value: sv,
            tangent: t,
        }
    }

    #[inline]
    fn ln(self) -> Self {
        // d/dx ln(x) = 1/x
        let d = 1.0 / self.value;
        let mut t = [0.0; N];
        for i in 0..N {
            t[i] = d * self.tangent[i];
        }
        Self {
            value: self.value.ln(),
            tangent: t,
        }
    }

    #[inline]
    fn abs(self) -> Self {
        let d = if self.value >= 0.0 { 1.0 } else { -1.0 };
        let mut t = [0.0; N];
        for i in 0..N {
            t[i] = d * self.tangent[i];
        }
        Self {
            value: self.value.abs(),
            tangent: t,
        }
    }

    #[inline]
    fn acos(self) -> Self {
        // d/dx acos(x) = -1/sqrt(1 - x²)
        let d = -1.0 / (1.0 - self.value * self.value).sqrt();
        let mut t = [0.0; N];
        for i in 0..N {
            t[i] = d * self.tangent[i];
        }
        Self {
            value: self.value.acos(),
            tangent: t,
        }
    }

    #[inline]
    fn atan2(self, x: Self) -> Self {
        // d/dy atan2(y,x) = x/(x²+y²),  d/dx atan2(y,x) = -y/(x²+y²)
        let denom = x.value * x.value + self.value * self.value;
        let dy = x.value / denom;
        let dx = -self.value / denom;
        let mut t = [0.0; N];
        for i in 0..N {
            t[i] = dy * self.tangent[i] + dx * x.tangent[i];
        }
        Self {
            value: self.value.atan2(x.value),
            tangent: t,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic operators
// ═══════════════════════════════════════════════════════════════════════════

impl<const N: usize> Add for adfn<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut t = [0.0; N];
        for i in 0..N {
            t[i] = self.tangent[i] + rhs.tangent[i];
        }
        Self {
            value: self.value + rhs.value,
            tangent: t,
        }
    }
}

impl<const N: usize> Sub for adfn<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut t = [0.0; N];
        for i in 0..N {
            t[i] = self.tangent[i] - rhs.tangent[i];
        }
        Self {
            value: self.value - rhs.value,
            tangent: t,
        }
    }
}

impl<const N: usize> Mul for adfn<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // d(a·b) = da·b + a·db
        let mut t = [0.0; N];
        for i in 0..N {
            t[i] = self.tangent[i] * rhs.value + self.value * rhs.tangent[i];
        }
        Self {
            value: self.value * rhs.value,
            tangent: t,
        }
    }
}

impl<const N: usize> Div for adfn<N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // d(a/b) = (da·b - a·db) / b²
        let inv_b2 = 1.0 / (rhs.value * rhs.value);
        let mut t = [0.0; N];
        for i in 0..N {
            t[i] = (self.tangent[i] * rhs.value - self.value * rhs.tangent[i]) * inv_b2;
        }
        Self {
            value: self.value / rhs.value,
            tangent: t,
        }
    }
}

impl<const N: usize> Neg for adfn<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut t = [0.0; N];
        for i in 0..N {
            t[i] = -self.tangent[i];
        }
        Self {
            value: -self.value,
            tangent: t,
        }
    }
}

impl<const N: usize> AddAssign for adfn<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize> SubAssign for adfn<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize> MulAssign for adfn<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize> DivAssign for adfn<N> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Debug, PartialEq, PartialOrd (compare on primal value only)
// ═══════════════════════════════════════════════════════════════════════════

impl<const N: usize> fmt::Debug for adfn<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "adfn({:.6})", self.value)
    }
}

impl<const N: usize> PartialEq for adfn<N> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<const N: usize> PartialOrd for adfn<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}
