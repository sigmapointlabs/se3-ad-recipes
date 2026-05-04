//! Nestable forward-mode AD with compile-time tangent dimension.
//!
//! `Dual<T, N>` carries a primal value of type `T` and an N-dimensional
//! tangent vector of type `T`. When `T = f64`, this is equivalent to `adfn<N>`.
//! When `T = Dual<f64, M>`, this gives second-order derivatives (Hessian).
//! When `T = Dual<Dual<f64, M>, M>`, third-order (cubic). And so on.
//!
//! This eliminates all finite differences from the derivative pipeline:
//! - Depth 1: `Dual<f64, 6>` → exact Jacobian (= `adfn<6>`)
//! - Depth 2: `Dual<Dual<f64, 6>, 6>` → exact Hessian
//! - Depth 3: `Dual<Dual<Dual<f64, 6>, 6>, 6>` → exact cubic

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::ad_trait::AD;

/// Nestable forward-mode AD scalar.
///
/// `T` is the inner scalar type (f64, or another Dual for higher orders).
/// `N` is the number of tangent directions at this level.
#[derive(Clone, Copy)]
pub struct Dual<T: AD, const N: usize> {
    pub value: T,
    pub tangent: [T; N],
}

impl<T: AD, const N: usize> Dual<T, N> {
    /// Create with explicit value and tangent.
    #[inline]
    pub fn new(value: T, tangent: [T; N]) -> Self {
        Self { value, tangent }
    }

    /// Create a constant (zero tangent).
    #[inline]
    pub fn new_constant(value: T) -> Self {
        Self {
            value,
            tangent: [T::constant(0.0); N],
        }
    }

    /// Seed: create from f64 with tangent direction i set to 1.
    #[inline]
    pub fn seed(value: T, direction: usize) -> Self {
        let mut tangent = [T::constant(0.0); N];
        tangent[direction] = T::constant(1.0);
        Self { value, tangent }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AD trait implementation
// ═══════════════════════════════════════════════════════════════════════════

impl<T: AD, const N: usize> AD for Dual<T, N> {
    #[inline]
    fn constant(v: f64) -> Self {
        Self::new_constant(T::constant(v))
    }

    #[inline]
    fn to_constant(&self) -> f64 {
        self.value.to_constant()
    }

    #[inline]
    fn sin(self) -> Self {
        let c = self.value.cos();
        let mut t = [T::constant(0.0); N];
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
        let neg_s = T::constant(0.0) - self.value.sin();
        let mut t = [T::constant(0.0); N];
        for i in 0..N {
            t[i] = neg_s * self.tangent[i];
        }
        Self {
            value: self.value.cos(),
            tangent: t,
        }
    }

    #[inline]
    fn sqrt(self) -> Self {
        let sv = self.value.sqrt();
        let half = T::constant(0.5);
        let d = if sv.to_constant().abs() > 0.0 {
            half / sv
        } else {
            T::constant(0.0)
        };
        let mut t = [T::constant(0.0); N];
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
        let d = T::constant(1.0) / self.value;
        let mut t = [T::constant(0.0); N];
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
        let d = if self.value.to_constant() >= 0.0 {
            T::constant(1.0)
        } else {
            T::constant(-1.0)
        };
        let mut t = [T::constant(0.0); N];
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
        let one = T::constant(1.0);
        let d = T::constant(0.0) - one / (one - self.value * self.value).sqrt();
        let mut t = [T::constant(0.0); N];
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
        let denom = x.value * x.value + self.value * self.value;
        let dy = x.value / denom;
        let dx = (T::constant(0.0) - self.value) / denom;
        let mut t = [T::constant(0.0); N];
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

impl<T: AD, const N: usize> Add for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut t = [T::constant(0.0); N];
        for i in 0..N {
            t[i] = self.tangent[i] + rhs.tangent[i];
        }
        Self {
            value: self.value + rhs.value,
            tangent: t,
        }
    }
}

impl<T: AD, const N: usize> Sub for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut t = [T::constant(0.0); N];
        for i in 0..N {
            t[i] = self.tangent[i] - rhs.tangent[i];
        }
        Self {
            value: self.value - rhs.value,
            tangent: t,
        }
    }
}

impl<T: AD, const N: usize> Mul for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut t = [T::constant(0.0); N];
        for i in 0..N {
            t[i] = self.tangent[i] * rhs.value + self.value * rhs.tangent[i];
        }
        Self {
            value: self.value * rhs.value,
            tangent: t,
        }
    }
}

impl<T: AD, const N: usize> Div for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv_b2 = T::constant(1.0) / (rhs.value * rhs.value);
        let mut t = [T::constant(0.0); N];
        for i in 0..N {
            t[i] = (self.tangent[i] * rhs.value - self.value * rhs.tangent[i]) * inv_b2;
        }
        Self {
            value: self.value / rhs.value,
            tangent: t,
        }
    }
}

impl<T: AD, const N: usize> Neg for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut t = [T::constant(0.0); N];
        for i in 0..N {
            t[i] = T::constant(0.0) - self.tangent[i];
        }
        Self {
            value: T::constant(0.0) - self.value,
            tangent: t,
        }
    }
}

impl<T: AD, const N: usize> AddAssign for Dual<T, N> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<T: AD, const N: usize> SubAssign for Dual<T, N> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<T: AD, const N: usize> MulAssign for Dual<T, N> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<T: AD, const N: usize> DivAssign for Dual<T, N> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: AD, const N: usize> fmt::Debug for Dual<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dual({:.6})", self.value.to_constant())
    }
}

impl<T: AD, const N: usize> PartialEq for Dual<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.value.to_constant() == other.value.to_constant()
    }
}

impl<T: AD, const N: usize> PartialOrd for Dual<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value
            .to_constant()
            .partial_cmp(&other.value.to_constant())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Type aliases for common nesting depths
// ═══════════════════════════════════════════════════════════════════════════

/// First-order: exact Jacobian (equivalent to `adfn<N>`).
pub type D1<const N: usize> = Dual<f64, N>;
/// Second-order: exact Hessian. Zero FD.
pub type D2<const N: usize> = Dual<Dual<f64, N>, N>;
/// Third-order: exact cubic. Zero FD.
pub type D3<const N: usize> = Dual<Dual<Dual<f64, N>, N>, N>;
