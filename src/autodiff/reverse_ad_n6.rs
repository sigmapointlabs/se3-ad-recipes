//! Reverse-mode AD with `adfn<6>` partials — enables forward-over-reverse Hessians.
//!
//! Distinct from `reverse_ad.rs` (whose tape stores plain `f64` partials),
//! this module's tape stores **`adfn<6>` partials** so that the backward
//! sweep propagates an outer 6-direction forward tangent through every
//! recorded operation.  After the reverse pass each input's adjoint is an
//! `adfn<6>` whose `value` is the ordinary gradient component and whose
//! `tangent` is the corresponding **row of the Hessian**.
//!
//! That's automatic forward-over-reverse (FoR) at fixed `n = 6`:
//!
//! ```text
//!   forward through f, with x: adr_n6 carrying inner adfn<6>     →   one tape pass
//!   backward sweep with output adjoint = adfn<6>::constant(1)    →   one reverse pass
//!   read adjoint[input[i]].tangent() = row i of ∇²f                  total: O(n) per scalar op
//! ```
//!
//! For larger `n` you'd parameterise the partials over an arbitrary `T: AD`
//! (so `Dual<adr<T>, M>` covers any input dimension), at the cost of adding
//! generics to the thread-local tape.  At `n = 6` the hardcoded specialisation
//! is enough and keeps the implementation simple.

use std::cell::UnsafeCell;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::ad_trait::AD;
use super::forward_ad::adfn;

// ═══════════════════════════════════════════════════════════════════════════
// Tape infrastructure (partials are adfn<6>, so the backward sweep is FoR).
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
struct TapeEntryN6 {
    parents: [(usize, adfn<6>); 2],
    num_parents: u8,
}

// `UnsafeCell` instead of `RefCell` — the borrow-counter round-trip on every
// `tape_push` showed up as ~2 µs of pure ceremony in the FoR Hessian bench
// (≈ 870 ops × 2–3 ns of `borrow_mut` overhead).  The unchecked access is
// sound here under one invariant:
//
//   **No `&mut Vec<TapeEntryN6>` reference derived from this cell may live
//   across a call to `push`, `tape_n6_clear`, or `tape_n6_backward`.**
//
// Concretely each function below acquires `&mut *t.get()` for the duration of
// a single `len()` / `push` / `clear` / read-only loop and drops it before
// returning, so two simultaneous mutable references to the same `Vec` cannot
// be constructed by safe code.  All callers run on a single thread (the tape
// is `thread_local`), so cross-thread aliasing is impossible by construction.
//
// The 4096 initial capacity comfortably covers our SE(3) NLL high-water mark
// (≈ 870 entries per Hessian eval — measured directly), so steady-state calls
// never reallocate the tape itself.
thread_local! {
    static TAPE_N6: UnsafeCell<Vec<TapeEntryN6>> = UnsafeCell::new(Vec::with_capacity(4096));
}

#[inline]
fn push(entry: TapeEntryN6) -> usize {
    TAPE_N6.with(|t| {
        // SAFETY: single writer; the &mut borrow lives only for this block.
        let v = unsafe { &mut *t.get() };
        let idx = v.len();
        v.push(entry);
        idx
    })
}

/// Clear the tape — call at the start of each Hessian evaluation.
pub fn tape_n6_clear() {
    TAPE_N6.with(|t| {
        // SAFETY: single writer; the &mut borrow lives only for this call.
        let v = unsafe { &mut *t.get() };
        v.clear();
    });
}

/// Run the backward sweep from `output_index` into a caller-provided scratch
/// buffer — the allocation-free variant.
///
/// `adj` is resized to the current tape length and filled with zero-adjoints
/// before the sweep, then populated in-place.  Reusing a thread-local Vec
/// across calls eliminates the per-call ~6 KB malloc/free round-trip that
/// the owning [`tape_n6_backward`] otherwise pays.
///
/// Adjoints are `adfn<6>`-valued: each adjoint's `.value` is the f64 partial
/// gradient, and its `.tangent[k]` is the partial second derivative w.r.t.
/// the `k`-th seeded outer-forward direction.
pub fn tape_n6_backward_into(output_index: usize, adj: &mut Vec<adfn<6>>) {
    TAPE_N6.with(|t| {
        // SAFETY: read-only access for the duration of the sweep; we hold a
        // shared reference and never invoke anything that would re-enter and
        // request `&mut` to the same cell.
        let v = unsafe { &*t.get() };
        let n = v.len();
        adj.clear();
        adj.resize(n, adfn::<6>::new_constant(0.0));
        adj[output_index] = adfn::<6>::new_constant(1.0);
        for i in (0..n).rev() {
            let a = adj[i];
            // Skip cheap if the adjoint primal is exactly zero AND its tangent
            // is all-zero — short-circuiting on the f64 part alone would lose
            // legitimate Hessian contributions.
            if a.to_constant() == 0.0 && a.tangent().iter().all(|x| *x == 0.0) {
                continue;
            }
            let entry = v[i];
            for k in 0..entry.num_parents as usize {
                let (parent_idx, partial) = entry.parents[k];
                adj[parent_idx] += a * partial;
            }
        }
    })
}

/// Owning variant of [`tape_n6_backward_into`]: allocates a fresh adjoint Vec
/// each call.  Convenient for one-shot use; for repeated Hessian evaluations
/// prefer the `_into` variant with a thread-local scratch buffer.
pub fn tape_n6_backward(output_index: usize) -> Vec<adfn<6>> {
    let mut adj = Vec::new();
    tape_n6_backward_into(output_index, &mut adj);
    adj
}

// ═══════════════════════════════════════════════════════════════════════════
// `adr_n6`: reverse-mode AD scalar with adfn<6> partials.
// ═══════════════════════════════════════════════════════════════════════════

/// Reverse-mode AD scalar whose tape stores `adfn<6>` partials.
///
/// Each instance carries a primal of type `adfn<6>` (an outer 6-direction
/// forward tangent ride-along) and an index into the thread-local tape.
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct adr_n6 {
    value: adfn<6>,
    index: usize,
}

impl adr_n6 {
    /// Create a new input variable on the tape.  Caller seeds the outer
    /// forward tangent in `value` (`adfn<6>::new(primal, [direction; 6])`).
    pub fn new_input(value: adfn<6>) -> Self {
        let index = push(TapeEntryN6 {
            parents: [(0, adfn::<6>::new_constant(0.0)); 2],
            num_parents: 0,
        });
        Self { value, index }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    fn unary(self, result_value: adfn<6>, partial: adfn<6>) -> Self {
        let index = push(TapeEntryN6 {
            parents: [(self.index, partial), (0, adfn::<6>::new_constant(0.0))],
            num_parents: 1,
        });
        Self {
            value: result_value,
            index,
        }
    }

    fn binary(self, rhs: Self, result_value: adfn<6>, d_self: adfn<6>, d_rhs: adfn<6>) -> Self {
        let index = push(TapeEntryN6 {
            parents: [(self.index, d_self), (rhs.index, d_rhs)],
            num_parents: 2,
        });
        Self {
            value: result_value,
            index,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AD trait: every op records a tape entry whose partials are adfn<6>.
// The partial is computed using forward-mode arithmetic on adfn<6>, so the
// reverse sweep transports outer-forward tangents along with f64 gradients.
// ═══════════════════════════════════════════════════════════════════════════

impl AD for adr_n6 {
    #[inline]
    fn constant(v: f64) -> Self {
        Self::new_input(adfn::<6>::new_constant(v))
    }

    #[inline]
    fn to_constant(&self) -> f64 {
        self.value.to_constant()
    }

    #[inline]
    fn sin(self) -> Self {
        let v = self.value.sin();
        let d = self.value.cos();
        self.unary(v, d)
    }

    #[inline]
    fn cos(self) -> Self {
        let v = self.value.cos();
        let d = -self.value.sin();
        self.unary(v, d)
    }

    #[inline]
    fn sqrt(self) -> Self {
        let v = self.value.sqrt();
        // d/dx √x = 1/(2√x); guard against the all-zero case as in the f64 tape.
        let d = if v.to_constant().abs() > 0.0 {
            adfn::<6>::new_constant(0.5) / v
        } else {
            adfn::<6>::new_constant(0.0)
        };
        self.unary(v, d)
    }

    #[inline]
    fn ln(self) -> Self {
        let v = self.value.ln();
        let d = adfn::<6>::new_constant(1.0) / self.value;
        self.unary(v, d)
    }

    #[inline]
    fn abs(self) -> Self {
        let d = if self.value.to_constant() >= 0.0 {
            adfn::<6>::new_constant(1.0)
        } else {
            adfn::<6>::new_constant(-1.0)
        };
        self.unary(self.value.abs(), d)
    }

    #[inline]
    fn acos(self) -> Self {
        let v = self.value.acos();
        let one = adfn::<6>::new_constant(1.0);
        let d = -one / (one - self.value * self.value).sqrt();
        self.unary(v, d)
    }

    #[inline]
    fn atan2(self, x: Self) -> Self {
        let denom = x.value * x.value + self.value * self.value;
        let dy = x.value / denom;
        let dx = -self.value / denom;
        self.binary(x, self.value.atan2(x.value), dy, dx)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic operators.
// ═══════════════════════════════════════════════════════════════════════════

impl Add for adr_n6 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let one = adfn::<6>::new_constant(1.0);
        self.binary(rhs, self.value + rhs.value, one, one)
    }
}

impl Sub for adr_n6 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let one = adfn::<6>::new_constant(1.0);
        let neg_one = adfn::<6>::new_constant(-1.0);
        self.binary(rhs, self.value - rhs.value, one, neg_one)
    }
}

impl Mul for adr_n6 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // ∂(a·b)/∂a = b,  ∂(a·b)/∂b = a — partials are adfn<6>, transporting
        // the outer-forward tangent through the multiplication.
        self.binary(rhs, self.value * rhs.value, rhs.value, self.value)
    }
}

impl Div for adr_n6 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv_b = adfn::<6>::new_constant(1.0) / rhs.value;
        let inv_b2 = inv_b * inv_b;
        self.binary(rhs, self.value * inv_b, inv_b, -self.value * inv_b2)
    }
}

impl Neg for adr_n6 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let neg_one = adfn::<6>::new_constant(-1.0);
        self.unary(-self.value, neg_one)
    }
}

impl AddAssign for adr_n6 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl SubAssign for adr_n6 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl MulAssign for adr_n6 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl DivAssign for adr_n6 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl fmt::Debug for adr_n6 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "adr_n6({:.6}, #{})",
            self.value.to_constant(),
            self.index
        )
    }
}

impl PartialEq for adr_n6 {
    fn eq(&self, other: &Self) -> bool {
        self.value.to_constant() == other.value.to_constant()
    }
}

impl PartialOrd for adr_n6 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value
            .to_constant()
            .partial_cmp(&other.value.to_constant())
    }
}
