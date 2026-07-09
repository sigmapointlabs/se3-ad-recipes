// Numerical / matrix code uses index-based loops and complex types pervasively.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::cloned_ref_to_slice_refs)]

//! # se3-ad-recipes
//!
//! Companion code for the arXiv preprint *"Exact Higher-Order Derivatives
//! for SE(3) via Analytical/AD Methods"*. Implements the AD-safe fused
//! scalar basis described in the paper, the eight Hessian recipes that
//! reproduce Table I, and the SE(3) / SE_2(3) / quaternion-storage SE(3)
//! group primitives that the recipes build on.
//!
//! All matrices are stack-allocated fixed-size arrays — no heap. Default
//! features include zero runtime dependencies; the optional `serde`
//! feature pulls in `serde` for the on-disk pose representations.
//!
//! ## Where to start
//!
//! Application code should use the curated [`api`] module or glob-import
//! [`prelude`]. The [`api`] module exposes three nested stability tiers —
//! the application tier ([`api::pose`], [`api::extended_pose`],
//! [`api::quaternion_pose`], [`api::rotation`], [`api::types`]); the
//! expert tier ([`api::expert`]) organized per group as `expert::{so3,
//! se3, se23, quat_se3, ad}`; and the raw paper-aligned modules. See the
//! [`api`] module's rustdoc for the contract on each tier.
//!
//! The raw modules (`so3_*`, `se3_*`, `se23_adsafe`, `se3_quat_adsafe`,
//! `projective`, `jacobians_*`, `autodiff`, ...) remain `pub` for source
//! compatibility and reproducibility but are hidden from rustdoc.
//!
//! ## Dimension-generic operations
//!
//! All vector and matrix operations use `const N: usize` generics:
//! [`dot`], [`norm`], [`add_vec`], [`sub_vec`], [`scale_vec`], [`outer`]
//! for vectors; [`mm`], [`mv`], [`mm_right_transpose`], [`transpose`],
//! [`add_mat`], [`sub_mat`], [`scale_mat`], [`trace`], [`cholesky`] for
//! matrices; [`frob`], [`frob_diff`], [`frob_block`], [`l2_diff`] for
//! norms. Rust infers the dimension from the type aliases (`Mat6`,
//! `Vec3`, etc.), so no size suffix is needed at call sites.
//!
//! The only non-generic helpers are [`cross3`], [`det3`], [`inv3`],
//! and the block insert / extract functions, which are inherently
//! dimension-specific.

pub mod api;
pub mod prelude;

// Paper-level and implementation APIs remain public for source compatibility
// and reproducibility, but are not part of the curated application API.
#[doc(hidden)]
pub mod act;
#[doc(hidden)]
pub mod autodiff;
#[doc(hidden)]
pub mod jacobians_ad;
#[doc(hidden)]
pub mod jacobians_se23_adsafe;
#[doc(hidden)]
pub mod projective;
#[doc(hidden)]
pub mod se23_adsafe;
#[doc(hidden)]
pub mod se3_adsafe;
#[doc(hidden)]
pub mod se3_quat_adsafe;
#[doc(hidden)]
pub mod se3_unsafe;
#[doc(hidden)]
pub mod so3_adsafe;
#[doc(hidden)]
pub mod so3_unsafe;

#[cfg(any(test, feature = "bench-support"))]
#[doc(hidden)]
pub mod graph;
#[cfg(any(test, feature = "bench-support"))]
#[doc(hidden)]
pub mod linalg;
#[cfg(any(test, feature = "bench-support"))]
#[doc(hidden)]
pub mod nll_bench;

#[cfg(test)]
mod nll_tests;

// ─── Type aliases ───────────────────────────────────────────────────────

/// 3×3 matrix, row-major.
#[doc(hidden)]
pub type Mat3 = api::types::Mat3;
/// 6×6 matrix, row-major.
#[doc(hidden)]
pub type Mat6 = api::types::Mat6;
/// 9×9 matrix, row-major. Used for SE_2(3) adjoints, extended-pose
/// covariances, and the lazy-chart filter on the extended pose group.
#[doc(hidden)]
pub type Mat9 = api::types::Mat9;

/// 3-vector.
#[doc(hidden)]
pub type Vec3 = api::types::Vec3;
/// 6-vector: \[ω₁, ω₂, ω₃, v₁, v₂, v₃\] for SE(3) tangent.
#[doc(hidden)]
pub type Vec6 = api::types::Vec6;
/// 9-vector: \[ω₁, ω₂, ω₃, ν₁, ν₂, ν₃, ρ₁, ρ₂, ρ₃\] for SE_2(3) tangent.
/// Index order: rotation \[0..3\], velocity \[3..6\], position \[6..9\].
#[doc(hidden)]
pub type Vec9 = api::types::Vec9;

// ─── Constants ──────────────────────────────────────────────────────────

#[doc(hidden)]
pub const I3: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
#[doc(hidden)]
pub const Z3: Mat3 = [[0.0; 3]; 3];

#[doc(hidden)]
pub const I6: Mat6 = {
    let mut m = [[0.0f64; 6]; 6];
    let mut i = 0;
    while i < 6 {
        m[i][i] = 1.0;
        i += 1;
    }
    m
};
#[doc(hidden)]
pub const Z6: Mat6 = [[0.0; 6]; 6];

#[doc(hidden)]
pub const I9: Mat9 = {
    let mut m = [[0.0f64; 9]; 9];
    let mut i = 0;
    while i < 9 {
        m[i][i] = 1.0;
        i += 1;
    }
    m
};
#[doc(hidden)]
pub const Z9: Mat9 = [[0.0; 9]; 9];

// ─── Dimension-generic helpers ─────────────────────────────────────────
//
// These use const generics so a single implementation covers every
// fixed-size vector / matrix dimension (3, 6, 9, …).
//
// Sized aliases (norm, mm, transpose, …) are thin wrappers kept for
// call-site readability.

// ── Vector operations ──────────────────────────────────────────────────

/// Dot product of two N-vectors.
#[doc(hidden)]
#[inline]
pub fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    let mut s = 0.0;
    for i in 0..N {
        s += a[i] * b[i];
    }
    s
}

/// Euclidean (L₂) norm of an N-vector.
#[doc(hidden)]
#[inline]
pub fn norm<const N: usize>(v: &[f64; N]) -> f64 {
    dot(v, v).sqrt()
}

/// Euclidean distance ‖a − b‖₂ between two N-vectors.
#[doc(hidden)]
#[inline]
pub fn l2_diff<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    let mut s = 0.0;
    for i in 0..N {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

/// Element-wise sum of two N-vectors.
#[doc(hidden)]
#[inline]
pub fn add_vec<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    let mut c = [0.0f64; N];
    for i in 0..N {
        c[i] = a[i] + b[i];
    }
    c
}

/// Element-wise difference of two N-vectors.
#[doc(hidden)]
#[inline]
pub fn sub_vec<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    let mut c = [0.0f64; N];
    for i in 0..N {
        c[i] = a[i] - b[i];
    }
    c
}

/// Scalar-vector multiply.
#[doc(hidden)]
#[inline]
pub fn scale_vec<const N: usize>(s: f64, v: &[f64; N]) -> [f64; N] {
    let mut c = [0.0f64; N];
    for i in 0..N {
        c[i] = s * v[i];
    }
    c
}

/// Outer product a bᵀ → N×N matrix.
#[doc(hidden)]
#[inline]
pub fn outer<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [[f64; N]; N] {
    let mut c = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            c[i][j] = a[i] * b[j];
        }
    }
    c
}

// ── Matrix operations ──────────────────────────────────────────────────

/// N×N matrix-vector multiply.
#[doc(hidden)]
#[inline]
pub fn mv<const N: usize>(m: &[[f64; N]; N], v: &[f64; N]) -> [f64; N] {
    let mut r = [0.0f64; N];
    for i in 0..N {
        for j in 0..N {
            r[i] += m[i][j] * v[j];
        }
    }
    r
}

/// N×N matrix multiply A · B.
#[doc(hidden)]
#[inline]
pub fn mm<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// A · Bᵀ for N×N matrices.
#[doc(hidden)]
#[inline]
pub fn mm_right_transpose<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                c[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    c
}

/// Transpose of an N×N matrix.
#[doc(hidden)]
#[inline]
pub fn transpose<const N: usize>(m: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            c[i][j] = m[j][i];
        }
    }
    c
}

/// Element-wise sum of two N×N matrices.
#[doc(hidden)]
#[inline]
pub fn add_mat<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

/// Element-wise difference of two N×N matrices.
#[doc(hidden)]
#[inline]
pub fn sub_mat<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

/// Scalar-matrix multiply.
#[doc(hidden)]
#[inline]
pub fn scale_mat<const N: usize>(s: f64, m: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            c[i][j] = s * m[i][j];
        }
    }
    c
}

/// Trace of an N×N matrix.
#[doc(hidden)]
#[inline]
pub fn trace<const N: usize>(m: &[[f64; N]; N]) -> f64 {
    let mut s = 0.0;
    for i in 0..N {
        s += m[i][i];
    }
    s
}

/// Cholesky decomposition of an N×N positive-definite matrix.
/// Returns L such that A = L Lᵀ.
#[doc(hidden)]
pub fn cholesky<const N: usize>(a: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut l = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag = a[i][i] - sum;
                l[i][j] = if diag > 0.0 { diag.sqrt() } else { 0.0 };
            } else {
                l[i][j] = if l[j][j].abs() > 1e-30 {
                    (a[i][j] - sum) / l[j][j]
                } else {
                    0.0
                };
            }
        }
    }
    l
}

// ── Norm operations ────────────────────────────────────────────────────

/// Frobenius norm ‖M‖_F of an N×N matrix.
#[doc(hidden)]
#[inline]
pub fn frob<const N: usize>(m: &[[f64; N]; N]) -> f64 {
    let mut s = 0.0;
    for i in 0..N {
        for j in 0..N {
            s += m[i][j] * m[i][j];
        }
    }
    s.sqrt()
}

/// Frobenius norm of the difference ‖A − B‖_F for N×N matrices.
#[doc(hidden)]
#[inline]
pub fn frob_diff<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> f64 {
    let mut s = 0.0;
    for i in 0..N {
        for j in 0..N {
            let d = a[i][j] - b[i][j];
            s += d * d;
        }
    }
    s.sqrt()
}

/// Frobenius norm of a `size × size` sub-block of two N×N matrices,
/// starting at row `r0`, column `c0`: ‖A[r0..r0+size, c0..c0+size] − B[…]‖_F.
#[doc(hidden)]
#[inline]
pub fn frob_block<const N: usize>(
    a: &[[f64; N]; N],
    b: &[[f64; N]; N],
    r0: usize,
    c0: usize,
    size: usize,
) -> f64 {
    let mut s = 0.0;
    for i in 0..size {
        for j in 0..size {
            let d = a[r0 + i][c0 + j] - b[r0 + i][c0 + j];
            s += d * d;
        }
    }
    s.sqrt()
}

// ─── Dimension-specific helpers (non-generic) ──────────────────────────

#[doc(hidden)]
#[inline]
pub fn cross3(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Determinant of 3×3 matrix.
#[doc(hidden)]
pub fn det3(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Inverse of 3×3 matrix (panics if singular).
#[doc(hidden)]
pub fn inv3(m: &Mat3) -> Mat3 {
    let d = det3(m);
    assert!(d.abs() > 1e-15, "inv3: singular matrix, det={:.2e}", d);
    let id = 1.0 / d;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id,
        ],
    ]
}

/// Extract a 3×3 block from a 6×6 matrix at the given row/col offset.
#[doc(hidden)]
pub fn extract_block3(m: &Mat6, row: usize, col: usize) -> Mat3 {
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            b[i][j] = m[row + i][col + j];
        }
    }
    b
}

// ─── Dimension-specific block / structural ops ─────────────────────────

/// Write a 3×3 block into a 6×6 matrix at the given row/col offset.
/// Mirrors `extract_block3` (the read direction).
#[doc(hidden)]
pub fn set_block3_in6(m: &mut Mat6, row: usize, col: usize, b: &Mat3) {
    for i in 0..3 {
        for j in 0..3 {
            m[row + i][col + j] = b[i][j];
        }
    }
}

/// Extract a 3×3 block from a 9×9 matrix at the given row/col offset.
/// Useful for inspecting individual blocks of the SE_2(3) adjoint.
#[doc(hidden)]
pub fn extract_block3_from9(m: &Mat9, row: usize, col: usize) -> Mat3 {
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            b[i][j] = m[row + i][col + j];
        }
    }
    b
}

/// Write a 3×3 block into a 9×9 matrix at the given row/col offset.
/// Used to assemble the SE_2(3) adjoint from its R, \[v\]×R, \[p\]×R blocks.
#[doc(hidden)]
pub fn set_block3_in9(m: &mut Mat9, row: usize, col: usize, b: &Mat3) {
    for i in 0..3 {
        for j in 0..3 {
            m[row + i][col + j] = b[i][j];
        }
    }
}

// ─── Tests for new operations ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_mat9(a: &Mat9, b: &Mat9, tol: f64) -> bool {
        for i in 0..9 {
            for j in 0..9 {
                if (a[i][j] - b[i][j]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    fn approx_eq_vec9(a: &Vec9, b: &Vec9, tol: f64) -> bool {
        (0..9).all(|i| (a[i] - b[i]).abs() < tol)
    }

    // ─── I9 / Z9 sanity ────────────────────────────────────────────────

    #[test]
    fn i9_is_identity_for_mm9() {
        // Build a non-trivial test matrix
        let mut m = [[0.0f64; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                m[i][j] = (i * 9 + j) as f64 + 1.0;
            }
        }
        let lhs = mm(&I9, &m);
        let rhs = mm(&m, &I9);
        assert!(approx_eq_mat9(&lhs, &m, 1e-15));
        assert!(approx_eq_mat9(&rhs, &m, 1e-15));
    }

    // ─── mv ───────────────────────────────────────────────────────────

    #[test]
    fn mv9_with_identity_returns_input() {
        let v: Vec9 = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0];
        let r = mv(&I9, &v);
        assert!(approx_eq_vec9(&v, &r, 1e-15));
    }

    // ─── mm / mm_right_transpose ─────────────────────────────────────

    #[test]
    fn mm9_right_transpose_matches_mm9_with_explicit_transpose() {
        let mut a = [[0.0f64; 9]; 9];
        let mut b = [[0.0f64; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                a[i][j] = ((i + 1) as f64 * 0.13 - (j as f64) * 0.07).sin();
                b[i][j] = ((i + 1) as f64 * 0.21 + (j as f64) * 0.11).cos();
            }
        }
        let direct = mm_right_transpose(&a, &b);
        let via_transpose = mm(&a, &transpose(&b));
        assert!(approx_eq_mat9(&direct, &via_transpose, 1e-12));
    }

    // ─── transpose ────────────────────────────────────────────────────

    #[test]
    fn transpose9_is_involutive() {
        let mut m = [[0.0f64; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                m[i][j] = (i as f64 - j as f64).powi(3);
            }
        }
        let t = transpose(&m);
        let tt = transpose(&t);
        assert!(approx_eq_mat9(&m, &tt, 1e-15));
    }

    // ─── add_mat / sub_mat ───────────────────────────────────────────────────

    #[test]
    fn add9_sub9_roundtrip() {
        let mut a = [[0.0f64; 9]; 9];
        let mut b = [[0.0f64; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                a[i][j] = (i * 7 + j * 3) as f64;
                b[i][j] = (i + 11 * j) as f64;
            }
        }
        let sum = add_mat(&a, &b);
        let diff = sub_mat(&sum, &b);
        assert!(approx_eq_mat9(&diff, &a, 1e-15));
    }

    #[test]
    fn scale_mat9_doubling() {
        let mut m = [[0.0f64; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                m[i][j] = (i + j) as f64;
            }
        }
        let doubled = scale_mat(2.0, &m);
        let added = add_mat(&m, &m);
        assert!(approx_eq_mat9(&doubled, &added, 1e-15));
    }

    // ─── cholesky ─────────────────────────────────────────────────────

    #[test]
    fn cholesky9_reconstructs_diagonal() {
        let mut a = [[0.0f64; 9]; 9];
        for i in 0..9 {
            a[i][i] = ((i + 1) as f64) * 2.0;
        }
        let l = cholesky(&a);
        let reconstructed = mm_right_transpose(&l, &l);
        assert!(approx_eq_mat9(&a, &reconstructed, 1e-12));
    }

    #[test]
    fn cholesky9_reconstructs_dense_spd() {
        // Build a dense SPD matrix as A = M Mᵀ + αI
        let mut m = [[0.0f64; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                m[i][j] = ((i + 1) as f64 * 0.31 - (j as f64) * 0.17).cos();
            }
        }
        let mut a = mm_right_transpose(&m, &m);
        for i in 0..9 {
            a[i][i] += 1.0; // ensure positive definiteness
        }
        let l = cholesky(&a);
        let reconstructed = mm_right_transpose(&l, &l);
        assert!(approx_eq_mat9(&a, &reconstructed, 1e-10));
    }

    // ─── frob_diff ─────────────────────────────────────────────────────────

    #[test]
    fn frob9_of_zero_versus_identity_is_three() {
        // ‖I9‖_F = √(trace(I9ᵀI9)) = √9 = 3
        let f = frob_diff(&I9, &Z9);
        assert!((f - 3.0).abs() < 1e-15);
    }

    // ─── Block ops ─────────────────────────────────────────────────────

    #[test]
    fn block3_in9_round_trip() {
        let block: Mat3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let mut m = [[0.0f64; 9]; 9];
        set_block3_in9(&mut m, 3, 6, &block);
        let extracted = extract_block3_from9(&m, 3, 6);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(extracted[i][j], block[i][j]);
            }
        }
    }

    #[test]
    fn block3_in9_assembles_block_diagonal_correctly() {
        // Build a 9×9 matrix with three 3×3 blocks on the diagonal
        // and verify the structure.
        let r = I3;
        let mut ad = Z9;
        set_block3_in9(&mut ad, 0, 0, &r);
        set_block3_in9(&mut ad, 3, 3, &r);
        set_block3_in9(&mut ad, 6, 6, &r);
        // This should equal I9
        assert!(approx_eq_mat9(&ad, &I9, 1e-15));
    }

    #[test]
    fn block3_in6_round_trip() {
        let block: Mat3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let mut m = [[0.0f64; 6]; 6];
        set_block3_in6(&mut m, 0, 3, &block);
        let extracted = extract_block3(&m, 0, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(extracted[i][j], block[i][j]);
            }
        }
    }
}
