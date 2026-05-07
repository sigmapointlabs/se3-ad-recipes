// Numerical / matrix code uses index-based loops and complex types pervasively.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::cloned_ref_to_slice_refs)]

//! # se3-inference
//!
//! Higher-order uncertainty propagation and saddlepoint marginalization
//! on the SE(3) Lie group, with extension to SE_2(3) (extended pose group)
//! for inertial navigation.
//!
//! All matrices are stack-allocated fixed-size arrays — no heap, no deps.
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

pub mod autodiff;
pub mod jacobians_ad;
pub mod projective;
pub mod se3_adsafe;
pub mod se3_unsafe;
pub mod so3_adsafe;
pub mod so3_unsafe;

pub mod nll_bench;

#[cfg(test)]
mod nll_tests;

#[cfg(test)]
mod manif_recipe_minimal;

// ─── Type aliases ───────────────────────────────────────────────────────

/// 3×3 matrix, row-major.
pub type Mat3 = [[f64; 3]; 3];
/// 6×6 matrix, row-major.
pub type Mat6 = [[f64; 6]; 6];
/// 9×9 matrix, row-major. Used for SE_2(3) adjoints, extended-pose
/// covariances, and the lazy-chart filter on the extended pose group.
pub type Mat9 = [[f64; 9]; 9];

/// 3-vector.
pub type Vec3 = [f64; 3];
/// 6-vector: \[ω₁, ω₂, ω₃, v₁, v₂, v₃\] for SE(3) tangent.
pub type Vec6 = [f64; 6];
/// 9-vector: \[ω₁, ω₂, ω₃, ν₁, ν₂, ν₃, ρ₁, ρ₂, ρ₃\] for SE_2(3) tangent.
/// Index order: rotation \[0..3\], velocity \[3..6\], position \[6..9\].
pub type Vec9 = [f64; 9];

// ─── Constants ──────────────────────────────────────────────────────────

pub const I3: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
pub const Z3: Mat3 = [[0.0; 3]; 3];

pub const I6: Mat6 = {
    let mut m = [[0.0f64; 6]; 6];
    let mut i = 0;
    while i < 6 {
        m[i][i] = 1.0;
        i += 1;
    }
    m
};
pub const Z6: Mat6 = [[0.0; 6]; 6];

pub const I9: Mat9 = {
    let mut m = [[0.0f64; 9]; 9];
    let mut i = 0;
    while i < 9 {
        m[i][i] = 1.0;
        i += 1;
    }
    m
};
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
#[inline]
pub fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    let mut s = 0.0;
    for i in 0..N {
        s += a[i] * b[i];
    }
    s
}

/// Euclidean (L₂) norm of an N-vector.
#[inline]
pub fn norm<const N: usize>(v: &[f64; N]) -> f64 {
    dot(v, v).sqrt()
}

/// Euclidean distance ‖a − b‖₂ between two N-vectors.
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
#[inline]
pub fn add_vec<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    let mut c = [0.0f64; N];
    for i in 0..N {
        c[i] = a[i] + b[i];
    }
    c
}

/// Element-wise difference of two N-vectors.
#[inline]
pub fn sub_vec<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    let mut c = [0.0f64; N];
    for i in 0..N {
        c[i] = a[i] - b[i];
    }
    c
}

/// Scalar-vector multiply.
#[inline]
pub fn scale_vec<const N: usize>(s: f64, v: &[f64; N]) -> [f64; N] {
    let mut c = [0.0f64; N];
    for i in 0..N {
        c[i] = s * v[i];
    }
    c
}

/// Outer product a bᵀ → N×N matrix.
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

#[inline]
pub fn cross3(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Determinant of 3×3 matrix.
pub fn det3(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Inverse of 3×3 matrix (panics if singular).
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
pub fn set_block3_in6(m: &mut Mat6, row: usize, col: usize, b: &Mat3) {
    for i in 0..3 {
        for j in 0..3 {
            m[row + i][col + j] = b[i][j];
        }
    }
}

/// Extract a 3×3 block from a 9×9 matrix at the given row/col offset.
/// Useful for inspecting individual blocks of the SE_2(3) adjoint.
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

    #[test]
    fn z9_is_zero() {
        for i in 0..9 {
            for j in 0..9 {
                assert_eq!(Z9[i][j], 0.0);
            }
        }
    }

    #[test]
    fn i9_diagonal_only() {
        for i in 0..9 {
            for j in 0..9 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(I9[i][j], expected, "I9[{}][{}]", i, j);
            }
        }
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
    fn cholesky9_reconstructs_identity() {
        let l = cholesky(&I9);
        // Cholesky of identity is identity
        assert!(approx_eq_mat9(&l, &I9, 1e-15));
    }

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
    fn frob9_of_identity_minus_identity_is_zero() {
        assert_eq!(frob_diff(&I9, &I9), 0.0);
    }

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

    // ─── Mat6 additions sanity ─────────────────────────────────────────

    #[test]
    fn sub6_undoes_add6() {
        let mut a = [[0.0f64; 6]; 6];
        let mut b = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                a[i][j] = (i + 2 * j) as f64;
                b[i][j] = (3 * i) as f64 - j as f64;
            }
        }
        let s = add_mat(&a, &b);
        let recovered = sub_mat(&s, &b);
        for i in 0..6 {
            for j in 0..6 {
                assert!((recovered[i][j] - a[i][j]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn scale_mat6_zero_gives_zero() {
        let mut m = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                m[i][j] = (i + j) as f64 + 1.0;
            }
        }
        let zeroed = scale_mat(0.0, &m);
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(zeroed[i][j], 0.0);
            }
        }
    }
}
