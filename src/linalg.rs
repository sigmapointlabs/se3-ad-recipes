//! Test/bench support: runtime-dimension Cholesky factorization with honest
//! positive-definiteness detection.
//!
//! [`cholesky_n`] factors an n×n symmetric matrix as A = L·Lᵀ and returns
//! `None` the moment a pivot fails to be strictly positive — no clamping, no
//! silent substitution of a tiny diagonal.  Callers that need a PD guard
//! (skip-and-count in Monte-Carlo consistency studies, damping retries in a
//! Gauss–Newton loop) branch on the `Option`; callers that "know" their
//! matrix is PD get a loud failure path instead of a poisoned solve.
//!
//! This is the runtime-n port of the fixed-size `chol6` used by
//! `examples/nees_consistency.rs`.  The const-generic [`crate::cholesky`]
//! helper predates it and zero-fills bad pivots; prefer this module whenever
//! the factorization can legitimately fail.

/// Lower-triangular Cholesky factor of an n×n positive-definite matrix.
///
/// Obtained from [`cholesky_n`]; consumed via [`Chol::solve`].
pub struct Chol {
    l: Vec<Vec<f64>>,
}

/// Cholesky factorization A = L·Lᵀ of a symmetric n×n matrix, reading only
/// the lower triangle of `a`.  Returns `None` if any pivot is ≤ 0, i.e. the
/// matrix is not (numerically) positive-definite.
pub fn cholesky_n(a: &[Vec<f64>]) -> Option<Chol> {
    let n = a.len();
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let d = a[i][i] - sum;
                if d <= 0.0 {
                    return None;
                }
                l[i][i] = d.sqrt();
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    Some(Chol { l })
}

impl Chol {
    /// Dimension n of the factored matrix.
    pub fn dim(&self) -> usize {
        self.l.len()
    }

    /// Solve (L·Lᵀ)·x = b by forward + back substitution.
    ///
    /// # Panics
    /// Panics if `b.len() != self.dim()`.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.l.len();
        assert_eq!(
            b.len(),
            n,
            "rhs length {} != factor dimension {}",
            b.len(),
            n
        );
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= self.l[i][k] * y[k];
            }
            y[i] = s / self.l[i][i];
        }
        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= self.l[k][i] * x[k];
            }
            x[i] = s / self.l[i][i];
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic SPD test matrix A = MᵀM + I (n = 7, deliberately not a
    /// multiple of 6, exercising the runtime dimension).
    fn spd7() -> Vec<Vec<f64>> {
        let n = 7;
        // Fixed pseudo-random entries (no RNG dependency in unit tests).
        let m: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| ((i * 31 + j * 17 + 3) % 13) as f64 / 13.0 - 0.5)
                    .collect()
            })
            .collect();
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    a[i][j] += m[k][i] * m[k][j];
                }
            }
            a[i][i] += 1.0;
        }
        a
    }

    #[test]
    fn solve_roundtrip_spd() {
        let a = spd7();
        let x_true: Vec<f64> = (0..7).map(|i| (i as f64) - 2.5).collect();
        let b: Vec<f64> = (0..7)
            .map(|i| (0..7).map(|j| a[i][j] * x_true[j]).sum())
            .collect();
        let chol = cholesky_n(&a).expect("SPD matrix must factor");
        assert_eq!(chol.dim(), 7);
        let x = chol.solve(&b);
        for i in 0..7 {
            assert!(
                (x[i] - x_true[i]).abs() < 1e-12,
                "x[{i}] = {} vs {}",
                x[i],
                x_true[i]
            );
        }
    }

    #[test]
    fn indefinite_returns_none() {
        // Eigenvalues {1, -1}: pivot 2 fails.
        let a = vec![vec![1.0, 0.0], vec![0.0, -1.0]];
        assert!(cholesky_n(&a).is_none());
    }

    #[test]
    fn singular_psd_returns_none() {
        // Rank-1 outer product vvᵀ, v = [1, 2]: second pivot is exactly 0
        // and must be rejected, not clamped.
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(cholesky_n(&a).is_none());
    }

    #[test]
    fn negative_leading_pivot_returns_none() {
        let a = vec![vec![-1.0]];
        assert!(cholesky_n(&a).is_none());
    }
}
