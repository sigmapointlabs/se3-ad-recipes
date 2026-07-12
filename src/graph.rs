//! Test/bench support: SE(3) pose-graph factors with exact seeded-AD Hessians.
//!
//! Graduated aggregation patterns for pose-graph estimation: between/prior
//! factors, their AD-generic NLL and *analytical* gradient, a one-pass
//! seeded-AD exact Hessian per factor, dense linearized couplings from
//! marginalization ([`LinearFactor`]), and a [`GraphProblem`] that assembles
//! the robust Gauss–Newton information and the exact observed information
//! over a K-node graph.  Consumed by `examples/posegraph_consistency.rs`
//! (the NEES consistency study, posegraph edition) and the in-tree tests.
//!
//! # Conventions (fixed here, used everywhere below)
//!
//! - **Right perturbation of states.** Node k's pose is parameterized as
//!   `X_k · Exp(δ_k)` with `δ_k ∈ ℝ⁶` in the tangent at the current base
//!   `X_k` (body-frame perturbation). Tangent ordering is `ξ = [ω; t]`,
//!   rotation first.
//! - **Residual on the right.** A between factor measuring `Z ≈ X_i⁻¹·X_j`
//!   defines its residual `r` by
//!
//!   ```text
//!   X_i⁻¹·X_j = Z·Exp(r)   ⟺   r = Log(Z⁻¹ · X_i⁻¹ · X_j)
//!   ```
//!
//!   i.e. `r` is the right-tangent discrepancy of the predicted relative
//!   pose, expressed in the measurement frame. A prior factor with
//!   reference `X_ref` is the same construction against a frozen endpoint:
//!   `X_ref = X·Exp(r)`, `r = Log(X⁻¹·X_ref)` (a between factor with
//!   `Z = I` and the reference as the fixed second node).
//! - **Whitening is a square root of the information.** Each factor
//!   carries `sqrt_info = L`, any matrix satisfying `W = LᵀL` for the
//!   measurement information `W`.  Mind the orientation for correlated
//!   noise: the conventional **lower** Cholesky factor `C` of `W`
//!   satisfies `C·Cᵀ = W`, so pass its transpose, `L = Cᵀ` (an upper
//!   factor) — passing `C` itself silently weights by `CᵀC ≠ W`.  For
//!   independent per-axis noise use `diag(1/σ)` (orientation-free; see
//!   [`diagonal_sqrt_info`]).  The whitened residual is `r_w = L·r`,
//!   `s = ‖L·r‖²`.  Gaussian factors cost `½s`; robust factors the
//!   pseudo-Huber kernel `κ²(√(1+s/κ²) − 1)` (see
//!   [`crate::nll_bench::pseudo_huber`]).
//!
//! # The one-pass exact Hessian (`between_hessian_seeded`)
//!
//! [`between_gradient_g`] is the *analytical* gradient of the factor NLL
//! with respect to the stacked coordinate `d = [δ_i; δ_j] ∈ ℝ¹²`, valid at
//! any `d` — not just `d = 0`. That requires the chart parameterization
//! factors `J_r(δ)` **inside the AD-generic body**: perturbing the
//! coordinate, `Exp(δ + ε) = Exp(δ)·Exp(J_r(δ)·ε) + O(ε²)`, so
//!
//! ```text
//! ∂r/∂δ_j =  J_r⁻¹(r) · J_r(δ_j)
//! ∂r/∂δ_i = −J_r⁻¹(r) · Ad(X_rel⁻¹) · J_r(δ_i),   X_rel = X_i'⁻¹·X_j'
//! ```
//!
//! At `d = 0` the factors reduce to `J_r(0) = I`, but their *derivative*
//! does not vanish — it is exactly the curvature term that distinguishes
//! the observed information from the Gauss–Newton information. Seeding the
//! gradient once with `adfn<12>` therefore yields the exact 12×12 factor
//! Hessian in a single pass (no nested duals, no FD): each gradient
//! component carries its full 12-vector of derivatives.
//!
//! Scattering the per-factor 12×12 blocks into the 6K×6K matrix reproduces
//! the global Hessian of the total NLL exactly (the NLL is a sum of factor
//! terms, each touching only its own two nodes); the in-tree test pins this
//! against a global nested-dual `D2<36>` oracle at ≤ 1e-12 relative.
//!
//! # Dense linearized couplings ([`LinearFactor`])
//!
//! Marginalizing shared variables at the current bases (landmarks in a
//! GraphSLAM reduce step, an eliminated subchain) leaves quadratic
//! "springs" between the surviving nodes: a gradient block and an
//! information block with no `Z`-style measurement generating them.
//! [`LinearFactor`] carries those blocks verbatim over any number of
//! nodes.  It is a **chart-local** object — the blocks are the quadratic
//! model `½ dᵀ·info·d + gradᵀ·d` in the tangents at the bases it was
//! built from, so it must be rebuilt after every [`GraphProblem::re_base`]
//! (exactly like re-linearizing the factors it summarizes).

use crate::autodiff::ad_trait::AD;
use crate::autodiff::forward_ad::adfn;
use crate::nll_bench::pseudo_huber;
use crate::se3_adsafe::{Mat6G, PoseG, Vec6G, adjoint_g, mv6_g, pose_to_g, se3_jr_g, se3_jr_inv_g};
use crate::se3_unsafe::{Pose, right_update};
use crate::{Mat6, Vec6, mm, mv};

// ─── Factor types ────────────────────────────────────────────────────────

/// Relative-pose factor between nodes `i` and `j`: measurement
/// `Z ≈ X_i⁻¹·X_j`, whitening `L` (`W = LᵀL`), optional pseudo-Huber `κ`.
#[derive(Debug, Clone)]
pub struct BetweenFactor {
    pub i: usize,
    pub j: usize,
    /// Measured relative pose `Z`.
    pub z: Pose,
    /// Square root `L` of the measurement information: `s = ‖L·r‖²`.
    /// Use [`diagonal_sqrt_info`] for independent per-axis noise.
    pub sqrt_info: Mat6,
    /// `None` → Gaussian `½s`; `Some(κ)` → pseudo-Huber `κ²(√(1+s/κ²)−1)`.
    pub kappa: Option<f64>,
}

/// Gaussian prior anchoring node `node` to `x_ref`:
/// `r = Log(X⁻¹·X_ref)`, cost `½‖L·r‖²`.
#[derive(Debug, Clone)]
pub struct PriorFactor {
    pub node: usize,
    pub x_ref: Pose,
    /// Square root `L` of the prior information: `W = LᵀL`.
    pub sqrt_info: Mat6,
}

/// Dense linearized coupling over `nodes`: the quadratic model
/// `½ dᵀ·info·d + gradᵀ·d` in the stacked right perturbations of its
/// nodes, valid **only at the bases it was built from** (rebuild after
/// [`GraphProblem::re_base`]).  Being exactly quadratic, its Gauss–Newton
/// information and exact Hessian coincide (`info` contributes to both).
#[derive(Debug, Clone)]
pub struct LinearFactor {
    /// Global node indices, in the order the blocks are stacked.
    pub nodes: Vec<usize>,
    /// Gradient at `d = 0`, length `6·nodes.len()`.
    pub grad: Vec<f64>,
    /// Symmetric information block, `6·nodes.len()` square.
    pub info: Vec<Vec<f64>>,
}

/// Whitening matrix for independent per-axis noise: `L = diag(entries)`
/// with `entries = 1/σ` per tangent component.
pub fn diagonal_sqrt_info(entries: &Vec6) -> Mat6 {
    let mut l = [[0.0f64; 6]; 6];
    for (a, e) in entries.iter().enumerate() {
        l[a][a] = *e;
    }
    l
}

// ─── Small AD-generic helpers ────────────────────────────────────────────

/// y = Mᵀ·v for a 6×6 `T`-valued matrix.
fn mtv6_g<T: AD>(m: &Mat6G<T>, v: &Vec6G<T>) -> Vec6G<T> {
    std::array::from_fn(|i| {
        let mut s = T::constant(0.0);
        for k in 0..6 {
            s += m[k][i] * v[k];
        }
        s
    })
}

/// Whitened square `s = ‖L·r‖²` and the pulled-back weight vector
/// `Lᵀ·(L·r) = W·r`.
fn whitened_square_g<T: AD>(l: &Mat6G<T>, r: &Vec6G<T>) -> (T, Vec6G<T>) {
    let rw = mv6_g(l, r);
    let mut s = T::constant(0.0);
    for a in 0..6 {
        s += rw[a] * rw[a];
    }
    (s, mtv6_g(l, &rw))
}

/// Perturbed error term shared by NLL and gradient: applies the stacked
/// right perturbation `d = [δ_i; δ_j]`, returns the residual
/// `r = Log(Z⁻¹·X_rel)` and the perturbed relative pose
/// `X_rel = (X_i·Exp(δ_i))⁻¹·(X_j·Exp(δ_j))`.
fn perturbed_error_g<T: AD>(
    d: &[T; 12],
    base_i: &PoseG<T>,
    base_j: &PoseG<T>,
    z: &PoseG<T>,
) -> (Vec6G<T>, PoseG<T>) {
    let di: Vec6G<T> = std::array::from_fn(|a| d[a]);
    let dj: Vec6G<T> = std::array::from_fn(|a| d[6 + a]);
    let xi = base_i.compose(&PoseG::exp(&di));
    let xj = base_j.compose(&PoseG::exp(&dj));
    let x_rel = xi.inverse().compose(&xj);
    let e = z.inverse().compose(&x_rel);
    (e.log(), x_rel)
}

// ─── AD-generic NLL and analytical gradient ──────────────────────────────

/// Between-factor NLL at stacked right perturbation `d = [δ_i; δ_j]`.
///
/// `r = Log(Z⁻¹·(X_i·Exp(δ_i))⁻¹·(X_j·Exp(δ_j)))`, `s = ‖L·r‖²`;
/// Gaussian `½s` for `kappa = None`, pseudo-Huber `κ²(√(1+s/κ²)−1)` else.
pub fn between_nll_g<T: AD>(
    d: &[T; 12],
    base_i: &PoseG<T>,
    base_j: &PoseG<T>,
    z: &PoseG<T>,
    sqrt_info: &Mat6G<T>,
    kappa: Option<f64>,
) -> T {
    let (r, _) = perturbed_error_g(d, base_i, base_j, z);
    let (s, _) = whitened_square_g(sqrt_info, &r);
    match kappa {
        None => T::constant(0.5) * s,
        Some(k) => pseudo_huber(s, T::constant(k * k)),
    }
}

/// **Analytical** gradient of [`between_nll_g`] with respect to `d`, valid
/// at any `d` — the chart parameterization factors `J_r(δ_i)`, `J_r(δ_j)`
/// are applied inside the `T`-generic body (see the module doc).
///
/// With `q = w·LᵀL·r` (`w` the robust weight, `1` for Gaussian):
///
/// ```text
/// g_i = −J_r(δ_i)ᵀ · Ad(X_rel⁻¹)ᵀ · J_r⁻¹(r)ᵀ · q
/// g_j =  J_r(δ_j)ᵀ · J_r⁻¹(r)ᵀ · q
/// ```
///
/// Seeding this with `adfn<12>` tangents at `d = 0` yields the exact
/// 12×12 factor Hessian in one pass ([`between_hessian_seeded`]).
pub fn between_gradient_g<T: AD>(
    d: &[T; 12],
    base_i: &PoseG<T>,
    base_j: &PoseG<T>,
    z: &PoseG<T>,
    sqrt_info: &Mat6G<T>,
    kappa: Option<f64>,
) -> [T; 12] {
    let (r, x_rel) = perturbed_error_g(d, base_i, base_j, z);

    // Robust weight w = 2ρ'(s): 1 for Gaussian, 1/√(1+s/κ²) for pseudo-Huber.
    let (s, wr) = whitened_square_g(sqrt_info, &r);
    let w = match kappa {
        None => T::constant(1.0),
        Some(k) => {
            let k2 = T::constant(k * k);
            T::constant(1.0) / (T::constant(1.0) + s / k2).sqrt()
        }
    };

    // q = w·W·r with W = LᵀL.
    let q: Vec6G<T> = std::array::from_fn(|a| w * wr[a]);

    // u = J_r⁻¹(r)ᵀ·q, then pull back through the two endpoint chains.
    let jr_inv_r = se3_jr_inv_g::<T>(&r);
    let u = mtv6_g(&jr_inv_r, &q);

    let xr_inv = x_rel.inverse();
    let ad = adjoint_g::<T>(&xr_inv.rot, &xr_inv.trans);
    let v = mtv6_g(&ad, &u);

    let di: Vec6G<T> = std::array::from_fn(|a| d[a]);
    let dj: Vec6G<T> = std::array::from_fn(|a| d[6 + a]);
    let g_i = mtv6_g(&se3_jr_g::<T>(&di), &v);
    let g_j = mtv6_g(&se3_jr_g::<T>(&dj), &u);

    std::array::from_fn(|k| if k < 6 { -g_i[k] } else { g_j[k - 6] })
}

// ─── f64 fast path: linearization at δ = 0 ──────────────────────────────

/// Linearization of one between factor at `δ = 0`: residual, the two
/// residual Jacobians, and the robust weight.  This is the Gauss–Newton
/// building block (`J_r(0) = I`, so no chart factors appear here).
#[derive(Debug, Clone)]
pub struct BetweenLin {
    /// `r = Log(Z⁻¹·X_i⁻¹·X_j)`.
    pub r: Vec6,
    /// `∂r/∂δ_i = −J_r⁻¹(r)·Ad(X_rel⁻¹)`.
    pub j_i: Mat6,
    /// `∂r/∂δ_j = J_r⁻¹(r)`.
    pub j_j: Mat6,
    /// Robust weight `w = 2ρ'(s)` (`1.0` for Gaussian).
    pub w: f64,
}

/// f64 fast path: linearize one between factor at the current bases.
pub fn between_linearize(
    base_i: &Pose,
    base_j: &Pose,
    z: &Pose,
    sqrt_info: &Mat6,
    kappa: Option<f64>,
) -> BetweenLin {
    let x_rel = base_i.inverse().compose(base_j);
    let e = z.inverse().compose(&x_rel);
    let r = e.log();

    let jr_inv = se3_jr_inv_g::<f64>(&r);
    let xr_inv = x_rel.inverse();
    let ad = adjoint_g::<f64>(&xr_inv.rot, &xr_inv.trans);
    let mut j_i = [[0.0f64; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            let mut acc = 0.0;
            for k in 0..6 {
                acc += jr_inv[row][k] * ad[k][col];
            }
            j_i[row][col] = -acc;
        }
    }

    let w = match kappa {
        None => 1.0,
        Some(k) => {
            let rw = mv(sqrt_info, &r);
            let s: f64 = rw.iter().map(|v| v * v).sum();
            1.0 / (1.0 + s / (k * k)).sqrt()
        }
    };

    BetweenLin {
        r,
        j_i,
        j_j: jr_inv,
        w,
    }
}

// ─── Exact factor Hessian: one adfn<12> pass ─────────────────────────────

/// Exact 12×12 Hessian of one between factor at `δ = 0` via a **single**
/// `adfn<12>`-seeded evaluation of the analytical gradient
/// [`between_gradient_g`] — no nested duals, no finite differences.
/// Row `k` is the tangent of gradient component `k`; the result is
/// symmetric up to floating-point roundoff.
pub fn between_hessian_seeded(
    base_i: &Pose,
    base_j: &Pose,
    z: &Pose,
    sqrt_info: &Mat6,
    kappa: Option<f64>,
) -> [[f64; 12]; 12] {
    type T = adfn<12>;
    let d: [T; 12] = std::array::from_fn(|k| {
        let mut t = [0.0; 12];
        t[k] = 1.0;
        T::new(0.0, t)
    });
    let bi = pose_to_g::<T>(base_i);
    let bj = pose_to_g::<T>(base_j);
    let zg = pose_to_g::<T>(z);
    let l: Mat6G<T> =
        std::array::from_fn(|a| std::array::from_fn(|b| T::constant(sqrt_info[a][b])));
    let g = between_gradient_g::<T>(&d, &bi, &bj, &zg, &l, kappa);
    std::array::from_fn(|k| g[k].tangent())
}

// ─── Graph assembly ──────────────────────────────────────────────────────

/// A pose graph at its current linearization point: node bases, prior
/// factors, between factors, and dense linearized couplings.  All matrix
/// outputs are dense `6K × 6K` row-major `Vec<Vec<f64>>` in node-major
/// block order.
pub struct GraphProblem {
    /// Current base pose of each node; perturbations are `X_k·Exp(δ_k)`.
    pub poses: Vec<Pose>,
    pub priors: Vec<PriorFactor>,
    pub factors: Vec<BetweenFactor>,
    /// Chart-local quadratic couplings (marginalization springs).  Valid
    /// only at the current bases — rebuild after [`Self::re_base`].
    pub linear: Vec<LinearFactor>,
}

impl GraphProblem {
    /// Total tangent dimension `6K`.
    pub fn dim(&self) -> usize {
        6 * self.poses.len()
    }

    /// Gradient of the total NLL at `δ = 0` (length `6K`).
    pub fn gradient(&self) -> Vec<f64> {
        let dim = self.dim();
        let mut g = vec![0.0f64; dim];
        let id = Pose::identity();

        for p in &self.priors {
            let lin = between_linearize(&self.poses[p.node], &p.x_ref, &id, &p.sqrt_info, None);
            let q = weighted_residual(&p.sqrt_info, &lin.r, lin.w);
            accumulate_gradient_block(&mut g, p.node, &lin.j_i, &q);
        }
        for f in &self.factors {
            let lin = between_linearize(
                &self.poses[f.i],
                &self.poses[f.j],
                &f.z,
                &f.sqrt_info,
                f.kappa,
            );
            let q = weighted_residual(&f.sqrt_info, &lin.r, lin.w);
            accumulate_gradient_block(&mut g, f.i, &lin.j_i, &q);
            accumulate_gradient_block(&mut g, f.j, &lin.j_j, &q);
        }
        for lf in &self.linear {
            debug_assert_eq!(lf.grad.len(), 6 * lf.nodes.len());
            for (bi, &node) in lf.nodes.iter().enumerate() {
                for a in 0..6 {
                    g[6 * node + a] += lf.grad[6 * bi + a];
                }
            }
        }
        g
    }

    /// Robust Gauss–Newton information at `δ = 0`: per factor
    /// `w·JᵀWJ` blocks (IRLS weighting — the ρ″ rank-1 "Triggs correction"
    /// is deliberately dropped) plus the [`LinearFactor`] blocks,
    /// scattered into `6K × 6K`.  This is the information a robust-GN
    /// solver reports — it drops the residual-curvature terms that
    /// [`Self::exact_hessian`] keeps.
    pub fn gn_information(&self) -> Vec<Vec<f64>> {
        let dim = self.dim();
        let mut info = vec![vec![0.0f64; dim]; dim];
        let id = Pose::identity();

        for p in &self.priors {
            let lin = between_linearize(&self.poses[p.node], &p.x_ref, &id, &p.sqrt_info, None);
            let lj = mm(&p.sqrt_info, &lin.j_i);
            accumulate_gn_block(&mut info, p.node, p.node, &lj, &lj, lin.w);
        }
        for f in &self.factors {
            let lin = between_linearize(
                &self.poses[f.i],
                &self.poses[f.j],
                &f.z,
                &f.sqrt_info,
                f.kappa,
            );
            let lji = mm(&f.sqrt_info, &lin.j_i);
            let ljj = mm(&f.sqrt_info, &lin.j_j);
            accumulate_gn_block(&mut info, f.i, f.i, &lji, &lji, lin.w);
            accumulate_gn_block(&mut info, f.j, f.j, &ljj, &ljj, lin.w);
            accumulate_gn_block(&mut info, f.i, f.j, &lji, &ljj, lin.w);
            accumulate_gn_block(&mut info, f.j, f.i, &ljj, &lji, lin.w);
        }
        for lf in &self.linear {
            scatter_linear_info(&mut info, lf);
        }
        info
    }

    /// Exact Hessian of the total NLL at `δ = 0`: per-factor 12×12 blocks
    /// from [`between_hessian_seeded`] (priors: the frozen-endpoint 6×6
    /// sub-block) plus the [`LinearFactor`] blocks (quadratic by
    /// construction, so their exact Hessian *is* their information),
    /// scattered into `6K × 6K`.  Equals the global nested-dual Hessian to
    /// machine precision (pinned by the in-tree oracle test).
    pub fn exact_hessian(&self) -> Vec<Vec<f64>> {
        let dim = self.dim();
        let mut h = vec![vec![0.0f64; dim]; dim];
        let id = Pose::identity();

        for p in &self.priors {
            // Prior = between factor against the frozen reference endpoint:
            // only the (node, node) 6×6 sub-block is scattered; the
            // reference's rows/columns are discarded.
            let h12 =
                between_hessian_seeded(&self.poses[p.node], &p.x_ref, &id, &p.sqrt_info, None);
            for a in 0..6 {
                for b in 0..6 {
                    h[6 * p.node + a][6 * p.node + b] += h12[a][b];
                }
            }
        }
        for f in &self.factors {
            let h12 = between_hessian_seeded(
                &self.poses[f.i],
                &self.poses[f.j],
                &f.z,
                &f.sqrt_info,
                f.kappa,
            );
            let idx = [f.i, f.j];
            for (ba, &na) in idx.iter().enumerate() {
                for (bb, &nb) in idx.iter().enumerate() {
                    for a in 0..6 {
                        for b in 0..6 {
                            h[6 * na + a][6 * nb + b] += h12[6 * ba + a][6 * bb + b];
                        }
                    }
                }
            }
        }
        for lf in &self.linear {
            scatter_linear_info(&mut h, lf);
        }
        h
    }

    /// Re-base the linearization point: `X_k ← X_k·Exp(step[6k..6k+6])`
    /// for every node, so the next iteration linearizes at `δ = 0`.
    ///
    /// [`LinearFactor`]s are **not** touched: their blocks live in the
    /// chart at the old bases and become stale — rebuild them (exactly as
    /// the marginalization that produced them would be redone at the new
    /// linearization point).
    ///
    /// # Panics
    /// Panics if `step.len() != self.dim()`.
    pub fn re_base(&mut self, step: &[f64]) {
        assert_eq!(step.len(), self.dim(), "step length != 6K");
        for (k, pose) in self.poses.iter_mut().enumerate() {
            let d: Vec6 = std::array::from_fn(|a| step[6 * k + a]);
            *pose = right_update(pose, &d);
        }
    }
}

/// `q = w·LᵀL·r` — the weighted residual pulled back through the whitening.
fn weighted_residual(l: &Mat6, r: &Vec6, w: f64) -> Vec6 {
    let rw = mv(l, r);
    std::array::from_fn(|a| {
        let mut acc = 0.0;
        for k in 0..6 {
            acc += l[k][a] * rw[k];
        }
        w * acc
    })
}

/// `g[node] += Jᵀ·q`.
fn accumulate_gradient_block(g: &mut [f64], node: usize, j: &Mat6, q: &Vec6) {
    for col in 0..6 {
        let mut acc = 0.0;
        for row in 0..6 {
            acc += j[row][col] * q[row];
        }
        g[6 * node + col] += acc;
    }
}

/// `info[na, nb] += w·(L·Ja)ᵀ(L·Jb)` given the pre-whitened `L·J` factors.
fn accumulate_gn_block(
    info: &mut [Vec<f64>],
    na: usize,
    nb: usize,
    lja: &Mat6,
    ljb: &Mat6,
    w: f64,
) {
    for a in 0..6 {
        for b in 0..6 {
            let mut acc = 0.0;
            for k in 0..6 {
                acc += lja[k][a] * ljb[k][b];
            }
            info[6 * na + a][6 * nb + b] += w * acc;
        }
    }
}

/// Scatter a [`LinearFactor`]'s information block into the global matrix.
fn scatter_linear_info(m: &mut [Vec<f64>], lf: &LinearFactor) {
    let k = lf.nodes.len();
    debug_assert_eq!(lf.info.len(), 6 * k);
    for (bi, &na) in lf.nodes.iter().enumerate() {
        for (bj, &nb) in lf.nodes.iter().enumerate() {
            for a in 0..6 {
                for b in 0..6 {
                    m[6 * na + a][6 * nb + b] += lf.info[6 * bi + a][6 * bj + b];
                }
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::nested_ad::{D2, Dual};

    // ── Deterministic RNG (SplitMix64; two-uniform Box–Muller) ──────────

    struct TestRng(u64);
    impl TestRng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /// max |a - b| / max(|b|_maxabs, floor).  Asserts both operands are
    /// finite — `f64::max` silently drops NaN, so without the check a
    /// NaN-poisoned oracle would vanish from both diff and scale and the
    /// comparison would pass.
    fn rel_err_vec(a: &[f64], b: &[f64], floor: f64) -> f64 {
        for (x, y) in a.iter().zip(b) {
            assert!(x.is_finite() && y.is_finite(), "non-finite operand");
        }
        let scale = b.iter().fold(floor, |m, x| m.max(x.abs()));
        a.iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
            / scale
    }

    /// Same finiteness contract as [`rel_err_vec`].
    fn rel_err_mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
        let mut diff = 0.0f64;
        let mut scale = 0.0f64;
        for (ra, rb) in a.iter().zip(b) {
            for (x, y) in ra.iter().zip(rb) {
                assert!(x.is_finite() && y.is_finite(), "non-finite operand");
                diff = diff.max((x - y).abs());
                scale = scale.max(y.abs());
            }
        }
        diff / scale
    }

    fn to_vecmat(h: &[[f64; 12]; 12]) -> Vec<Vec<f64>> {
        h.iter().map(|r| r.to_vec()).collect()
    }

    /// Correlated whitening: diagonal `[100³, 50³]` plus deterministic
    /// lower-triangular cross terms — exercises the full `W = LᵀL` path
    /// (including rotation↔translation coupling) in every oracle test.
    fn correlated_l() -> Mat6 {
        let mut l = diagonal_sqrt_info(&[100.0, 100.0, 100.0, 50.0, 50.0, 50.0]);
        for i in 0..6 {
            for j in 0..i {
                l[i][j] = 4.0 * ((i + 2 * j) % 3) as f64 - 2.0;
            }
        }
        l
    }

    /// One representative factor at the given rotation scale.  `rot_scale = 1`
    /// is the moderate regime; `rot_scale ~ 1e-9` puts every rotation (relative
    /// pose, residual, perturbation) in the fused-basis small-angle branch
    /// while keeping translation residuals finite.
    fn factor_setup(rot_scale: f64) -> (Pose, Pose, Pose, Mat6) {
        let base_i = Pose::exp(&[0.2, -0.1, 0.3, 0.5, -0.2, 0.8]);
        let step = [
            0.30 * rot_scale,
            -0.20 * rot_scale,
            0.25 * rot_scale,
            0.9,
            0.1,
            -0.2,
        ];
        let base_j = base_i.compose(&Pose::exp(&step));
        // Measurement offset: finite translation error, rotation error at the
        // same scale as the relative rotation.
        let z_off = [
            0.02 * rot_scale,
            0.01 * rot_scale,
            -0.03 * rot_scale,
            -0.05,
            0.03,
            0.04,
        ];
        let z = Pose::exp(&step).compose(&Pose::exp(&z_off));
        (base_i, base_j, z, correlated_l())
    }

    /// Stacked perturbation with rotations at `rot_scale`.
    fn d_setup(rot_scale: f64) -> [f64; 12] {
        [
            0.08 * rot_scale,
            -0.05 * rot_scale,
            0.11 * rot_scale,
            0.07,
            -0.04,
            0.09,
            -0.06 * rot_scale,
            0.10 * rot_scale,
            0.03 * rot_scale,
            -0.08,
            0.05,
            -0.02,
        ]
    }

    fn lift_pose<T: AD>(p: &Pose) -> PoseG<T> {
        pose_to_g::<T>(p)
    }

    fn lift_mat6<T: AD>(m: &Mat6) -> Mat6G<T> {
        std::array::from_fn(|a| std::array::from_fn(|b| T::constant(m[a][b])))
    }

    fn nll_f64(d: &[f64; 12], bi: &Pose, bj: &Pose, z: &Pose, l: &Mat6, kappa: Option<f64>) -> f64 {
        between_nll_g::<f64>(d, &lift_pose(bi), &lift_pose(bj), &lift_pose(z), l, kappa)
    }

    fn grad_f64(
        d: &[f64; 12],
        bi: &Pose,
        bj: &Pose,
        z: &Pose,
        l: &Mat6,
        kappa: Option<f64>,
    ) -> [f64; 12] {
        between_gradient_g::<f64>(d, &lift_pose(bi), &lift_pose(bj), &lift_pose(z), l, kappa)
    }

    // ── (a) analytical gradient vs central FD of the NLL ────────────────

    fn check_gradient_vs_fd(rot_scale: f64, kappa: Option<f64>, d0: &[f64; 12], tol: f64) {
        let (bi, bj, z, l) = factor_setup(rot_scale);
        let g = grad_f64(d0, &bi, &bj, &z, &l, kappa);
        let h = 1e-6;
        let mut g_fd = [0.0f64; 12];
        for k in 0..12 {
            let mut dp = *d0;
            dp[k] += h;
            let mut dm = *d0;
            dm[k] -= h;
            g_fd[k] = (nll_f64(&dp, &bi, &bj, &z, &l, kappa)
                - nll_f64(&dm, &bi, &bj, &z, &l, kappa))
                / (2.0 * h);
        }
        for v in &g {
            assert!(v.is_finite(), "NaN/inf in analytical gradient");
        }
        let rel = rel_err_vec(&g, &g_fd, 1.0);
        assert!(rel < tol, "gradient vs FD rel = {rel:.3e} (tol {tol:.0e})");
    }

    #[test]
    fn between_gradient_matches_fd_at_zero() {
        check_gradient_vs_fd(1.0, None, &[0.0; 12], 1e-6);
        check_gradient_vs_fd(1.0, Some(3.0), &[0.0; 12], 1e-6);
    }

    #[test]
    fn between_gradient_matches_fd_at_nonzero_d() {
        let d = d_setup(1.0);
        check_gradient_vs_fd(1.0, None, &d, 1e-6);
        check_gradient_vs_fd(1.0, Some(3.0), &d, 1e-6);
    }

    // ── (b) seeded Hessian vs central FD of the analytical gradient ─────

    fn check_hessian_vs_fd_of_grad(rot_scale: f64, kappa: Option<f64>) {
        let (bi, bj, z, l) = factor_setup(rot_scale);
        let h_seed = between_hessian_seeded(&bi, &bj, &z, &l, kappa);
        let h = 1e-5;
        let mut h_fd = [[0.0f64; 12]; 12];
        for k in 0..12 {
            let mut dp = [0.0f64; 12];
            dp[k] = h;
            let mut dm = [0.0f64; 12];
            dm[k] = -h;
            let gp = grad_f64(&dp, &bi, &bj, &z, &l, kappa);
            let gm = grad_f64(&dm, &bi, &bj, &z, &l, kappa);
            for row in 0..12 {
                h_fd[row][k] = (gp[row] - gm[row]) / (2.0 * h);
            }
        }
        let rel = rel_err_mat(&to_vecmat(&h_seed), &to_vecmat(&h_fd));
        assert!(
            rel < 1e-4,
            "seeded Hessian vs FD-of-gradient rel = {rel:.3e}"
        );
    }

    #[test]
    fn between_hessian_seeded_matches_fd_of_gradient() {
        check_hessian_vs_fd_of_grad(1.0, None);
        check_hessian_vs_fd_of_grad(1.0, Some(3.0));
    }

    // ── (c) seeded Hessian vs nested-dual D2<12> oracle ──────────────────

    fn d2_12_hessian(
        bi: &Pose,
        bj: &Pose,
        z: &Pose,
        l: &Mat6,
        kappa: Option<f64>,
    ) -> Vec<Vec<f64>> {
        type T = D2<12>;
        let d: [T; 12] = std::array::from_fn(|k| {
            let inner = Dual::<f64, 12>::seed(0.0, k);
            T::seed(inner, k)
        });
        let r = between_nll_g::<T>(
            &d,
            &lift_pose(bi),
            &lift_pose(bj),
            &lift_pose(z),
            &lift_mat6(l),
            kappa,
        );
        (0..12)
            .map(|q| (0..12).map(|s| r.tangent[q].tangent[s]).collect())
            .collect()
    }

    fn check_hessian_vs_d2(rot_scale: f64, kappa: Option<f64>) {
        let (bi, bj, z, l) = factor_setup(rot_scale);
        let h_seed = between_hessian_seeded(&bi, &bj, &z, &l, kappa);
        let h_d2 = d2_12_hessian(&bi, &bj, &z, &l, kappa);
        for row in &h_seed {
            for v in row {
                assert!(v.is_finite(), "NaN/inf in seeded Hessian");
            }
        }
        let rel = rel_err_mat(&to_vecmat(&h_seed), &h_d2);
        assert!(
            rel < 1e-12,
            "seeded Hessian vs D2<12> oracle rel = {rel:.3e}"
        );
    }

    #[test]
    fn between_hessian_seeded_matches_d2_oracle() {
        check_hessian_vs_d2(1.0, None);
        check_hessian_vs_d2(1.0, Some(3.0));
    }

    // ── K = 6 instance (the posegraph_factorlocal_check.py protocol) ────

    const K: usize = 6;
    const CLOSURES: [(usize, usize); 4] = [(0, 3), (1, 4), (2, 5), (0, 5)];

    /// Truth chain + noisy measurements + dead-reckoned bases, one spurious
    /// closure (index 1).  All rotational magnitudes scale with `rot_scale`.
    /// Closure factors get a correlated whitening (off-diagonal `L`) so the
    /// global oracle also certifies the non-diagonal path.
    fn build_k6(rot_scale: f64) -> GraphProblem {
        let mut rng = TestRng(7);
        let sig_anchor: Vec6 = sig6(0.01 * rot_scale, 0.02);
        let sig_odo: Vec6 = sig6(0.01 * rot_scale, 0.02);
        let sig_clo: Vec6 = sig6(0.01 * rot_scale, 0.02);
        let sig_step: Vec6 = sig6(0.05 * rot_scale, 0.10);
        let spur_sig: Vec6 = sig6(0.3 * rot_scale, 0.5);
        let mean_step: Vec6 = [0.0, 0.0, 0.35 * rot_scale, 1.0, 0.0, 0.0];

        // Correlated closure whitening: diag(1/σ) + lower-triangular cross
        // terms at ~5% of the diagonal scale.
        let mut l_clo = diagonal_sqrt_info(&inv6(&sig_clo));
        for i in 0..6 {
            for j in 0..i {
                l_clo[i][j] = 2.5 * ((i + j) % 3) as f64;
            }
        }

        let x_anc = Pose::identity();
        let mut truth = Vec::with_capacity(K);
        truth.push(x_anc.compose(&Pose::exp(&draw(&mut rng, &sig_anchor))));
        for k in 0..K - 1 {
            let mut u = draw(&mut rng, &sig_step);
            for a in 0..6 {
                u[a] += mean_step[a];
            }
            truth.push(truth[k].compose(&Pose::exp(&u)));
        }

        let mut factors = Vec::new();
        let mut z_odo = Vec::with_capacity(K - 1);
        for k in 0..K - 1 {
            let rel = truth[k].inverse().compose(&truth[k + 1]);
            let z = rel.compose(&Pose::exp(&draw(&mut rng, &sig_odo)));
            z_odo.push(z);
            factors.push(BetweenFactor {
                i: k,
                j: k + 1,
                z,
                sqrt_info: diagonal_sqrt_info(&inv6(&sig_odo)),
                kappa: None,
            });
        }
        for (c, &(i, j)) in CLOSURES.iter().enumerate() {
            let rel = truth[i].inverse().compose(&truth[j]);
            let sig = if c == 1 { &spur_sig } else { &sig_clo };
            let z = rel.compose(&Pose::exp(&draw(&mut rng, sig)));
            factors.push(BetweenFactor {
                i,
                j,
                z,
                sqrt_info: l_clo,
                kappa: Some(3.0),
            });
        }

        // Dead-reckoned bases from the odometry chain.
        let mut poses = Vec::with_capacity(K);
        poses.push(x_anc);
        for k in 0..K - 1 {
            poses.push(poses[k].compose(&z_odo[k]));
        }

        GraphProblem {
            poses,
            priors: vec![PriorFactor {
                node: 0,
                x_ref: x_anc,
                sqrt_info: diagonal_sqrt_info(&inv6(&sig_anchor)),
            }],
            factors,
            linear: vec![],
        }
    }

    fn sig6(rot: f64, trans: f64) -> Vec6 {
        [rot, rot, rot, trans, trans, trans]
    }
    fn inv6(sig: &Vec6) -> Vec6 {
        std::array::from_fn(|a| 1.0 / sig[a])
    }
    fn draw(rng: &mut TestRng, sig: &Vec6) -> Vec6 {
        std::array::from_fn(|a| sig[a] * rng.normal())
    }

    /// Total graph NLL as a `T`-generic function of the full 6K-dim
    /// perturbation — the independent global path for the oracle.
    fn graph_nll_oracle<T: AD>(p: &GraphProblem, delta: &[T]) -> T {
        let poses_g: Vec<PoseG<T>> = p
            .poses
            .iter()
            .enumerate()
            .map(|(k, x)| {
                let d: Vec6G<T> = std::array::from_fn(|a| delta[6 * k + a]);
                pose_to_g::<T>(x).compose(&PoseG::exp(&d))
            })
            .collect();
        let mut tot = T::constant(0.0);
        for pr in &p.priors {
            let r = poses_g[pr.node]
                .inverse()
                .compose(&pose_to_g::<T>(&pr.x_ref))
                .log();
            let (s, _) = whitened_square_g(&lift_mat6(&pr.sqrt_info), &r);
            tot += T::constant(0.5) * s;
        }
        for f in &p.factors {
            let x_rel = poses_g[f.i].inverse().compose(&poses_g[f.j]);
            let e = pose_to_g::<T>(&f.z).inverse().compose(&x_rel);
            let r = e.log();
            let (s, _) = whitened_square_g(&lift_mat6(&f.sqrt_info), &r);
            tot += match f.kappa {
                None => T::constant(0.5) * s,
                Some(k) => pseudo_huber(s, T::constant(k * k)),
            };
        }
        tot
    }

    // ── (d) factor-local assembly vs global D2<36> oracle ────────────────

    fn check_graph_vs_global_oracle(rot_scale: f64) {
        // D2<36> temporaries are ~131 KB per pose; run on a fat stack so the
        // 2 MB default test-thread stack can't overflow.
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(move || {
                let p = build_k6(rot_scale);
                let dim = p.dim();

                type T = D2<36>;
                let delta: Vec<T> = (0..dim)
                    .map(|k| T::seed(Dual::<f64, 36>::seed(0.0, k), k))
                    .collect();
                let res = graph_nll_oracle::<T>(&p, &delta);

                let h_global: Vec<Vec<f64>> = (0..dim)
                    .map(|q| (0..dim).map(|s| res.tangent[q].tangent[s]).collect())
                    .collect();
                let g_global: Vec<f64> = (0..dim).map(|q| res.tangent[q].value).collect();

                let h_local = p.exact_hessian();
                for row in &h_local {
                    for v in row {
                        assert!(v.is_finite(), "NaN/inf in exact_hessian");
                    }
                }
                let rel = rel_err_mat(&h_local, &h_global);
                assert!(
                    rel <= 1e-12,
                    "exact_hessian vs global D2<36> oracle rel = {rel:.3e}"
                );

                // The graph gradient must match the oracle's first-order
                // tangents (validates robust weights + scatter).
                let g = p.gradient();
                let rel_g = rel_err_vec(&g, &g_global, 1.0);
                assert!(rel_g <= 1e-12, "gradient vs oracle rel = {rel_g:.3e}");
            })
            .expect("spawn oracle thread")
            .join()
            .expect("oracle thread panicked");
    }

    #[test]
    fn exact_hessian_matches_global_d2_oracle() {
        check_graph_vs_global_oracle(1.0);
    }

    // ── (e) zero-residual sanity ─────────────────────────────────────────

    #[test]
    fn prior_gradient_zero_at_reference() {
        let x_ref = Pose::exp(&[0.3, -0.2, 0.4, 1.0, -0.5, 0.7]);
        let p = GraphProblem {
            poses: vec![x_ref],
            priors: vec![PriorFactor {
                node: 0,
                x_ref,
                sqrt_info: correlated_l(),
            }],
            factors: vec![],
            linear: vec![],
        };
        let g = p.gradient();
        for v in &g {
            assert!(v.abs() < 1e-10, "prior gradient at reference: {v:.3e}");
        }
    }

    #[test]
    fn between_residual_zero_at_exact_measurement() {
        let (bi, bj, _, l) = factor_setup(1.0);
        let z_exact = bi.inverse().compose(&bj);
        let lin = between_linearize(&bi, &bj, &z_exact, &l, Some(3.0));
        for a in 0..6 {
            assert!(lin.r[a].abs() < 1e-14, "residual[{a}] = {:.3e}", lin.r[a]);
        }
        assert!(
            (lin.w - 1.0).abs() < 1e-14,
            "robust weight at zero residual"
        );
    }

    // ── LinearFactor ─────────────────────────────────────────────────────

    /// A Gaussian between factor, re-expressed as a `LinearFactor` from its
    /// own linearization, must reproduce the graph gradient and GN
    /// information exactly (same formulas, different assembly path).
    #[test]
    fn linear_factor_matches_linearized_between() {
        let (bi, bj, z, l) = factor_setup(1.0);
        let factor = BetweenFactor {
            i: 0,
            j: 1,
            z,
            sqrt_info: l,
            kappa: None,
        };

        let p_between = GraphProblem {
            poses: vec![bi, bj],
            priors: vec![],
            factors: vec![factor],
            linear: vec![],
        };

        // Build the equivalent dense blocks from the linearization.
        let lin = between_linearize(&bi, &bj, &z, &l, None);
        let q = weighted_residual(&l, &lin.r, 1.0);
        let mut grad = vec![0.0f64; 12];
        accumulate_gradient_block(&mut grad[..], 0, &lin.j_i, &q);
        accumulate_gradient_block(&mut grad[..], 1, &lin.j_j, &q);
        let lji = mm(&l, &lin.j_i);
        let ljj = mm(&l, &lin.j_j);
        let mut info = vec![vec![0.0f64; 12]; 12];
        accumulate_gn_block(&mut info, 0, 0, &lji, &lji, 1.0);
        accumulate_gn_block(&mut info, 1, 1, &ljj, &ljj, 1.0);
        accumulate_gn_block(&mut info, 0, 1, &lji, &ljj, 1.0);
        accumulate_gn_block(&mut info, 1, 0, &ljj, &lji, 1.0);

        let p_linear = GraphProblem {
            poses: vec![bi, bj],
            priors: vec![],
            factors: vec![],
            linear: vec![LinearFactor {
                nodes: vec![0, 1],
                grad,
                info,
            }],
        };

        let g_a = p_between.gradient();
        let g_b = p_linear.gradient();
        assert!(
            rel_err_vec(&g_b, &g_a, 1.0) < 1e-14,
            "gradient: linear-factor path diverges from between path"
        );
        let i_a = p_between.gn_information();
        let i_b = p_linear.gn_information();
        assert!(
            rel_err_mat(&i_b, &i_a) < 1e-14,
            "gn_information: linear-factor path diverges from between path"
        );

        // A LinearFactor is exactly quadratic: its exact Hessian IS its
        // information (unlike the between factor, whose exact Hessian
        // carries curvature terms on top).
        let h_b = p_linear.exact_hessian();
        assert!(rel_err_mat(&h_b, &i_b) < 1e-15);
    }

    /// Multi-node scatter guard: two overlapping factors (3-node and
    /// 2-node, sharing node 2, on a K = 4 graph) must sum into the global
    /// matrix exactly like a hand-scattered dense construction.
    #[test]
    fn linear_factor_multinode_scatter() {
        let mut rng = TestRng(99);
        let make = |rng: &mut TestRng, nodes: Vec<usize>| {
            let d = 6 * nodes.len();
            let grad: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
            // Symmetric info block.
            let mut info = vec![vec![0.0f64; d]; d];
            for a in 0..d {
                for b in 0..=a {
                    let v = rng.normal();
                    info[a][b] = v;
                    info[b][a] = v;
                }
            }
            LinearFactor { nodes, grad, info }
        };
        let lf_a = make(&mut rng, vec![0, 2, 3]);
        let lf_b = make(&mut rng, vec![2, 1]);

        let poses: Vec<Pose> = (0..4)
            .map(|k| Pose::exp(&[0.1 * k as f64, 0.0, -0.05, 1.0, 0.0, 0.5]))
            .collect();
        let p = GraphProblem {
            poses,
            priors: vec![],
            factors: vec![],
            linear: vec![lf_a.clone(), lf_b.clone()],
        };

        // Hand-scattered reference.
        let dim = 24;
        let mut g_ref = vec![0.0f64; dim];
        let mut m_ref = vec![vec![0.0f64; dim]; dim];
        for lf in [&lf_a, &lf_b] {
            for (bi, &na) in lf.nodes.iter().enumerate() {
                for a in 0..6 {
                    g_ref[6 * na + a] += lf.grad[6 * bi + a];
                }
                for (bj, &nb) in lf.nodes.iter().enumerate() {
                    for a in 0..6 {
                        for b in 0..6 {
                            m_ref[6 * na + a][6 * nb + b] += lf.info[6 * bi + a][6 * bj + b];
                        }
                    }
                }
            }
        }

        assert!(rel_err_vec(&p.gradient(), &g_ref, 1.0) < 1e-15);
        assert!(rel_err_mat(&p.gn_information(), &m_ref) < 1e-15);
        assert!(rel_err_mat(&p.exact_hessian(), &m_ref) < 1e-15);
    }

    // ── (f) small-angle regime: a–d at ‖relative rotation‖ ~ 1e-9 ───────
    //
    // The fused-basis regime: every Log/Exp/Jr evaluation sits in the
    // θ → 0 Taylor branch while translation residuals stay finite.
    // A NaN anywhere here is a regression.

    const TINY: f64 = 1e-9;

    #[test]
    fn small_angle_gradient_matches_fd() {
        check_gradient_vs_fd(TINY, None, &[0.0; 12], 1e-6);
        check_gradient_vs_fd(TINY, Some(3.0), &[0.0; 12], 1e-6);
        let d = d_setup(TINY);
        check_gradient_vs_fd(TINY, None, &d, 1e-6);
        check_gradient_vs_fd(TINY, Some(3.0), &d, 1e-6);
    }

    #[test]
    fn small_angle_hessian_matches_fd_of_gradient() {
        check_hessian_vs_fd_of_grad(TINY, None);
        check_hessian_vs_fd_of_grad(TINY, Some(3.0));
    }

    #[test]
    fn small_angle_hessian_matches_d2_oracle() {
        check_hessian_vs_d2(TINY, None);
        check_hessian_vs_d2(TINY, Some(3.0));
    }

    #[test]
    fn small_angle_exact_hessian_matches_global_d2_oracle() {
        check_graph_vs_global_oracle(TINY);
    }
}
