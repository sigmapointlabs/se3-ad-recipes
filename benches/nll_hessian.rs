//! Cost comparison: SE(3) NLL Hessian computed five ways.
//!
//! All five rows of Table II benchmarked on the same problem instance
//! (`build_problem(κ = 0.05)`), so the timings are directly comparable.
//! The diagnostic numbers are reported by the test
//! `nll_hessian_four_ways_table_ii` in `nll_tests.rs`; see the README of
//! `nll_bench` for the trait/impl details.
//!
//!   1. `value_fd`              FD of value (no AD), 4-point stencil
//!   2. `mixed_ad_fd_grad`      FD of `adfn<6>` gradient (12 grad evals)
//!   3. `nested_d2_naive`       depth-2 nested AD, naïve θ-Taylor basis
//!   4. `nested_d2_fixed`       depth-2 nested AD, production s-basis
//!   5. `gradient_adfn6`        single `adfn<6>` gradient (cost reference)
//!
//! Method (5) is the "AD of seeded gradient" reference — *not* a Hessian
//! method by itself.  Run `cargo bench --bench nll_hessian` to refresh.

use criterion::{Criterion, criterion_group, criterion_main};
use se3_ad_recipes::Vec6;
use se3_ad_recipes::nll_bench::{
    FixedBasis, NaiveBasis, build_problem, hessian_ad_of_analytical_grad, hessian_d2,
    hessian_fd_analytical_grad, hessian_fd_grad, hessian_fd_value, hessian_for, nll_f64,
    nll_gradient, nll_gradient_analytical_g,
};

fn bench_nll_value(c: &mut Criterion) {
    let p = build_problem(0.05);
    let delta: Vec6 = [0.0; 6];
    c.bench_function("nll_hessian/value_f64", |b| {
        b.iter(|| nll_f64::<FixedBasis>(&p, &delta))
    });
}

fn bench_nll_gradient(c: &mut Criterion) {
    let p = build_problem(0.05);
    let delta: Vec6 = [0.0; 6];
    c.bench_function("nll_hessian/gradient_adfn6", |b| {
        b.iter(|| nll_gradient::<FixedBasis>(&p, &delta))
    });
}

fn bench_hessian_value_fd(c: &mut Criterion) {
    let p = build_problem(0.05);
    c.bench_function("nll_hessian/value_fd (1+12+24 NLL evals)", |b| {
        b.iter(|| hessian_fd_value::<FixedBasis>(&p, 1e-3))
    });
}

fn bench_hessian_mixed_ad_fd(c: &mut Criterion) {
    let p = build_problem(0.05);
    c.bench_function("nll_hessian/mixed_ad_fd_grad (12 adfn6 grad evals)", |b| {
        b.iter(|| hessian_fd_grad::<FixedBasis>(&p, 1e-5))
    });
}

fn bench_hessian_nested_d2_naive(c: &mut Criterion) {
    let p = build_problem(0.05);
    c.bench_function("nll_hessian/nested_d2_naive (1 D2<6> eval)", |b| {
        b.iter(|| hessian_d2::<NaiveBasis>(&p))
    });
}

fn bench_hessian_nested_d2_fixed(c: &mut Criterion) {
    let p = build_problem(0.05);
    c.bench_function("nll_hessian/nested_d2_fixed (1 D2<6> eval)", |b| {
        b.iter(|| hessian_d2::<FixedBasis>(&p))
    });
}

fn bench_nll_gradient_analytical(c: &mut Criterion) {
    let p = build_problem(0.05);
    let delta: [f64; 6] = [0.0; 6];
    c.bench_function("nll_hessian/gradient_analytical_f64", |b| {
        b.iter(|| nll_gradient_analytical_g::<f64>(&p, &delta))
    });
}

fn bench_hessian_fd_analytical_grad(c: &mut Criterion) {
    let p = build_problem(0.05);
    c.bench_function(
        "nll_hessian/fd_analytical_grad_fused (12 f64 analytical-grad evals)",
        |b| b.iter(|| hessian_fd_analytical_grad(&p, 1e-5)),
    );
}

fn bench_hessian_ad_of_analytical_grad(c: &mut Criterion) {
    let p = build_problem(0.05);
    c.bench_function(
        "nll_hessian/ad_seeded_analytical_grad_fused (1 adfn<6> eval)",
        |b| b.iter(|| hessian_ad_of_analytical_grad(&p)),
    );
}

fn bench_hessian_auto_for(c: &mut Criterion) {
    let p = build_problem(0.05);
    c.bench_function(
        "nll_hessian/auto_for_dual_adr_adfn6 (1 fwd + 1 rev pass)",
        |b| b.iter(|| hessian_for::<FixedBasis>(&p)),
    );
}

criterion_group!(
    benches,
    bench_nll_value,
    bench_nll_gradient,
    bench_nll_gradient_analytical,
    bench_hessian_value_fd,
    bench_hessian_mixed_ad_fd,
    bench_hessian_nested_d2_naive,
    bench_hessian_nested_d2_fixed,
    bench_hessian_fd_analytical_grad,
    bench_hessian_ad_of_analytical_grad,
    bench_hessian_auto_for,
);
criterion_main!(benches);
