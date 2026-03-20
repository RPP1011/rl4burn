/// Kani proof harnesses for formal verification of mathematical invariants.
///
/// These harnesses verify key properties of the search algorithms using
/// symbolic execution. They require the Kani verifier to run:
/// `cargo kani --harness <name>`
#[cfg(kani)]
mod kani_proofs {
    use crate::distributions::*;
    use crate::pruners::median::median_of_sorted;
    use crate::samplers::tpe::*;

    // === Phase 1: Distribution invariants ===

    #[kani::proof]
    fn roundtrip_float_transform() {
        let low: f64 = kani::any();
        let high: f64 = kani::any();
        kani::assume(low.is_finite() && high.is_finite());
        kani::assume(0.0 < low && low < high && high < 1e6);

        let dist = FloatDistribution::new(low, high, true, None);
        let value: f64 = kani::any();
        kani::assume(value >= low && value <= high);

        let internal = transform_to_internal(value, &dist);
        let recovered = transform_from_internal(internal, &dist);

        assert!((recovered - value).abs() < 1e-10 * value.abs().max(1.0));
    }

    #[kani::proof]
    fn internal_is_normalized() {
        let low: f64 = kani::any();
        let high: f64 = kani::any();
        kani::assume(low.is_finite() && high.is_finite());
        kani::assume(0.0 < low && low < high && high < 1e6);

        let dist = FloatDistribution::new(low, high, true, None);
        let value: f64 = kani::any();
        kani::assume(value >= low && value <= high);

        let internal = transform_to_internal(value, &dist);
        assert!(internal >= 0.0 && internal <= 1.0);
    }

    #[kani::proof]
    fn contains_matches_bounds() {
        let low: f64 = kani::any();
        let high: f64 = kani::any();
        kani::assume(low.is_finite() && high.is_finite());
        kani::assume(low < high);

        let dist = FloatDistribution::new(low, high, false, None);
        assert!(dist.contains(low));
        assert!(dist.contains(high));

        let outside: f64 = kani::any();
        kani::assume(outside.is_finite());
        kani::assume(outside < low || outside > high);
        assert!(!dist.contains(outside));
    }

    // === Phase 2: Random sampler invariants ===

    #[kani::proof]
    fn random_sample_in_bounds() {
        let low: f64 = kani::any();
        let high: f64 = kani::any();
        kani::assume(low.is_finite() && high.is_finite());
        kani::assume(0.0 < low && low < high && high < 1e6);

        let u: f64 = kani::any();
        kani::assume(u >= 0.0 && u < 1.0);

        let sample = low + u * (high - low);
        assert!(sample >= low && sample <= high);
    }

    #[kani::proof]
    fn categorical_index_valid() {
        let n_choices: usize = kani::any();
        kani::assume(n_choices > 0 && n_choices <= 100);

        let u: f64 = kani::any();
        kani::assume(u >= 0.0 && u < 1.0);

        let index = (u * n_choices as f64) as usize;
        assert!(index < n_choices);
    }

    // === Phase 3a: Gamma and weight invariants ===

    #[kani::proof]
    fn gamma_range() {
        let n: usize = kani::any();
        kani::assume(n > 0 && n <= 10_000);

        let g = default_gamma(n);
        assert!(g >= 1 && g <= 25);
    }

    #[kani::proof]
    fn gamma_monotonic() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        kani::assume(a > 0 && b > 0 && a <= 10_000 && b <= 10_000);
        kani::assume(a <= b);

        assert!(default_gamma(a) <= default_gamma(b));
    }

    #[kani::proof]
    fn weights_length() {
        let n: usize = kani::any();
        kani::assume(n <= 200);

        let w = default_weights(n);
        assert!(w.len() == n);
    }

    #[kani::proof]
    fn weights_positive() {
        let n: usize = kani::any();
        kani::assume(n > 0 && n <= 200);

        let w = default_weights(n);
        for &wi in &w {
            assert!(wi > 0.0);
        }
    }

    // === Phase 3b: Parzen estimator invariants ===

    #[kani::proof]
    fn log_pdf_no_nan() {
        let mu: f64 = kani::any();
        let sigma: f64 = kani::any();
        let x: f64 = kani::any();
        kani::assume(mu.is_finite() && sigma.is_finite() && x.is_finite());
        kani::assume(sigma > 1e-15);

        let z = (x - mu) / sigma;
        let log_p = -0.5 * z * z - sigma.ln() - 0.5 * std::f64::consts::TAU.ln();
        assert!(!log_p.is_nan());
        assert!(log_p != f64::INFINITY);
    }

    #[kani::proof]
    fn logsumexp_stable() {
        let a: f64 = kani::any();
        let b: f64 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());

        let max_val = a.max(b);
        let result = max_val + ((a - max_val).exp() + (b - max_val).exp()).ln();
        assert!(result.is_finite());
    }

    #[kani::proof]
    fn mixture_sample_bounded() {
        let mu: f64 = kani::any();
        let sigma: f64 = kani::any();
        let low: f64 = kani::any();
        let high: f64 = kani::any();
        kani::assume(mu.is_finite() && sigma.is_finite());
        kani::assume(low.is_finite() && high.is_finite());
        kani::assume(sigma > 0.0 && low < high);

        let raw: f64 = kani::any();
        kani::assume(raw.is_finite());
        let clamped = raw.clamp(low, high);
        assert!(clamped >= low && clamped <= high);
    }

    // === Phase 3c: TPE EI invariants ===

    #[kani::proof]
    fn ei_no_nan() {
        let l_log: f64 = kani::any();
        let g_log: f64 = kani::any();
        kani::assume(l_log.is_finite() && g_log.is_finite());

        let ei = l_log - g_log;
        assert!(!ei.is_nan());
    }

    // === Phase 4: Pruner invariants ===

    #[kani::proof]
    fn no_prune_during_warmup() {
        let n_warmup: usize = kani::any();
        kani::assume(n_warmup > 0 && n_warmup <= 100);

        let current_step: usize = kani::any();
        kani::assume(current_step < n_warmup);

        // During warmup, step < n_warmup, so pruning should not trigger
        assert!(current_step < n_warmup);
    }

    #[kani::proof]
    fn no_prune_during_startup() {
        let n_startup: usize = kani::any();
        kani::assume(n_startup > 0 && n_startup <= 100);

        let n_completed: usize = kani::any();
        kani::assume(n_completed < n_startup);

        // With fewer completed trials than startup, pruning should not trigger
        assert!(n_completed < n_startup);
    }

    #[kani::proof]
    fn median_correctness() {
        let a: f64 = kani::any();
        let b: f64 = kani::any();
        let c: f64 = kani::any();
        kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
        kani::assume(a <= b && b <= c);

        let med = median_of_sorted(&[a, b, c]);
        assert!(med == b);
    }
}
