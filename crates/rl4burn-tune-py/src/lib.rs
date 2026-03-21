use pyo3::prelude::*;
use rl4burn_tune::{
    calculate_order as rs_calculate_order, default_gamma as rs_default_gamma,
    default_weights as rs_default_weights, gaussian_log_pdf as rs_gaussian_log_pdf,
    hyperopt_default_gamma as rs_hyperopt_default_gamma, CategoricalParzenEstimator,
    ParzenEstimator,
};
use rl4burn_tune::distributions::{
    int_transform_from_internal, int_transform_to_internal, transform_from_internal,
    transform_to_internal, FloatDistribution, IntDistribution,
};

// --- Gamma and weight functions ---

#[pyfunction]
fn default_gamma(n: usize) -> usize {
    rs_default_gamma(n)
}

#[pyfunction]
fn hyperopt_default_gamma(n: usize) -> usize {
    rs_hyperopt_default_gamma(n)
}

#[pyfunction]
fn default_weights(n: usize) -> Vec<f64> {
    rs_default_weights(n)
}

// --- Ordering ---

#[pyfunction]
fn calculate_order(values: Vec<f64>) -> Vec<usize> {
    rs_calculate_order(&values)
}

// --- Gaussian log PDF ---

#[pyfunction]
fn gaussian_log_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    rs_gaussian_log_pdf(x, mu, sigma)
}

// --- ParzenEstimator ---

/// Build a ParzenEstimator and return (mus, sigmas, weights).
#[pyfunction]
#[pyo3(signature = (observations, low, high, prior_weight, consider_magic_clip, consider_endpoints, weights))]
fn parzen_estimator_new(
    observations: Vec<f64>,
    low: f64,
    high: f64,
    prior_weight: f64,
    consider_magic_clip: bool,
    consider_endpoints: bool,
    weights: Vec<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let pe = ParzenEstimator::new(
        &observations,
        low,
        high,
        prior_weight,
        consider_magic_clip,
        consider_endpoints,
        &weights,
    );
    (pe.mus, pe.sigmas, pe.weights)
}

/// Evaluate log_pdf for a ParzenEstimator at multiple x values.
#[pyfunction]
#[pyo3(signature = (mus, sigmas, weights, low, high, x_values))]
fn parzen_estimator_log_pdf(
    mus: Vec<f64>,
    sigmas: Vec<f64>,
    weights: Vec<f64>,
    low: f64,
    high: f64,
    x_values: Vec<f64>,
) -> Vec<f64> {
    let pe = ParzenEstimator {
        mus,
        sigmas,
        weights,
        low,
        high,
    };
    x_values.iter().map(|&x| pe.log_pdf(x)).collect()
}

// --- CategoricalParzenEstimator ---

/// Build a CategoricalParzenEstimator and return (component_weights, mixture_weights).
#[pyfunction]
#[pyo3(signature = (observations, n_choices, prior_weight, weights))]
fn categorical_parzen_estimator_new(
    observations: Vec<usize>,
    n_choices: usize,
    prior_weight: f64,
    weights: Vec<f64>,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let cpe =
        CategoricalParzenEstimator::new(&observations, n_choices, prior_weight, &weights);
    (cpe.component_weights, cpe.mixture_weights)
}

/// Evaluate log_pdf for a CategoricalParzenEstimator at each index.
#[pyfunction]
#[pyo3(signature = (component_weights, mixture_weights, n_choices, indices))]
fn categorical_parzen_estimator_log_pdf(
    component_weights: Vec<Vec<f64>>,
    mixture_weights: Vec<f64>,
    n_choices: usize,
    indices: Vec<usize>,
) -> Vec<f64> {
    let cpe = CategoricalParzenEstimator {
        component_weights,
        mixture_weights,
        n_choices,
    };
    indices.iter().map(|&i| cpe.log_pdf(i)).collect()
}

// --- Transform functions ---

#[pyfunction]
#[pyo3(signature = (value, low, high, log))]
fn float_transform_to_internal(value: f64, low: f64, high: f64, log: bool) -> f64 {
    let dist = FloatDistribution {
        low,
        high,
        log,
        step: None,
    };
    transform_to_internal(value, &dist)
}

#[pyfunction]
#[pyo3(signature = (internal, low, high, log, step=None))]
fn float_transform_from_internal(
    internal: f64,
    low: f64,
    high: f64,
    log: bool,
    step: Option<f64>,
) -> f64 {
    let dist = FloatDistribution {
        low,
        high,
        log,
        step,
    };
    transform_from_internal(internal, &dist)
}

#[pyfunction]
#[pyo3(signature = (value, low, high, log))]
fn int_transform_to_internal_py(value: i64, low: i64, high: i64, log: bool) -> f64 {
    let dist = IntDistribution {
        low,
        high,
        log,
        step: None,
    };
    int_transform_to_internal(value, &dist)
}

#[pyfunction]
#[pyo3(signature = (internal, low, high, log, step=None))]
fn int_transform_from_internal_py(
    internal: f64,
    low: i64,
    high: i64,
    log: bool,
    step: Option<i64>,
) -> i64 {
    let dist = IntDistribution {
        low,
        high,
        log,
        step,
    };
    int_transform_from_internal(internal, &dist)
}

#[pymodule]
fn rl4burn_tune_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(default_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(hyperopt_default_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(default_weights, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_order, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_log_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(parzen_estimator_new, m)?)?;
    m.add_function(wrap_pyfunction!(parzen_estimator_log_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(categorical_parzen_estimator_new, m)?)?;
    m.add_function(wrap_pyfunction!(categorical_parzen_estimator_log_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(float_transform_to_internal, m)?)?;
    m.add_function(wrap_pyfunction!(float_transform_from_internal, m)?)?;
    m.add_function(wrap_pyfunction!(int_transform_to_internal_py, m)?)?;
    m.add_function(wrap_pyfunction!(int_transform_from_internal_py, m)?)?;
    Ok(())
}
