use std::collections::HashMap;

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

// ============================================================
// Python wrapper classes for Study, Trial, Samplers, Pruners
// ============================================================

/// Python wrapper for Study.
#[pyclass(name = "Study")]
struct PyStudy {
    inner: rl4burn_tune::Study,
}

#[pymethods]
impl PyStudy {
    /// Create a new single-objective study.
    /// direction: "minimize" or "maximize"
    #[new]
    #[pyo3(signature = (direction = "minimize"))]
    fn new(direction: &str) -> PyResult<Self> {
        let dir = match direction {
            "minimize" => rl4burn_tune::study::Direction::Minimize,
            "maximize" => rl4burn_tune::study::Direction::Maximize,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "direction must be 'minimize' or 'maximize'",
                ))
            }
        };
        Ok(Self {
            inner: rl4burn_tune::Study::new(dir),
        })
    }

    /// Create a multi-objective study.
    #[staticmethod]
    fn create_multi(directions: Vec<String>) -> PyResult<Self> {
        let dirs: Result<Vec<_>, _> = directions
            .iter()
            .map(|d| match d.as_str() {
                "minimize" => Ok(rl4burn_tune::study::Direction::Minimize),
                "maximize" => Ok(rl4burn_tune::study::Direction::Maximize),
                _ => Err(pyo3::exceptions::PyValueError::new_err(
                    "direction must be 'minimize' or 'maximize'",
                )),
            })
            .collect();
        Ok(Self {
            inner: rl4burn_tune::Study::new_multi(dirs?),
        })
    }

    /// Get the best objective value.
    fn best_value(&self) -> Option<f64> {
        self.inner.best_value()
    }

    /// Get the best trial's parameters.
    fn best_params(&self) -> Option<HashMap<String, f64>> {
        self.inner.best_trial().map(|t| t.params.clone())
    }

    /// Get the number of completed trials.
    fn n_completed(&self) -> usize {
        self.inner.n_completed()
    }

    /// Get the total number of trials.
    fn n_trials(&self) -> usize {
        self.inner.trials().len()
    }

    /// Ask for a new trial (ask/tell interface).
    fn ask(&self) -> PyTrial {
        let trial = self.inner.ask();
        PyTrial { inner: Some(trial) }
    }

    /// Tell the study the result of a trial.
    #[pyo3(signature = (trial, value, state = "complete"))]
    fn tell(&mut self, trial: &mut PyTrial, value: Option<f64>, state: &str) -> PyResult<()> {
        let inner_trial = trial.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trial has already been told")
        })?;
        let trial_state = match state {
            "complete" => rl4burn_tune::trial::TrialState::Complete,
            "pruned" => rl4burn_tune::trial::TrialState::Pruned,
            "fail" => rl4burn_tune::trial::TrialState::Fail,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "state must be 'complete', 'pruned', or 'fail'",
                ))
            }
        };
        self.inner.tell(inner_trial, trial_state, value);
        Ok(())
    }

    /// Add a completed trial with known params and value.
    fn add_completed_trial(&mut self, params: HashMap<String, f64>, value: f64) {
        self.inner.add_completed_trial(params, value);
    }

    /// Enqueue a trial with pre-set parameters.
    fn enqueue_trial(&mut self, params: HashMap<String, f64>) {
        self.inner.enqueue_trial(params);
    }

    /// Stop the study.
    fn stop(&mut self) {
        self.inner.stop();
    }

    /// Check if the study has been stopped.
    fn is_stopped(&self) -> bool {
        self.inner.is_stopped()
    }

    /// Get all trial data as a list of dicts.
    fn get_trials(&self) -> Vec<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            self.inner
                .trials()
                .iter()
                .map(|t| {
                    let mut d = HashMap::new();
                    d.insert("number".to_string(), t.number.into_pyobject(py).unwrap().into_any().unbind());
                    d.insert(
                        "state".to_string(),
                        format!("{:?}", t.state).into_pyobject(py).unwrap().into_any().unbind(),
                    );
                    if let Some(v) = t.value {
                        d.insert("value".to_string(), v.into_pyobject(py).unwrap().into_any().unbind());
                    }
                    d.insert(
                        "params".to_string(),
                        t.params.clone().into_pyobject(py).unwrap().into_any().unbind(),
                    );
                    d
                })
                .collect()
        })
    }
}

/// Python wrapper for Trial (ask/tell interface).
#[pyclass(name = "Trial")]
struct PyTrial {
    inner: Option<rl4burn_tune::Trial>,
}

#[pymethods]
impl PyTrial {
    /// Suggest a float parameter.
    #[pyo3(signature = (name, low, high, sampler, study, log = false, step = None))]
    fn suggest_float(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        sampler: &PySampler,
        study: &PyStudy,
        log: bool,
        step: Option<f64>,
    ) -> PyResult<f64> {
        let trial = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trial has already been told")
        })?;
        Ok(trial.suggest_float(name, low, high, log, step, sampler.as_sampler(), &study.inner))
    }

    /// Suggest an integer parameter.
    #[pyo3(signature = (name, low, high, sampler, study, log = false, step = None))]
    fn suggest_int(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        sampler: &PySampler,
        study: &PyStudy,
        log: bool,
        step: Option<i64>,
    ) -> PyResult<i64> {
        let trial = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trial has already been told")
        })?;
        Ok(trial.suggest_int(name, low, high, log, step, sampler.as_sampler(), &study.inner))
    }

    /// Suggest a categorical parameter (returns index).
    fn suggest_categorical(
        &mut self,
        name: &str,
        choices: Vec<String>,
        sampler: &PySampler,
        study: &PyStudy,
    ) -> PyResult<usize> {
        let trial = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trial has already been told")
        })?;
        Ok(trial.suggest_categorical(name, choices, sampler.as_sampler(), &study.inner))
    }

    /// Report an intermediate value.
    fn report(&mut self, step: usize, value: f64) -> PyResult<()> {
        let trial = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trial has already been told")
        })?;
        trial.report(step, value);
        Ok(())
    }

    /// Set a user attribute.
    fn set_user_attr(&mut self, key: String, value: String) -> PyResult<()> {
        let trial = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trial has already been told")
        })?;
        trial.set_user_attr(key, value);
        Ok(())
    }

    /// Get trial number.
    #[getter]
    fn number(&self) -> PyResult<usize> {
        let trial = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trial has already been told")
        })?;
        Ok(trial.number)
    }

    /// Get current params.
    #[getter]
    fn params(&self) -> PyResult<HashMap<String, f64>> {
        let trial = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trial has already been told")
        })?;
        Ok(trial.params.clone())
    }
}

/// Enum wrapping all sampler types for Python.
#[pyclass(name = "Sampler")]
struct PySampler {
    inner: SamplerKind,
}

enum SamplerKind {
    Random(rl4burn_tune::RandomSampler),
    Tpe(rl4burn_tune::TpeSampler),
    CmaEs(rl4burn_tune::CmaEsSampler),
    NsgaII(rl4burn_tune::NsgaIISampler),
}

impl PySampler {
    fn as_sampler(&self) -> &dyn rl4burn_tune::Sampler {
        match &self.inner {
            SamplerKind::Random(s) => s,
            SamplerKind::Tpe(s) => s,
            SamplerKind::CmaEs(s) => s,
            SamplerKind::NsgaII(s) => s,
        }
    }
}

/// Create a Random sampler.
#[pyfunction]
fn random_sampler(seed: u64) -> PySampler {
    PySampler {
        inner: SamplerKind::Random(rl4burn_tune::RandomSampler::new(seed)),
    }
}

/// Create a TPE sampler.
#[pyfunction]
#[pyo3(signature = (seed, n_startup_trials = 10, constant_liar = false))]
fn tpe_sampler(seed: u64, n_startup_trials: usize, constant_liar: bool) -> PySampler {
    let config = rl4burn_tune::TpeSamplerConfig {
        n_startup_trials,
        constant_liar,
        ..Default::default()
    };
    PySampler {
        inner: SamplerKind::Tpe(rl4burn_tune::TpeSampler::new(config, seed)),
    }
}

/// Create a CMA-ES sampler.
#[pyfunction]
#[pyo3(signature = (seed, n_startup_trials = 10))]
fn cmaes_sampler(seed: u64, n_startup_trials: usize) -> PySampler {
    let config = rl4burn_tune::CmaEsConfig {
        n_startup_trials,
        ..Default::default()
    };
    PySampler {
        inner: SamplerKind::CmaEs(rl4burn_tune::CmaEsSampler::new(config, seed)),
    }
}

/// Create an NSGA-II sampler.
#[pyfunction]
#[pyo3(signature = (seed, population_size = 50))]
fn nsga2_sampler(seed: u64, population_size: usize) -> PySampler {
    let config = rl4burn_tune::NsgaIIConfig {
        population_size,
        ..Default::default()
    };
    PySampler {
        inner: SamplerKind::NsgaII(rl4burn_tune::NsgaIISampler::new(config, seed)),
    }
}

// ============================================================
// Original low-level functions (preserved for backward compatibility)
// ============================================================

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

#[pyfunction]
fn calculate_order(values: Vec<f64>) -> Vec<usize> {
    rs_calculate_order(&values)
}

#[pyfunction]
fn gaussian_log_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    rs_gaussian_log_pdf(x, mu, sigma)
}

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

// ============================================================
// Module registration
// ============================================================

#[pymodule]
fn rl4burn_tune_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyStudy>()?;
    m.add_class::<PyTrial>()?;
    m.add_class::<PySampler>()?;

    // Sampler constructors
    m.add_function(wrap_pyfunction!(random_sampler, m)?)?;
    m.add_function(wrap_pyfunction!(tpe_sampler, m)?)?;
    m.add_function(wrap_pyfunction!(cmaes_sampler, m)?)?;
    m.add_function(wrap_pyfunction!(nsga2_sampler, m)?)?;

    // Low-level functions (backward compatible)
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
