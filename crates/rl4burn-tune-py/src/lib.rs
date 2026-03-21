use std::collections::HashMap;
use std::sync::Mutex;

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
        let dir = parse_direction(direction)?;
        Ok(Self {
            inner: rl4burn_tune::Study::new(dir),
        })
    }

    /// Create a multi-objective study.
    #[staticmethod]
    fn create_multi(directions: Vec<String>) -> PyResult<Self> {
        let dirs: Result<Vec<_>, _> = directions.iter().map(|d| parse_direction(d)).collect();
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
        let trial_state = parse_trial_state(state)?;
        self.inner.tell(inner_trial, trial_state, value);
        Ok(())
    }

    /// Tell the study the result of a multi-objective trial.
    #[pyo3(signature = (trial, values, state = "complete"))]
    fn tell_multi(
        &mut self,
        trial: &mut PyTrial,
        values: Vec<f64>,
        state: &str,
    ) -> PyResult<()> {
        let inner_trial = trial.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trial has already been told")
        })?;
        let trial_state = parse_trial_state(state)?;
        self.inner.tell_multi(inner_trial, trial_state, values);
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
                    if let Some(ref cv) = t.constraint_values {
                        d.insert(
                            "constraint_values".to_string(),
                            cv.clone().into_pyobject(py).unwrap().into_any().unbind(),
                        );
                    }
                    let feasible: PyObject = t.is_feasible().into_pyobject(py).unwrap().to_owned().unbind().into();
                    d.insert("is_feasible".to_string(), feasible);
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

    /// Check if the trial should be pruned.
    fn should_prune(&mut self, study: &PyStudy, pruner: &PyPruner) -> PyResult<bool> {
        let trial = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trial has already been told")
        })?;
        Ok(trial.should_prune(&study.inner, Some(pruner.as_pruner())))
    }

    /// Set a user attribute.
    fn set_user_attr(&mut self, key: String, value: String) -> PyResult<()> {
        let trial = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trial has already been told")
        })?;
        trial.set_user_attr(key, value);
        Ok(())
    }

    /// Set constraint values for constrained optimization.
    fn set_constraints(&mut self, values: Vec<f64>) -> PyResult<()> {
        let trial = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trial has already been told")
        })?;
        trial.set_constraints(values);
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

// ============================================================
// Sampler bindings
// ============================================================

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
    Gp(rl4burn_tune::GpSampler),
}

impl PySampler {
    fn as_sampler(&self) -> &dyn rl4burn_tune::Sampler {
        match &self.inner {
            SamplerKind::Random(s) => s,
            SamplerKind::Tpe(s) => s,
            SamplerKind::CmaEs(s) => s,
            SamplerKind::NsgaII(s) => s,
            SamplerKind::Gp(s) => s,
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
#[pyo3(signature = (seed, n_startup_trials = 10, constant_liar = false, multivariate = false))]
fn tpe_sampler(seed: u64, n_startup_trials: usize, constant_liar: bool, multivariate: bool) -> PySampler {
    let config = rl4burn_tune::TpeSamplerConfig {
        n_startup_trials,
        constant_liar,
        multivariate,
        ..Default::default()
    };
    PySampler {
        inner: SamplerKind::Tpe(rl4burn_tune::TpeSampler::new(config, seed)),
    }
}

/// Create a CMA-ES sampler.
#[pyfunction]
#[pyo3(signature = (seed, n_startup_trials = 10, population_size = None))]
fn cmaes_sampler(seed: u64, n_startup_trials: usize, population_size: Option<usize>) -> PySampler {
    let config = rl4burn_tune::CmaEsConfig {
        n_startup_trials,
        population_size,
        ..Default::default()
    };
    PySampler {
        inner: SamplerKind::CmaEs(rl4burn_tune::CmaEsSampler::new(config, seed)),
    }
}

/// Create an NSGA-II sampler.
#[pyfunction]
#[pyo3(signature = (seed, population_size = 50, crossover_type = "sbx"))]
fn nsga2_sampler(seed: u64, population_size: usize, crossover_type: &str) -> PyResult<PySampler> {
    let ct = match crossover_type {
        "sbx" => rl4burn_tune::CrossoverType::SBX,
        "uniform" => rl4burn_tune::CrossoverType::Uniform,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "crossover_type must be 'sbx' or 'uniform'",
            ))
        }
    };
    let config = rl4burn_tune::NsgaIIConfig {
        population_size,
        crossover_type: ct,
        ..Default::default()
    };
    Ok(PySampler {
        inner: SamplerKind::NsgaII(rl4burn_tune::NsgaIISampler::new(config, seed)),
    })
}

/// Create a Gaussian Process sampler.
#[pyfunction]
#[pyo3(signature = (seed, n_startup_trials = 10, n_candidates = 2000))]
fn gp_sampler(seed: u64, n_startup_trials: usize, n_candidates: usize) -> PySampler {
    let config = rl4burn_tune::GpConfig {
        n_startup_trials,
        n_candidates,
    };
    PySampler {
        inner: SamplerKind::Gp(rl4burn_tune::GpSampler::new(config, seed)),
    }
}

// ============================================================
// Pruner bindings
// ============================================================

/// Enum wrapping all pruner types for Python.
#[pyclass(name = "Pruner")]
struct PyPruner {
    inner: PrunerKind,
}

enum PrunerKind {
    Nop(rl4burn_tune::NopPruner),
    Median(rl4burn_tune::MedianPruner),
    Percentile(rl4burn_tune::PercentilePruner),
    Threshold(rl4burn_tune::ThresholdPruner),
    Patient(rl4burn_tune::PatientPruner),
    SuccessiveHalving(rl4burn_tune::SuccessiveHalvingPruner),
    Hyperband(rl4burn_tune::HyperbandPruner),
    Wilcoxon(rl4burn_tune::WilcoxonPruner),
}

impl PyPruner {
    fn as_pruner(&self) -> &dyn rl4burn_tune::Pruner {
        match &self.inner {
            PrunerKind::Nop(p) => p,
            PrunerKind::Median(p) => p,
            PrunerKind::Percentile(p) => p,
            PrunerKind::Threshold(p) => p,
            PrunerKind::Patient(p) => p,
            PrunerKind::SuccessiveHalving(p) => p,
            PrunerKind::Hyperband(p) => p,
            PrunerKind::Wilcoxon(p) => p,
        }
    }
}

/// Create a NopPruner (never prunes).
#[pyfunction]
fn nop_pruner() -> PyPruner {
    PyPruner {
        inner: PrunerKind::Nop(rl4burn_tune::NopPruner),
    }
}

/// Create a MedianPruner.
#[pyfunction]
#[pyo3(signature = (n_startup_trials = 5, n_warmup_steps = 0, interval_steps = 1))]
fn median_pruner(n_startup_trials: usize, n_warmup_steps: usize, interval_steps: usize) -> PyPruner {
    PyPruner {
        inner: PrunerKind::Median(rl4burn_tune::MedianPruner::new(
            n_startup_trials,
            n_warmup_steps,
            interval_steps,
        )),
    }
}

/// Create a PercentilePruner.
#[pyfunction]
#[pyo3(signature = (percentile = 50.0, n_startup_trials = 5, n_warmup_steps = 0, interval_steps = 1))]
fn percentile_pruner(
    percentile: f64,
    n_startup_trials: usize,
    n_warmup_steps: usize,
    interval_steps: usize,
) -> PyPruner {
    PyPruner {
        inner: PrunerKind::Percentile(rl4burn_tune::PercentilePruner::new(
            percentile,
            n_startup_trials,
            n_warmup_steps,
            interval_steps,
        )),
    }
}

/// Create a ThresholdPruner.
#[pyfunction]
#[pyo3(signature = (upper = None, lower = None, n_warmup_steps = 0))]
fn threshold_pruner(upper: Option<f64>, lower: Option<f64>, n_warmup_steps: usize) -> PyPruner {
    PyPruner {
        inner: PrunerKind::Threshold(rl4burn_tune::ThresholdPruner::new(
            upper,
            lower,
            n_warmup_steps,
        )),
    }
}

/// Create a PatientPruner wrapping a MedianPruner.
#[pyfunction]
#[pyo3(signature = (patience = 3, n_startup_trials = 5, n_warmup_steps = 0, interval_steps = 1))]
fn patient_pruner(
    patience: usize,
    n_startup_trials: usize,
    n_warmup_steps: usize,
    interval_steps: usize,
) -> PyPruner {
    let inner = rl4burn_tune::MedianPruner::new(n_startup_trials, n_warmup_steps, interval_steps);
    PyPruner {
        inner: PrunerKind::Patient(rl4burn_tune::PatientPruner::new(
            Box::new(inner),
            patience,
        )),
    }
}

/// Create a SuccessiveHalvingPruner.
#[pyfunction]
#[pyo3(signature = (min_resource = 1, reduction_factor = 3, n_warmup_steps = 0))]
fn successive_halving_pruner(
    min_resource: usize,
    reduction_factor: usize,
    n_warmup_steps: usize,
) -> PyPruner {
    PyPruner {
        inner: PrunerKind::SuccessiveHalving(rl4burn_tune::SuccessiveHalvingPruner::new(
            min_resource,
            reduction_factor,
            n_warmup_steps,
        )),
    }
}

/// Create a HyperbandPruner.
#[pyfunction]
#[pyo3(signature = (min_resource = 1, max_resource = 100, reduction_factor = 3))]
fn hyperband_pruner(min_resource: usize, max_resource: usize, reduction_factor: usize) -> PyPruner {
    PyPruner {
        inner: PrunerKind::Hyperband(rl4burn_tune::HyperbandPruner::new(
            min_resource,
            max_resource,
            reduction_factor,
        )),
    }
}

/// Create a WilcoxonPruner.
#[pyfunction]
#[pyo3(signature = (p_threshold = 0.1, n_min_steps = 5, n_startup_trials = 5))]
fn wilcoxon_pruner(p_threshold: f64, n_min_steps: usize, n_startup_trials: usize) -> PyPruner {
    PyPruner {
        inner: PrunerKind::Wilcoxon(rl4burn_tune::WilcoxonPruner::new(
            p_threshold,
            n_min_steps,
            n_startup_trials,
        )),
    }
}

// ============================================================
// Storage bindings
// ============================================================

/// Python wrapper for storage backends.
#[pyclass(name = "Storage")]
struct PyStorage {
    inner: Mutex<Box<dyn rl4burn_tune::storage::Storage>>,
}

#[pymethods]
impl PyStorage {
    /// Create a new study and return its ID.
    fn create_study(&self, direction: &str) -> PyResult<usize> {
        let dir = parse_direction(direction)?;
        let mut storage = self.inner.lock().unwrap();
        storage
            .create_study(dir)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get all trials for a study.
    fn get_all_trials(&self, study_id: usize) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let storage = self.inner.lock().unwrap();
        let trials = storage
            .get_all_trials(study_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Python::with_gil(|py| {
            trials
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
        }))
    }

    /// Add a trial to a study.
    fn add_trial(
        &self,
        study_id: usize,
        number: usize,
        params: HashMap<String, f64>,
        value: Option<f64>,
        state: &str,
    ) -> PyResult<()> {
        let trial_state = parse_trial_state(state)?;
        let mut trial = rl4burn_tune::FrozenTrial::new(number);
        trial.params = params;
        trial.value = value;
        trial.state = trial_state;

        let mut storage = self.inner.lock().unwrap();
        storage
            .add_trial(study_id, trial)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

/// Create an in-memory storage backend.
#[pyfunction]
fn in_memory_storage() -> PyStorage {
    PyStorage {
        inner: Mutex::new(Box::new(rl4burn_tune::storage::InMemoryStorage::new())),
    }
}

/// Create a journal file storage backend.
#[pyfunction]
fn journal_storage(path: &str) -> PyResult<PyStorage> {
    let storage = rl4burn_tune::storage::JournalStorage::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(PyStorage {
        inner: Mutex::new(Box::new(storage)),
    })
}

// ============================================================
// Helper functions
// ============================================================

fn parse_direction(s: &str) -> PyResult<rl4burn_tune::study::Direction> {
    match s {
        "minimize" => Ok(rl4burn_tune::study::Direction::Minimize),
        "maximize" => Ok(rl4burn_tune::study::Direction::Maximize),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "direction must be 'minimize' or 'maximize'",
        )),
    }
}

fn parse_trial_state(s: &str) -> PyResult<rl4burn_tune::trial::TrialState> {
    match s {
        "complete" => Ok(rl4burn_tune::trial::TrialState::Complete),
        "pruned" => Ok(rl4burn_tune::trial::TrialState::Pruned),
        "fail" => Ok(rl4burn_tune::trial::TrialState::Fail),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "state must be 'complete', 'pruned', or 'fail'",
        )),
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
    m.add_class::<PyPruner>()?;
    m.add_class::<PyStorage>()?;

    // Sampler constructors
    m.add_function(wrap_pyfunction!(random_sampler, m)?)?;
    m.add_function(wrap_pyfunction!(tpe_sampler, m)?)?;
    m.add_function(wrap_pyfunction!(cmaes_sampler, m)?)?;
    m.add_function(wrap_pyfunction!(nsga2_sampler, m)?)?;
    m.add_function(wrap_pyfunction!(gp_sampler, m)?)?;

    // Pruner constructors
    m.add_function(wrap_pyfunction!(nop_pruner, m)?)?;
    m.add_function(wrap_pyfunction!(median_pruner, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_pruner, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_pruner, m)?)?;
    m.add_function(wrap_pyfunction!(patient_pruner, m)?)?;
    m.add_function(wrap_pyfunction!(successive_halving_pruner, m)?)?;
    m.add_function(wrap_pyfunction!(hyperband_pruner, m)?)?;
    m.add_function(wrap_pyfunction!(wilcoxon_pruner, m)?)?;

    // Storage constructors
    m.add_function(wrap_pyfunction!(in_memory_storage, m)?)?;
    m.add_function(wrap_pyfunction!(journal_storage, m)?)?;

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
