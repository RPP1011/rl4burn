mod cmaes;
mod random;
mod tpe;

pub use cmaes::{CmaEsConfig, CmaEsSampler};
pub use random::RandomSampler;
pub use tpe::{TpeSampler, TpeSamplerConfig};

use crate::distributions::Distribution;
use crate::study::Study;
use crate::trial::Trial;

/// Trait for hyperparameter samplers.
pub trait Sampler: Send + Sync {
    /// Sample a parameter value for the given distribution.
    fn sample(
        &self,
        study: &Study,
        trial: &Trial,
        param_name: &str,
        distribution: &Distribution,
    ) -> f64;
}
