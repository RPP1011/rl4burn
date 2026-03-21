use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A float hyperparameter distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatDistribution {
    pub low: f64,
    pub high: f64,
    pub log: bool,
    pub step: Option<f64>,
}

/// An integer hyperparameter distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntDistribution {
    pub low: i64,
    pub high: i64,
    pub log: bool,
    pub step: Option<i64>,
}

/// A categorical hyperparameter distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalDistribution {
    pub choices: Vec<String>,
}

/// Any hyperparameter distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Distribution {
    Float(FloatDistribution),
    Int(IntDistribution),
    Categorical(CategoricalDistribution),
}

impl Distribution {
    /// Check if this distribution has only one possible value.
    pub fn single(&self) -> bool {
        match self {
            Distribution::Float(d) => d.single(),
            Distribution::Int(d) => d.single(),
            Distribution::Categorical(d) => d.single(),
        }
    }
}

/// A named map from parameter names to distributions.
pub type SearchSpace = HashMap<String, Distribution>;

impl FloatDistribution {
    /// Create a new float distribution.
    ///
    /// # Panics
    /// - If `low >= high`
    /// - If `log` is true and `low <= 0`
    /// - If `step` is set and the range is not evenly divisible by it
    pub fn new(low: f64, high: f64, log: bool, step: Option<f64>) -> Self {
        assert!(
            low < high,
            "FloatDistribution requires low < high, got low={low}, high={high}"
        );
        if log {
            assert!(
                low > 0.0,
                "Log-scale FloatDistribution requires low > 0, got {low}"
            );
        }
        if let Some(s) = step {
            assert!(s > 0.0, "Step must be positive, got {s}");
            let range = high - low;
            let n_steps = (range / s).round();
            assert!(
                (n_steps * s - range).abs() < 1e-12 * range.abs().max(1.0),
                "Range {range} is not evenly divisible by step {s}"
            );
        }
        Self {
            low,
            high,
            log,
            step,
        }
    }

    /// Check if this distribution has only one possible value.
    pub fn single(&self) -> bool {
        if let Some(step) = self.step {
            ((self.high - self.low) / step).round() < 1.0
        } else {
            (self.high - self.low).abs() < 1e-15
        }
    }

    /// Check if a value is within the distribution's support.
    pub fn contains(&self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        if value < self.low || value > self.high {
            return false;
        }
        if let Some(step) = self.step {
            let shifted = value - self.low;
            let remainder = shifted % step;
            remainder.abs() < 1e-12 * step.abs().max(1.0)
                || (step - remainder.abs()).abs() < 1e-12 * step.abs().max(1.0)
        } else {
            true
        }
    }
}

impl IntDistribution {
    /// Create a new integer distribution.
    ///
    /// # Panics
    /// - If `low >= high`
    /// - If `log` is true and `low <= 0`
    pub fn new(low: i64, high: i64, log: bool, step: Option<i64>) -> Self {
        assert!(
            low < high,
            "IntDistribution requires low < high, got low={low}, high={high}"
        );
        if log {
            assert!(
                low > 0,
                "Log-scale IntDistribution requires low > 0, got {low}"
            );
        }
        if let Some(s) = step {
            assert!(s > 0, "Step must be positive, got {s}");
        }
        Self {
            low,
            high,
            log,
            step,
        }
    }

    /// Check if this distribution has only one possible value.
    pub fn single(&self) -> bool {
        let step = self.step.unwrap_or(1);
        (self.high - self.low) < step
    }

    /// Check if a value is within the distribution's support.
    pub fn contains(&self, value: i64) -> bool {
        if value < self.low || value > self.high {
            return false;
        }
        let step = self.step.unwrap_or(1);
        (value - self.low) % step == 0
    }
}

impl CategoricalDistribution {
    /// Create a new categorical distribution.
    ///
    /// # Panics
    /// - If `choices` is empty
    pub fn new(choices: Vec<String>) -> Self {
        assert!(!choices.is_empty(), "CategoricalDistribution requires at least one choice");
        Self { choices }
    }

    /// Check if this distribution has only one possible value.
    pub fn single(&self) -> bool {
        self.choices.len() == 1
    }

    /// Check if an index is valid for this distribution.
    pub fn contains_index(&self, index: usize) -> bool {
        index < self.choices.len()
    }
}

/// Transform a value from user-space to internal [0, 1] representation.
pub fn transform_to_internal(value: f64, dist: &FloatDistribution) -> f64 {
    let v = if dist.log {
        (value.ln() - dist.low.ln()) / (dist.high.ln() - dist.low.ln())
    } else {
        (value - dist.low) / (dist.high - dist.low)
    };
    v.clamp(0.0, 1.0)
}

/// Transform a value from internal [0, 1] representation back to user-space.
pub fn transform_from_internal(internal: f64, dist: &FloatDistribution) -> f64 {
    let value = if dist.log {
        (dist.low.ln() + internal * (dist.high.ln() - dist.low.ln())).exp()
    } else {
        dist.low + internal * (dist.high - dist.low)
    };

    if let Some(step) = dist.step {
        let shifted = value - dist.low;
        let quantized = (shifted / step).round() * step + dist.low;
        quantized.clamp(dist.low, dist.high)
    } else {
        value.clamp(dist.low, dist.high)
    }
}

/// Transform an integer value to internal [0, 1] representation.
pub fn int_transform_to_internal(value: i64, dist: &IntDistribution) -> f64 {
    let v = if dist.log {
        ((value as f64).ln() - (dist.low as f64).ln())
            / ((dist.high as f64).ln() - (dist.low as f64).ln())
    } else {
        (value - dist.low) as f64 / (dist.high - dist.low) as f64
    };
    v.clamp(0.0, 1.0)
}

/// Transform from internal [0, 1] back to an integer value.
pub fn int_transform_from_internal(internal: f64, dist: &IntDistribution) -> i64 {
    let step = dist.step.unwrap_or(1);
    let value = if dist.log {
        let log_low = (dist.low as f64).ln();
        let log_high = (dist.high as f64).ln();
        (log_low + internal * (log_high - log_low)).exp()
    } else {
        dist.low as f64 + internal * (dist.high - dist.low) as f64
    };

    // Quantize to the step grid
    let shifted = value - dist.low as f64;
    let quantized = (shifted / step as f64).round() * step as f64 + dist.low as f64;
    (quantized.round() as i64).clamp(dist.low, dist.high)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_distribution_basic() {
        let d = FloatDistribution::new(0.0, 1.0, false, None);
        assert!(d.contains(0.0));
        assert!(d.contains(0.5));
        assert!(d.contains(1.0));
        assert!(!d.contains(-0.1));
        assert!(!d.contains(1.1));
    }

    #[test]
    fn test_float_distribution_log() {
        let d = FloatDistribution::new(1e-5, 1.0, true, None);
        assert!(d.contains(1e-5));
        assert!(d.contains(0.01));
        assert!(!d.contains(0.0));
    }

    #[test]
    fn test_float_distribution_step() {
        let d = FloatDistribution::new(0.0, 1.0, false, Some(0.25));
        assert!(d.contains(0.0));
        assert!(d.contains(0.25));
        assert!(d.contains(0.5));
        assert!(d.contains(1.0));
    }

    #[test]
    #[should_panic]
    fn test_float_distribution_invalid_range() {
        FloatDistribution::new(1.0, 0.0, false, None);
    }

    #[test]
    #[should_panic]
    fn test_float_distribution_log_negative() {
        FloatDistribution::new(-1.0, 1.0, true, None);
    }

    #[test]
    fn test_int_distribution() {
        let d = IntDistribution::new(0, 10, false, None);
        assert!(d.contains(0));
        assert!(d.contains(5));
        assert!(d.contains(10));
        assert!(!d.contains(-1));
        assert!(!d.contains(11));
    }

    #[test]
    fn test_int_distribution_step() {
        let d = IntDistribution::new(0, 10, false, Some(2));
        assert!(d.contains(0));
        assert!(d.contains(2));
        assert!(d.contains(4));
        assert!(!d.contains(1));
        assert!(!d.contains(3));
    }

    #[test]
    fn test_categorical() {
        let d = CategoricalDistribution::new(vec!["a".into(), "b".into(), "c".into()]);
        assert!(d.contains_index(0));
        assert!(d.contains_index(2));
        assert!(!d.contains_index(3));
    }

    #[test]
    fn test_transform_roundtrip_linear() {
        let d = FloatDistribution::new(2.0, 8.0, false, None);
        let values = [2.0, 3.5, 5.0, 6.5, 8.0];
        for &v in &values {
            let internal = transform_to_internal(v, &d);
            let recovered = transform_from_internal(internal, &d);
            assert!((recovered - v).abs() < 1e-12, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_transform_roundtrip_log() {
        let d = FloatDistribution::new(1e-3, 1e3, true, None);
        let values = [1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1e3];
        for &v in &values {
            let internal = transform_to_internal(v, &d);
            assert!(internal >= 0.0 && internal <= 1.0);
            let recovered = transform_from_internal(internal, &d);
            assert!(
                (recovered - v).abs() < 1e-10 * v.abs().max(1.0),
                "roundtrip failed for {v}: got {recovered}"
            );
        }
    }

    #[test]
    fn test_int_transform_roundtrip() {
        let d = IntDistribution::new(1, 100, false, None);
        for v in [1, 25, 50, 75, 100] {
            let internal = int_transform_to_internal(v, &d);
            let recovered = int_transform_from_internal(internal, &d);
            assert_eq!(recovered, v, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_int_transform_roundtrip_log() {
        let d = IntDistribution::new(1, 1000, true, None);
        for v in [1, 10, 100, 1000] {
            let internal = int_transform_to_internal(v, &d);
            let recovered = int_transform_from_internal(internal, &d);
            assert_eq!(recovered, v, "roundtrip failed for {v}");
        }
    }
}
