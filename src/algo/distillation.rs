//! Policy distillation (Issue #17).
//!
//! Train a student network to match a teacher's action distribution using
//! KL divergence on softened logits (Hinton et al., 2015). Optionally
//! includes a hard-target cross-entropy loss and value distillation.
//!
//! # Example
//!
//! ```ignore
//! let config = DistillationConfig::default();
//! let loss = distillation_loss(teacher_logits.detach(), student_logits, &config);
//! ```

use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for policy distillation.
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Temperature for softening distributions. Default: 1.0.
    pub temperature: f32,
    /// Weight for soft-target (KL) loss. Default: 1.0.
    pub soft_weight: f32,
    /// Weight for hard-target (cross-entropy) loss. Default: 0.0.
    pub hard_weight: f32,
    /// Whether to apply T² scaling to soft loss. Default: true.
    pub t_squared_scaling: bool,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            soft_weight: 1.0,
            hard_weight: 0.0,
            t_squared_scaling: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Distillation loss
// ---------------------------------------------------------------------------

/// Compute distillation loss between teacher and student logits.
///
/// Soft loss: KL(teacher_soft || student_soft) where soft = softmax(logits / T).
///
/// The teacher logits should be detached (no gradient) before calling this.
///
/// # Arguments
/// * `teacher_logits` — `[batch, n_actions]` (detached)
/// * `student_logits` — `[batch, n_actions]`
/// * `config` — Distillation hyperparameters
///
/// # Returns
/// Scalar loss `[1]`.
pub fn distillation_loss<B: Backend>(
    teacher_logits: Tensor<B, 2>,
    student_logits: Tensor<B, 2>,
    config: &DistillationConfig,
) -> Tensor<B, 1> {
    let t = config.temperature;

    // Soft targets
    let teacher_soft = softmax(teacher_logits.clone() / t, 1);
    let student_log_soft = log_softmax(student_logits / t, 1);
    let teacher_log_soft = log_softmax(teacher_logits / t, 1);

    // KL(teacher || student) = sum(teacher * (log_teacher - log_student))
    let kl = (teacher_soft * (teacher_log_soft - student_log_soft))
        .sum_dim(1)
        .squeeze_dim::<1>(1);
    let mut soft_loss = kl.mean();

    if config.t_squared_scaling {
        soft_loss = soft_loss * (t * t);
    }

    (soft_loss * config.soft_weight).unsqueeze()
}

/// Compute value distillation loss (MSE between teacher and student values).
///
/// # Arguments
/// * `teacher_values` — `[batch]` (detached)
/// * `student_values` — `[batch]`
///
/// # Returns
/// Scalar loss `[1]`.
pub fn value_distillation_loss<B: Backend>(
    teacher_values: Tensor<B, 1>,
    student_values: Tensor<B, 1>,
) -> Tensor<B, 1> {
    (teacher_values - student_values)
        .powf_scalar(2.0)
        .mean()
        .unsqueeze()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    #[test]
    fn default_config() {
        let cfg = DistillationConfig::default();
        assert!((cfg.temperature - 1.0).abs() < 1e-6);
        assert!((cfg.soft_weight - 1.0).abs() < 1e-6);
        assert!((cfg.hard_weight - 0.0).abs() < 1e-6);
        assert!(cfg.t_squared_scaling);
    }

    #[test]
    fn loss_is_positive_scalar() {
        let teacher = Tensor::<B, 2>::from_data(
            TensorData::new(vec![2.0f32, 1.0, -1.0, -2.0, 3.0, 0.0], [2, 3]),
            &dev(),
        );
        let student = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0], [2, 3]),
            &dev(),
        );
        let config = DistillationConfig::default();
        let loss = distillation_loss(teacher, student, &config);
        let vals: Vec<f32> = loss.into_data().to_vec().unwrap();
        assert_eq!(vals.len(), 1);
        assert!(vals[0] > 0.0, "KL loss should be positive, got {}", vals[0]);
    }

    #[test]
    fn identical_logits_give_zero_loss() {
        let logits = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]),
            &dev(),
        );
        let config = DistillationConfig::default();
        let loss = distillation_loss(logits.clone(), logits, &config);
        let vals: Vec<f32> = loss.into_data().to_vec().unwrap();
        assert!(
            vals[0].abs() < 1e-5,
            "identical logits should give ~0 loss, got {}",
            vals[0]
        );
    }

    #[test]
    fn loss_decreases_as_student_matches_teacher() {
        let teacher = Tensor::<B, 2>::from_data(
            TensorData::new(vec![3.0f32, 1.0, 0.0], [1, 3]),
            &dev(),
        );
        // Student far from teacher
        let student_far = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 3.0], [1, 3]),
            &dev(),
        );
        // Student close to teacher
        let student_close = Tensor::<B, 2>::from_data(
            TensorData::new(vec![2.8f32, 1.1, 0.1], [1, 3]),
            &dev(),
        );
        let config = DistillationConfig::default();
        let loss_far: f32 = distillation_loss(teacher.clone(), student_far, &config)
            .into_data()
            .to_vec()
            .unwrap()[0];
        let loss_close: f32 = distillation_loss(teacher, student_close, &config)
            .into_data()
            .to_vec()
            .unwrap()[0];
        assert!(
            loss_close < loss_far,
            "closer student should have lower loss: close={loss_close}, far={loss_far}"
        );
    }

    #[test]
    fn temperature_scaling() {
        let teacher = Tensor::<B, 2>::from_data(
            TensorData::new(vec![3.0f32, 0.0, -1.0], [1, 3]),
            &dev(),
        );
        let student = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.0f32, 1.0, 0.0], [1, 3]),
            &dev(),
        );
        let cfg_t1 = DistillationConfig {
            temperature: 1.0,
            t_squared_scaling: false,
            ..Default::default()
        };
        let cfg_t5 = DistillationConfig {
            temperature: 5.0,
            t_squared_scaling: false,
            ..Default::default()
        };
        let loss_t1: f32 = distillation_loss(teacher.clone(), student.clone(), &cfg_t1)
            .into_data()
            .to_vec()
            .unwrap()[0];
        let loss_t5: f32 = distillation_loss(teacher, student, &cfg_t5)
            .into_data()
            .to_vec()
            .unwrap()[0];
        // Higher temperature softens distributions, typically reducing raw KL
        assert!(
            (loss_t1 - loss_t5).abs() > 1e-6,
            "different temperatures should give different losses"
        );
    }

    #[test]
    fn value_distillation_loss_positive() {
        let teacher = Tensor::<B, 1>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0], [3]),
            &dev(),
        );
        let student = Tensor::<B, 1>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.0], [3]),
            &dev(),
        );
        let loss: f32 = value_distillation_loss(teacher, student)
            .into_data()
            .to_vec()
            .unwrap()[0];
        // MSE of (1,2,3) vs (0,0,0) = (1+4+9)/3 = 14/3 ≈ 4.667
        assert!((loss - 14.0 / 3.0).abs() < 1e-4, "expected ~4.667, got {loss}");
    }
}
