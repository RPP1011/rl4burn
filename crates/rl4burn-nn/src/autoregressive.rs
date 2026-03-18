//! Hierarchical / auto-regressive composite action distributions.
//!
//! Provides a factored action space where `P(a) = P(a1) * P(a2|a1) * P(a3|a1,a2) * ...`.
//! Each head is a categorical distribution. The conditioning between heads is handled
//! by the model — this struct provides utilities for sampling, joint log-probs, and entropy.

use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::TensorData;
use rand::{Rng, RngExt};

/// Describes one head in a composite action distribution.
#[derive(Debug, Clone)]
pub struct ActionHead {
    /// Name for debugging/logging (e.g., "action_type", "target", "ability").
    pub name: String,
    /// Number of categories for this head.
    pub n_categories: usize,
    /// Offset into the flat logits tensor where this head's logits begin.
    pub logit_offset: usize,
}

/// Composite auto-regressive action distribution.
///
/// Describes a factored action space: `P(a) = P(a1) * P(a2|a1) * P(a3|a1,a2) * ...`
///
/// Each head is a categorical distribution. The conditioning between heads
/// is handled by the model (the model outputs all logits, potentially
/// conditioned on embeddings of previously sampled actions).
///
/// This struct provides utilities for:
/// - Sampling from flat logits with per-head masking
/// - Computing joint log-probabilities as sum of per-head log-probs
/// - Computing joint entropy
///
/// For the auto-regressive case, the model must be called multiple times
/// (once per head), feeding previous actions back. Use `sample_sequential`
/// with a closure for this pattern.
#[derive(Debug, Clone)]
pub struct CompositeDistribution {
    pub heads: Vec<ActionHead>,
}

impl CompositeDistribution {
    /// Create from a list of category counts per head.
    /// Logit offsets are computed automatically.
    pub fn new(head_sizes: &[usize]) -> Self {
        let mut offset = 0;
        let heads = head_sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                let head = ActionHead {
                    name: format!("head_{}", i),
                    n_categories: n,
                    logit_offset: offset,
                };
                offset += n;
                head
            })
            .collect();
        CompositeDistribution { heads }
    }

    /// Create with named heads.
    pub fn from_heads(names: &[&str], sizes: &[usize]) -> Self {
        assert_eq!(names.len(), sizes.len(), "names and sizes must match");
        let mut offset = 0;
        let heads = names
            .iter()
            .zip(sizes.iter())
            .map(|(&name, &n)| {
                let head = ActionHead {
                    name: name.to_string(),
                    n_categories: n,
                    logit_offset: offset,
                };
                offset += n;
                head
            })
            .collect();
        CompositeDistribution { heads }
    }

    /// Total number of logits across all heads.
    pub fn total_logits(&self) -> usize {
        self.heads.iter().map(|h| h.n_categories).sum()
    }

    /// Number of action dimensions (one per head).
    pub fn n_heads(&self) -> usize {
        self.heads.len()
    }

    /// Sample from all heads given pre-computed flat logits.
    ///
    /// This is for the NON-auto-regressive case where all logits are
    /// computed in a single forward pass (heads are independent or
    /// the model has already conditioned them).
    ///
    /// # Arguments
    /// * `logits` - `[batch, total_logits]` -- flat logits for all heads
    /// * `masks` - Optional per-head masks: `[batch, total_logits]` (1=valid, 0=invalid)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Actions `[batch][n_heads]` as f32 (integer-valued).
    pub fn sample<B: Backend>(
        &self,
        logits: &Tensor<B, 2>,
        masks: Option<&Tensor<B, 2>>,
        rng: &mut impl Rng,
    ) -> Vec<Vec<f32>> {
        let [batch, _] = logits.dims();

        let masked_logits = match masks {
            Some(m) => logits.clone() + (m.clone() - 1.0) * 1e9,
            None => logits.clone(),
        };

        let total = self.total_logits();
        let all_probs = softmax(masked_logits, 1);
        let probs_data: Vec<f32> = all_probs.into_data().to_vec().unwrap();

        let mut actions = Vec::with_capacity(batch);
        for b in 0..batch {
            let row = &probs_data[b * total..(b + 1) * total];
            let mut batch_actions = Vec::with_capacity(self.heads.len());
            for head in &self.heads {
                let head_probs = &row[head.logit_offset..head.logit_offset + head.n_categories];
                batch_actions.push(sample_categorical(head_probs, rng) as f32);
            }
            actions.push(batch_actions);
        }

        actions
    }

    /// Compute joint log-probability of actions across all heads.
    ///
    /// `log P(a) = sum_h log P(a_h | logits_h)`
    ///
    /// # Arguments
    /// * `logits` - `[batch, total_logits]`
    /// * `actions` - `[batch][n_heads]` as f32 (integer-valued)
    /// * `masks` - Optional `[batch, total_logits]` (1=valid, 0=invalid)
    ///
    /// # Returns
    /// Joint log-probabilities `[batch]`
    pub fn log_prob<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
        actions: &[Vec<f32>],
        masks: Option<&Tensor<B, 2>>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let batch = actions.len();

        let masked_logits = match masks {
            Some(m) => logits + (m.clone() - 1.0) * 1e9,
            None => logits,
        };

        let mut total_lp: Tensor<B, 1> = Tensor::zeros([batch], device);

        for (h_idx, head) in self.heads.iter().enumerate() {
            let start = head.logit_offset;
            let end = start + head.n_categories;

            let head_logits = masked_logits.clone().slice([0..batch, start..end]);
            let log_probs = log_softmax(head_logits, 1);

            let head_actions: Vec<i32> = actions.iter().map(|a| a[h_idx] as i32).collect();
            let idx_tensor: Tensor<B, 2, Int> = Tensor::from_data(
                TensorData::new(head_actions, [batch, 1]),
                device,
            );
            let gathered: Tensor<B, 1> = log_probs.gather(1, idx_tensor).squeeze_dim::<1>(1);
            total_lp = total_lp + gathered;
        }

        total_lp
    }

    /// Compute joint entropy across all heads.
    ///
    /// `H(a) = sum_h H(a_h)` (exact when heads are independent; upper bound otherwise)
    ///
    /// # Arguments
    /// * `logits` - `[batch, total_logits]`
    /// * `masks` - Optional `[batch, total_logits]` (1=valid, 0=invalid)
    ///
    /// # Returns
    /// Per-sample entropy `[batch]`
    pub fn entropy<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
        masks: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 1> {
        let [batch, _] = logits.dims();
        let device = logits.device();

        let masked_logits = match masks {
            Some(m) => logits + (m.clone() - 1.0) * 1e9,
            None => logits,
        };

        let mut total_entropy: Tensor<B, 1> = Tensor::zeros([batch], &device);

        for head in &self.heads {
            let start = head.logit_offset;
            let end = start + head.n_categories;

            let head_logits = masked_logits.clone().slice([0..batch, start..end]);
            let probs = softmax(head_logits.clone(), 1);
            let log_probs = log_softmax(head_logits, 1);

            // H = -sum(p * log_p, dim=1)
            let ent: Tensor<B, 1> = (probs * log_probs).sum_dim(1).squeeze_dim::<1>(1).neg();
            total_entropy = total_entropy + ent;
        }

        total_entropy
    }
}

fn sample_categorical(probs: &[f32], rng: &mut impl Rng) -> usize {
    let u: f32 = rng.random();
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u < cum {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use rand::SeedableRng;

    type B = NdArray;

    fn device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    #[test]
    fn sample_actions_in_valid_range() {
        let dist = CompositeDistribution::new(&[3, 5]);
        let device = device();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        let logits: Tensor<B, 2> = Tensor::zeros([100, 8], &device);
        let actions = dist.sample(&logits, None, &mut rng);

        assert_eq!(actions.len(), 100);
        for a in &actions {
            assert_eq!(a.len(), 2);
            assert!(a[0] >= 0.0 && a[0] < 3.0, "head 0 out of range: {}", a[0]);
            assert!(a[1] >= 0.0 && a[1] < 5.0, "head 1 out of range: {}", a[1]);
        }
    }

    #[test]
    fn log_prob_chain_rule() {
        // Joint log_prob should equal sum of per-head log_probs computed manually.
        let dist = CompositeDistribution::from_heads(&["type", "target"], &[3, 4]);
        let device = device();

        let logits_data = vec![1.0f32, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5];
        let logits: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(logits_data, [1, 7]), &device);

        let actions = vec![vec![1.0f32, 2.0]];
        let joint_lp: f32 = dist
            .log_prob(logits, &actions, None, &device)
            .into_data()
            .to_vec()
            .unwrap()[0];

        // Manual per-head computation
        let head0_logits = [1.0f32, 2.0, 3.0];
        let head0_max = 3.0;
        let head0_lse: f32 = head0_logits.iter().map(|x| (x - head0_max).exp()).sum();
        let head0_lp = (2.0 - head0_max) - head0_lse.ln(); // action 1

        let head1_logits = [0.5f32, 1.5, 2.5, 3.5];
        let head1_max = 3.5;
        let head1_lse: f32 = head1_logits.iter().map(|x| (x - head1_max).exp()).sum();
        let head1_lp = (2.5 - head1_max) - head1_lse.ln(); // action 2

        let expected = head0_lp + head1_lp;
        assert!(
            (joint_lp - expected).abs() < 1e-5,
            "joint lp {joint_lp} != manual sum {expected}"
        );
    }

    #[test]
    fn entropy_of_uniform_heads() {
        // For uniform logits, H = sum ln(K_i)
        let dist = CompositeDistribution::new(&[3, 5]);
        let device = device();

        let logits: Tensor<B, 2> = Tensor::zeros([1, 8], &device);
        let ent: f32 = dist
            .entropy(logits, None)
            .into_data()
            .to_vec()
            .unwrap()[0];

        let expected = 3.0f32.ln() + 5.0f32.ln();
        assert!(
            (ent - expected).abs() < 1e-4,
            "entropy {ent} != expected {expected}"
        );
    }

    #[test]
    fn exhaustive_enumeration_sums_to_one() {
        // 2 binary heads = 4 total outcomes. sum(exp(log_prob)) should be ~1.0.
        let dist = CompositeDistribution::new(&[2, 2]);
        let device = device();

        let logits_data = vec![0.3f32, 0.7, 1.0, -0.5];
        let logits: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(logits_data, [1, 4]), &device);

        let mut total_prob = 0.0f32;
        for a0 in 0..2 {
            for a1 in 0..2 {
                let actions = vec![vec![a0 as f32, a1 as f32]];
                let lp: f32 = dist
                    .log_prob(
                        logits.clone(),
                        &actions,
                        None,
                        &device,
                    )
                    .into_data()
                    .to_vec()
                    .unwrap()[0];
                total_prob += lp.exp();
            }
        }

        assert!(
            (total_prob - 1.0).abs() < 1e-5,
            "total probability {total_prob} != 1.0"
        );
    }

    #[test]
    fn all_log_probs_non_positive() {
        let dist = CompositeDistribution::new(&[4, 3]);
        let device = device();

        let logits: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(vec![0.1f32, 0.5, 0.3, 0.8, 1.0, -1.0, 0.5], [1, 7]),
            &device,
        );

        for a0 in 0..4 {
            for a1 in 0..3 {
                let actions = vec![vec![a0 as f32, a1 as f32]];
                let lp: f32 = dist
                    .log_prob(logits.clone(), &actions, None, &device)
                    .into_data()
                    .to_vec()
                    .unwrap()[0];
                assert!(lp <= 1e-6, "log_prob should be <= 0, got {lp}");
            }
        }
    }

    #[test]
    fn masked_actions_never_sampled() {
        let dist = CompositeDistribution::new(&[3, 4]);
        let device = device();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(123);

        // Mask: head0 only allows action 2, head1 only allows actions 1 and 3
        let mask_row = vec![
            0.0f32, 0.0, 1.0, // head 0: only action 2
            0.0, 1.0, 0.0, 1.0, // head 1: actions 1, 3
        ];
        let mask_flat: Vec<f32> = (0..1000).flat_map(|_| mask_row.iter().copied()).collect();
        let mask: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(mask_flat, [1000, 7]), &device);
        let logits: Tensor<B, 2> = Tensor::zeros([1000, 7], &device);

        let actions = dist.sample(&logits, Some(&mask), &mut rng);
        for (b, a) in actions.iter().enumerate() {
            assert_eq!(
                a[0] as i32, 2,
                "batch {b}: head 0 should always be 2, got {}",
                a[0]
            );
            let a1 = a[1] as i32;
            assert!(
                a1 == 1 || a1 == 3,
                "batch {b}: head 1 should be 1 or 3, got {a1}"
            );
        }
    }

    #[test]
    fn single_valid_action_entropy_near_zero() {
        let dist = CompositeDistribution::new(&[4, 3]);
        let device = device();

        let logits: Tensor<B, 2> = Tensor::zeros([1, 7], &device);
        // Only one valid action per head
        let mask: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(
                vec![
                    0.0f32, 0.0, 1.0, 0.0, // head 0: only action 2
                    1.0, 0.0, 0.0, // head 1: only action 0
                ],
                [1, 7],
            ),
            &device,
        );

        let ent: f32 = dist
            .entropy(logits, Some(&mask))
            .into_data()
            .to_vec()
            .unwrap()[0];
        assert!(
            ent.abs() < 1e-4,
            "entropy should be ~0 with single valid action per head, got {ent}"
        );
    }

    #[test]
    fn from_heads_names_preserved() {
        let dist = CompositeDistribution::from_heads(&["action_type", "target"], &[3, 5]);
        assert_eq!(dist.heads[0].name, "action_type");
        assert_eq!(dist.heads[1].name, "target");
        assert_eq!(dist.total_logits(), 8);
        assert_eq!(dist.n_heads(), 2);
    }

    #[test]
    fn batch_log_prob_and_entropy() {
        // Test with batch > 1
        let dist = CompositeDistribution::new(&[2, 3]);
        let device = device();

        let logits: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(
                vec![
                    0.5f32, -0.5, 1.0, 0.0, -1.0, // batch 0
                    -0.3, 0.3, 0.0, 0.5, 0.5, // batch 1
                ],
                [2, 5],
            ),
            &device,
        );

        let actions = vec![vec![0.0f32, 1.0], vec![1.0, 0.0]];
        let lp = dist.log_prob(logits.clone(), &actions, None, &device);
        let lp_data: Vec<f32> = lp.into_data().to_vec().unwrap();
        assert_eq!(lp_data.len(), 2);
        assert!(lp_data[0] <= 0.0 && lp_data[0].is_finite());
        assert!(lp_data[1] <= 0.0 && lp_data[1].is_finite());

        let ent = dist.entropy(logits, None);
        let ent_data: Vec<f32> = ent.into_data().to_vec().unwrap();
        assert_eq!(ent_data.len(), 2);
        assert!(ent_data[0] > 0.0);
        assert!(ent_data[1] > 0.0);
    }

    // -- Gradient flow tests --------------------------------------------------

    #[test]
    fn log_prob_is_differentiable() {
        // Verify that gradients flow through log_prob back to the input logits.
        // This confirms the autoregressive distribution is differentiable
        // end-to-end for policy gradient training.
        use burn::backend::Autodiff;
        type AB = Autodiff<NdArray>;

        let device = <AB as Backend>::Device::default();
        let dist = CompositeDistribution::new(&[3, 4]);

        let logits: Tensor<AB, 2> = Tensor::from_data(
            TensorData::new(
                vec![0.5f32, -0.3, 1.0, 0.2, -0.5, 0.8, 0.1],
                [1, 7],
            ),
            &device,
        )
        .require_grad();

        let actions = vec![vec![1.0f32, 2.0]];
        let lp = dist.log_prob(logits.clone(), &actions, None, &device);
        let loss = lp.sum();
        let grads = loss.backward();

        let logit_grad = logits.grad(&grads).expect("logits should have gradients");
        let grad_data: Vec<f32> = logit_grad.into_data().to_vec().unwrap();

        // At least some gradient elements should be non-zero
        let has_nonzero = grad_data.iter().any(|&g| g.abs() > 1e-8);
        assert!(
            has_nonzero,
            "log_prob gradients should be non-zero, got {:?}",
            grad_data
        );
    }

    #[test]
    fn entropy_is_differentiable() {
        // Verify gradients flow through entropy computation.
        use burn::backend::Autodiff;
        type AB = Autodiff<NdArray>;

        let device = <AB as Backend>::Device::default();
        let dist = CompositeDistribution::new(&[3, 4]);

        let logits: Tensor<AB, 2> = Tensor::from_data(
            TensorData::new(
                vec![0.5f32, -0.3, 1.0, 0.2, -0.5, 0.8, 0.1],
                [1, 7],
            ),
            &device,
        )
        .require_grad();

        let ent = dist.entropy(logits.clone(), None);
        let loss = ent.sum();
        let grads = loss.backward();

        let logit_grad = logits.grad(&grads).expect("logits should have gradients");
        let grad_data: Vec<f32> = logit_grad.into_data().to_vec().unwrap();
        let has_nonzero = grad_data.iter().any(|&g| g.abs() > 1e-8);
        assert!(
            has_nonzero,
            "entropy gradients should be non-zero, got {:?}",
            grad_data
        );
    }

    #[test]
    fn log_prob_gradient_affects_all_heads() {
        // Verify that perturbing logits for each head changes the joint log_prob.
        // This catches gradient disconnection between heads.
        let dist = CompositeDistribution::new(&[3, 4]);
        let device = device();

        let base_logits_data = vec![0.5f32, -0.3, 1.0, 0.2, -0.5, 0.8, 0.1];
        let actions = vec![vec![1.0f32, 2.0]];

        let base_logits: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(base_logits_data.clone(), [1, 7]),
            &device,
        );
        let base_lp: f32 = dist
            .log_prob(base_logits, &actions, None, &device)
            .into_data()
            .to_vec()
            .unwrap()[0];

        // Perturb head 0 logits (indices 0..3)
        let mut perturbed = base_logits_data.clone();
        perturbed[0] += 1.0;
        let perturbed_logits: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(perturbed, [1, 7]), &device);
        let perturbed_lp: f32 = dist
            .log_prob(perturbed_logits, &actions, None, &device)
            .into_data()
            .to_vec()
            .unwrap()[0];
        assert!(
            (perturbed_lp - base_lp).abs() > 1e-4,
            "perturbing head 0 should change log_prob"
        );

        // Perturb head 1 logits (indices 3..7)
        let mut perturbed = base_logits_data;
        perturbed[4] += 1.0;
        let perturbed_logits: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(perturbed, [1, 7]), &device);
        let perturbed_lp: f32 = dist
            .log_prob(perturbed_logits, &actions, None, &device)
            .into_data()
            .to_vec()
            .unwrap()[0];
        assert!(
            (perturbed_lp - base_lp).abs() > 1e-4,
            "perturbing head 1 should change log_prob"
        );
    }
}
