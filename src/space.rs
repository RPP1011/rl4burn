//! Action and observation space descriptions.

/// Description of an action or observation space.
#[derive(Debug, Clone)]
pub enum Space {
    /// Discrete space {0, 1, ..., n-1}.
    Discrete(usize),
    /// Continuous box with per-dimension bounds.
    Box {
        low: Vec<f32>,
        high: Vec<f32>,
    },
    /// Multiple independent discrete sub-spaces.
    MultiDiscrete(Vec<usize>),
}

impl Space {
    /// Total flat dimension of the space.
    ///
    /// For `Discrete(n)`, this is `n` (one-hot width).
    /// For `Box { low, .. }`, this is the number of dimensions.
    /// For `MultiDiscrete(nvec)`, this is the sum of all sub-space sizes.
    pub fn flat_dim(&self) -> usize {
        match self {
            Space::Discrete(n) => *n,
            Space::Box { low, .. } => low.len(),
            Space::MultiDiscrete(nvec) => nvec.iter().sum(),
        }
    }

    /// Shape as a vector of dimension sizes.
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Space::Discrete(n) => vec![*n],
            Space::Box { low, .. } => vec![low.len()],
            Space::MultiDiscrete(nvec) => nvec.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_flat_dim() {
        assert_eq!(Space::Discrete(4).flat_dim(), 4);
    }

    #[test]
    fn box_flat_dim() {
        let s = Space::Box {
            low: vec![-1.0; 3],
            high: vec![1.0; 3],
        };
        assert_eq!(s.flat_dim(), 3);
    }

    #[test]
    fn multi_discrete_flat_dim() {
        assert_eq!(Space::MultiDiscrete(vec![3, 5, 2]).flat_dim(), 10);
    }
}
