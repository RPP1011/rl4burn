//! Prioritized Fictitious Self-Play (PFSP) matchmaking (Issue #14).
//!
//! Provides a [`PfspMatchmaking`] opponent pool that tracks win rates and
//! samples harder opponents more frequently. Used by league training to
//! focus training compute on opponents the agent currently struggles against.

use rand::Rng;

// ---------------------------------------------------------------------------
// Player record
// ---------------------------------------------------------------------------

/// Win-rate record for a player in the opponent pool.
#[derive(Debug, Clone)]
pub struct PlayerRecord {
    pub id: u64,
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
}

impl PlayerRecord {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            wins: 0,
            losses: 0,
            draws: 0,
        }
    }

    pub fn total_games(&self) -> u64 {
        self.wins + self.losses + self.draws
    }

    /// Win rate from the perspective of the current agent.
    /// Returns 0.5 (prior) when no games have been played.
    pub fn win_rate(&self) -> f64 {
        let total = self.total_games();
        if total == 0 {
            0.5
        } else {
            self.wins as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// PFSP matchmaking configuration.
#[derive(Debug, Clone)]
pub struct PfspConfig {
    /// Exponent for weighting. Higher = more focus on hard opponents.
    /// `f(x) = (1 - win_rate)^p`. Default: 1.0
    pub power: f64,
    /// Minimum selection probability for any opponent. Default: 0.01
    pub min_prob: f64,
}

impl Default for PfspConfig {
    fn default() -> Self {
        Self {
            power: 1.0,
            min_prob: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// Matchmaking
// ---------------------------------------------------------------------------

/// PFSP matchmaking: opponent pool with win-rate tracking.
///
/// Harder opponents (lower win rate against the current agent) are sampled
/// more frequently, focusing training on weaknesses.
pub struct PfspMatchmaking {
    config: PfspConfig,
    /// Win rates of the current agent against each opponent.
    records: Vec<PlayerRecord>,
}

impl PfspMatchmaking {
    pub fn new(config: PfspConfig) -> Self {
        Self {
            config,
            records: Vec::new(),
        }
    }

    /// Add a new opponent to the pool.
    pub fn add_opponent(&mut self, id: u64) {
        self.records.push(PlayerRecord::new(id));
    }

    /// Record game result (from perspective of current agent).
    pub fn record_result(&mut self, opponent_id: u64, win: bool, draw: bool) {
        if let Some(record) = self.records.iter_mut().find(|r| r.id == opponent_id) {
            if draw {
                record.draws += 1;
            } else if win {
                record.wins += 1;
            } else {
                record.losses += 1;
            }
        }
    }

    /// Sample an opponent using PFSP weighting.
    /// Harder opponents (lower win rate) are sampled more frequently.
    pub fn sample_opponent(&self, rng: &mut impl Rng) -> Option<u64> {
        if self.records.is_empty() {
            return None;
        }

        let weights: Vec<f64> = self
            .records
            .iter()
            .map(|r| {
                (1.0 - r.win_rate())
                    .powf(self.config.power)
                    .max(self.config.min_prob)
            })
            .collect();

        let total: f64 = weights.iter().sum();
        let probs: Vec<f64> = weights.iter().map(|w| w / total).collect();

        let u: f64 = rng.random::<f64>();
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if u < cum {
                return Some(self.records[i].id);
            }
        }
        Some(self.records.last().unwrap().id)
    }

    /// Get all records.
    pub fn records(&self) -> &[PlayerRecord] {
        &self.records
    }

    /// Get selection probabilities.
    pub fn selection_probs(&self) -> Vec<f64> {
        if self.records.is_empty() {
            return vec![];
        }
        let weights: Vec<f64> = self
            .records
            .iter()
            .map(|r| {
                (1.0 - r.win_rate())
                    .powf(self.config.power)
                    .max(self.config.min_prob)
            })
            .collect();
        let total: f64 = weights.iter().sum();
        weights.iter().map(|w| w / total).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_probs_sum_to_one() {
        let mut mm = PfspMatchmaking::new(PfspConfig::default());
        mm.add_opponent(0);
        mm.add_opponent(1);
        mm.add_opponent(2);

        // Record some results: agent beats opponent 0 often, loses to opponent 2
        for _ in 0..10 {
            mm.record_result(0, true, false);
        }
        for _ in 0..5 {
            mm.record_result(1, true, false);
            mm.record_result(1, false, false);
        }
        for _ in 0..10 {
            mm.record_result(2, false, false);
        }

        let probs = mm.selection_probs();
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "probs sum to {sum}, expected 1.0");
    }

    #[test]
    fn harder_opponents_sampled_more() {
        let mut mm = PfspMatchmaking::new(PfspConfig::default());
        mm.add_opponent(0); // easy: 90% win rate
        mm.add_opponent(1); // hard: 10% win rate

        for _ in 0..9 {
            mm.record_result(0, true, false);
        }
        mm.record_result(0, false, false);

        mm.record_result(1, true, false);
        for _ in 0..9 {
            mm.record_result(1, false, false);
        }

        let probs = mm.selection_probs();
        // Opponent 1 (hard) should have higher probability than opponent 0 (easy)
        assert!(
            probs[1] > probs[0],
            "hard opponent prob {:.3} should exceed easy opponent prob {:.3}",
            probs[1],
            probs[0]
        );
    }

    #[test]
    fn sample_returns_none_when_empty() {
        let mm = PfspMatchmaking::new(PfspConfig::default());
        let mut rng = rand::rng();
        assert!(mm.sample_opponent(&mut rng).is_none());
    }

    #[test]
    fn sample_returns_valid_id() {
        let mut mm = PfspMatchmaking::new(PfspConfig::default());
        mm.add_opponent(42);
        mm.add_opponent(99);

        let mut rng = rand::rng();
        for _ in 0..20 {
            let id = mm.sample_opponent(&mut rng).unwrap();
            assert!(id == 42 || id == 99, "unexpected id {id}");
        }
    }

    #[test]
    fn win_rate_default_is_half() {
        let record = PlayerRecord::new(0);
        assert!((record.win_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn draws_counted_in_total() {
        let mut mm = PfspMatchmaking::new(PfspConfig::default());
        mm.add_opponent(0);
        mm.record_result(0, false, true); // draw
        mm.record_result(0, false, true); // draw
        let record = &mm.records()[0];
        assert_eq!(record.total_games(), 2);
        assert_eq!(record.draws, 2);
        assert!((record.win_rate() - 0.0).abs() < 1e-10);
    }
}
