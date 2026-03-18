//! Curriculum Self-Play Learning orchestrator (Issue #18).
//!
//! Implements a three-phase training pipeline:
//! 1. **Specialist Training** — Train fixed-composition specialist teachers
//!    via self-play.
//! 2. **Distillation** — Distill multiple teachers into a single student.
//! 3. **Generalization** — Continue RL with random compositions.

// The League, AgentRole, and LeagueAgentConfig types from crate::algo::league
// are used alongside CSPL in practice (Phase 1 specialist training leverages
// the league infrastructure), but are not directly referenced in the state
// machine itself.

// ---------------------------------------------------------------------------
// Phase enum
// ---------------------------------------------------------------------------

/// Phase of the CSPL pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CsplPhase {
    /// Phase 1: Train fixed-composition specialist teachers via self-play.
    SpecialistTraining,
    /// Phase 2: Distill multiple teachers into a single student.
    Distillation,
    /// Phase 3: Continue RL with random compositions.
    Generalization,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the CSPL pipeline.
#[derive(Debug, Clone)]
pub struct CsplConfig {
    /// Steps for Phase 1 (specialist training).
    pub phase1_steps: u64,
    /// Steps for Phase 2 (distillation).
    pub phase2_steps: u64,
    /// Steps for Phase 3 (generalization). 0 = unlimited.
    pub phase3_steps: u64,
    /// Number of specialist compositions in Phase 1.
    pub n_specialists: usize,
}

impl Default for CsplConfig {
    fn default() -> Self {
        Self {
            phase1_steps: 100_000,
            phase2_steps: 50_000,
            phase3_steps: 0,
            n_specialists: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline state machine
// ---------------------------------------------------------------------------

/// CSPL pipeline state machine.
///
/// Tracks the current phase and step count, automatically transitioning
/// between phases when step budgets are exhausted.
pub struct CsplPipeline {
    config: CsplConfig,
    current_phase: CsplPhase,
    phase_step: u64,
}

impl CsplPipeline {
    /// Create a new pipeline starting at Phase 1 (SpecialistTraining).
    pub fn new(config: CsplConfig) -> Self {
        Self {
            config,
            current_phase: CsplPhase::SpecialistTraining,
            phase_step: 0,
        }
    }

    /// Current phase of the pipeline.
    pub fn current_phase(&self) -> CsplPhase {
        self.current_phase
    }

    /// Number of steps taken in the current phase.
    pub fn phase_step(&self) -> u64 {
        self.phase_step
    }

    /// Advance one training step. Returns `true` if the phase changed.
    pub fn step(&mut self) -> bool {
        self.phase_step += 1;
        match self.current_phase {
            CsplPhase::SpecialistTraining => {
                if self.phase_step >= self.config.phase1_steps {
                    self.current_phase = CsplPhase::Distillation;
                    self.phase_step = 0;
                    return true;
                }
            }
            CsplPhase::Distillation => {
                if self.phase_step >= self.config.phase2_steps {
                    self.current_phase = CsplPhase::Generalization;
                    self.phase_step = 0;
                    return true;
                }
            }
            CsplPhase::Generalization => {
                // Phase 3 runs indefinitely (or until phase3_steps)
                if self.config.phase3_steps > 0
                    && self.phase_step >= self.config.phase3_steps
                {
                    return true;
                }
            }
        }
        false
    }

    /// Check if the pipeline is complete.
    ///
    /// The pipeline is complete when Phase 3 has run for `phase3_steps`
    /// (only possible when `phase3_steps > 0`).
    pub fn is_complete(&self) -> bool {
        self.current_phase == CsplPhase::Generalization
            && self.config.phase3_steps > 0
            && self.phase_step >= self.config.phase3_steps
    }

    /// Get the number of specialists for Phase 1.
    pub fn n_specialists(&self) -> usize {
        self.config.n_specialists
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CsplConfig {
        CsplConfig {
            phase1_steps: 10,
            phase2_steps: 5,
            phase3_steps: 3,
            n_specialists: 2,
        }
    }

    #[test]
    fn starts_in_specialist_training() {
        let pipeline = CsplPipeline::new(default_config());
        assert_eq!(pipeline.current_phase(), CsplPhase::SpecialistTraining);
        assert_eq!(pipeline.phase_step(), 0);
    }

    #[test]
    fn transitions_at_correct_steps() {
        let mut pipeline = CsplPipeline::new(default_config());

        // Phase 1: 10 steps
        for i in 0..9 {
            let changed = pipeline.step();
            assert!(!changed, "should not change at step {}", i + 1);
            assert_eq!(pipeline.current_phase(), CsplPhase::SpecialistTraining);
        }
        // Step 10 triggers transition
        let changed = pipeline.step();
        assert!(changed, "should transition at step 10");
        assert_eq!(pipeline.current_phase(), CsplPhase::Distillation);
        assert_eq!(pipeline.phase_step(), 0);

        // Phase 2: 5 steps
        for _ in 0..4 {
            let changed = pipeline.step();
            assert!(!changed);
            assert_eq!(pipeline.current_phase(), CsplPhase::Distillation);
        }
        let changed = pipeline.step();
        assert!(changed, "should transition at step 5");
        assert_eq!(pipeline.current_phase(), CsplPhase::Generalization);
        assert_eq!(pipeline.phase_step(), 0);

        // Phase 3: 3 steps
        for _ in 0..2 {
            let changed = pipeline.step();
            assert!(!changed);
        }
        let changed = pipeline.step();
        assert!(changed, "should signal completion at step 3");
    }

    #[test]
    fn is_complete_with_finite_phase3() {
        let mut pipeline = CsplPipeline::new(default_config());
        assert!(!pipeline.is_complete());

        // Run through all phases
        for _ in 0..10 {
            pipeline.step();
        }
        assert_eq!(pipeline.current_phase(), CsplPhase::Distillation);
        assert!(!pipeline.is_complete());

        for _ in 0..5 {
            pipeline.step();
        }
        assert_eq!(pipeline.current_phase(), CsplPhase::Generalization);
        assert!(!pipeline.is_complete());

        for _ in 0..3 {
            pipeline.step();
        }
        assert!(pipeline.is_complete());
    }

    #[test]
    fn is_complete_never_with_unlimited_phase3() {
        let config = CsplConfig {
            phase1_steps: 2,
            phase2_steps: 2,
            phase3_steps: 0, // unlimited
            n_specialists: 1,
        };
        let mut pipeline = CsplPipeline::new(config);

        // Run through phases 1 and 2
        for _ in 0..4 {
            pipeline.step();
        }
        assert_eq!(pipeline.current_phase(), CsplPhase::Generalization);

        // Run many steps in phase 3 — never completes
        for _ in 0..1000 {
            pipeline.step();
            assert!(!pipeline.is_complete());
        }
    }

    #[test]
    fn n_specialists_from_config() {
        let config = CsplConfig {
            n_specialists: 7,
            ..Default::default()
        };
        let pipeline = CsplPipeline::new(config);
        assert_eq!(pipeline.n_specialists(), 7);
    }
}
