//! League training orchestrator (Issue #15).
//!
//! Implements the AlphaStar-style league training with three agent roles:
//! main agents, main exploiters, and league exploiters. Each role has
//! different matchmaking strategies and checkpoint/reset behavior.

use super::pfsp::{PfspConfig, PfspMatchmaking};
use super::self_play::SelfPlayPool;
use rand::{Rng, RngExt};

// ---------------------------------------------------------------------------
// Agent roles
// ---------------------------------------------------------------------------

/// Role of an agent in league training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AgentRole {
    /// Main agent: trained against diverse opponents via PFSP.
    MainAgent,
    /// Main exploiter: focuses on beating the current main agent.
    MainExploiter,
    /// League exploiter: exploits weaknesses across the full player pool.
    LeagueExploiter,
}

// ---------------------------------------------------------------------------
// Agent config
// ---------------------------------------------------------------------------

/// Configuration for an agent in the league.
#[derive(Debug, Clone)]
pub struct LeagueAgentConfig {
    pub role: AgentRole,
    /// Steps between checkpointing this agent to the frozen pool.
    pub checkpoint_interval: u64,
    /// For exploiters: reset to initial weights after this many steps if
    /// the agent stops making progress. 0 = no reset.
    pub reset_threshold: u64,
}

// ---------------------------------------------------------------------------
// League agent
// ---------------------------------------------------------------------------

/// An agent participating in the league.
pub struct LeagueAgent<M: Clone> {
    pub config: LeagueAgentConfig,
    pub model: M,
    pub training_step: u64,
    /// Matchmaking state for this agent.
    pub matchmaking: PfspMatchmaking,
}

// ---------------------------------------------------------------------------
// League
// ---------------------------------------------------------------------------

/// League training orchestrator.
///
/// Manages multiple simultaneously training agents with different roles.
/// Each agent has its own matchmaking distribution over the frozen pool
/// of past policy snapshots.
pub struct League<M: Clone> {
    agents: Vec<LeagueAgent<M>>,
    /// Frozen opponent pool (past snapshots of all agents).
    frozen_pool: SelfPlayPool<M>,
    /// Initial model weights for exploiter resets.
    initial_model: Option<M>,
}

impl<M: Clone> League<M> {
    /// Create a new league.
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            frozen_pool: SelfPlayPool::new(),
            initial_model: None,
        }
    }

    /// Set the initial model for exploiter resets.
    pub fn set_initial_model(&mut self, model: M) {
        self.initial_model = Some(model);
    }

    /// Add an agent to the league.
    pub fn add_agent(&mut self, model: M, config: LeagueAgentConfig) -> usize {
        let matchmaking = PfspMatchmaking::new(PfspConfig::default());
        self.agents.push(LeagueAgent {
            config,
            model,
            training_step: 0,
            matchmaking,
        });
        self.agents.len() - 1
    }

    /// Get an agent's opponent for a training game.
    pub fn get_opponent(&self, agent_idx: usize, rng: &mut impl Rng) -> Option<M> {
        let agent = &self.agents[agent_idx];
        match agent.config.role {
            AgentRole::MainAgent => {
                // Main agent: mix of self-play and PFSP against full pool
                if self.frozen_pool.is_empty() || rng.random::<f32>() < 0.35 {
                    // Self-play: use own current weights
                    Some(agent.model.clone())
                } else {
                    // PFSP against frozen pool
                    self.frozen_pool.sample(rng).cloned()
                }
            }
            AgentRole::MainExploiter => {
                // Main exploiter only faces the main agent
                self.agents
                    .iter()
                    .find(|a| a.config.role == AgentRole::MainAgent)
                    .map(|a| a.model.clone())
            }
            AgentRole::LeagueExploiter => {
                // League exploiter: PFSP against full pool
                self.frozen_pool
                    .sample(rng)
                    .cloned()
                    .or_else(|| Some(agent.model.clone()))
            }
        }
    }

    /// Update an agent after a training step.
    /// Handles checkpointing to the frozen pool.
    pub fn update_agent(&mut self, agent_idx: usize) {
        self.agents[agent_idx].training_step += 1;

        let step = self.agents[agent_idx].training_step;
        let interval = self.agents[agent_idx].config.checkpoint_interval;

        // Checkpoint to frozen pool
        if interval > 0 && step % interval == 0 {
            let model = self.agents[agent_idx].model.clone();
            let id = self.frozen_pool.add_snapshot(&model, step);
            // Add to all agents' matchmaking
            for agent in &mut self.agents {
                agent.matchmaking.add_opponent(id);
            }
        }
    }

    /// Reset an exploiter to initial weights.
    pub fn reset_exploiter(&mut self, agent_idx: usize) {
        if let Some(ref initial) = self.initial_model {
            self.agents[agent_idx].model = initial.clone();
            self.agents[agent_idx].training_step = 0;
        }
    }

    /// Get reference to an agent.
    pub fn agent(&self, idx: usize) -> &LeagueAgent<M> {
        &self.agents[idx]
    }

    /// Get mutable reference to an agent's model.
    pub fn agent_model_mut(&mut self, idx: usize) -> &mut M {
        &mut self.agents[idx].model
    }

    /// Number of agents.
    pub fn n_agents(&self) -> usize {
        self.agents.len()
    }

    /// Number of frozen snapshots.
    pub fn n_frozen(&self) -> usize {
        self.frozen_pool.len()
    }

    /// Get agent indices by role.
    pub fn agents_with_role(&self, role: AgentRole) -> Vec<usize> {
        self.agents
            .iter()
            .enumerate()
            .filter(|(_, a)| a.config.role == role)
            .map(|(i, _)| i)
            .collect()
    }
}

impl<M: Clone> Default for League<M> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn main_config() -> LeagueAgentConfig {
        LeagueAgentConfig {
            role: AgentRole::MainAgent,
            checkpoint_interval: 10,
            reset_threshold: 0,
        }
    }

    fn exploiter_config(role: AgentRole) -> LeagueAgentConfig {
        LeagueAgentConfig {
            role,
            checkpoint_interval: 20,
            reset_threshold: 100,
        }
    }

    #[test]
    fn agent_count_and_roles() {
        let mut league: League<i32> = League::new();
        league.add_agent(0, main_config());
        league.add_agent(1, exploiter_config(AgentRole::MainExploiter));
        league.add_agent(2, exploiter_config(AgentRole::LeagueExploiter));

        assert_eq!(league.n_agents(), 3);
        assert_eq!(league.agents_with_role(AgentRole::MainAgent), vec![0]);
        assert_eq!(league.agents_with_role(AgentRole::MainExploiter), vec![1]);
        assert_eq!(league.agents_with_role(AgentRole::LeagueExploiter), vec![2]);
    }

    #[test]
    fn main_exploiter_only_faces_main_agent() {
        let mut league: League<i32> = League::new();
        league.add_agent(100, main_config());
        league.add_agent(200, exploiter_config(AgentRole::MainExploiter));

        let mut rng = rand::rng();
        for _ in 0..20 {
            let opponent = league.get_opponent(1, &mut rng).unwrap();
            assert_eq!(
                opponent, 100,
                "main exploiter should only face main agent, got {opponent}"
            );
        }
    }

    #[test]
    fn checkpoint_adds_to_pool() {
        let mut league: League<i32> = League::new();
        league.add_agent(42, LeagueAgentConfig {
            role: AgentRole::MainAgent,
            checkpoint_interval: 5,
            reset_threshold: 0,
        });

        assert_eq!(league.n_frozen(), 0);

        // Step 5 times → should checkpoint once
        for _ in 0..5 {
            league.update_agent(0);
        }
        assert_eq!(league.n_frozen(), 1);

        // Step 5 more → second checkpoint
        for _ in 0..5 {
            league.update_agent(0);
        }
        assert_eq!(league.n_frozen(), 2);
    }

    #[test]
    fn exploiter_reset_restores_initial_model() {
        let mut league: League<i32> = League::new();
        league.set_initial_model(0);
        league.add_agent(99, exploiter_config(AgentRole::MainExploiter));

        // Simulate some training
        *league.agent_model_mut(0) = 999;
        assert_eq!(league.agent(0).model, 999);

        // Reset
        league.reset_exploiter(0);
        assert_eq!(league.agent(0).model, 0);
        assert_eq!(league.agent(0).training_step, 0);
    }

    #[test]
    fn empty_league() {
        let league: League<i32> = League::new();
        assert_eq!(league.n_agents(), 0);
        assert_eq!(league.n_frozen(), 0);
    }

    #[test]
    fn main_agent_self_play_when_pool_empty() {
        let mut league: League<i32> = League::new();
        league.add_agent(42, main_config());

        let mut rng = rand::rng();
        // Pool is empty, so main agent should self-play
        let opponent = league.get_opponent(0, &mut rng).unwrap();
        assert_eq!(opponent, 42);
    }

    #[test]
    fn league_exploiter_falls_back_to_self_play() {
        let mut league: League<i32> = League::new();
        league.add_agent(77, exploiter_config(AgentRole::LeagueExploiter));

        let mut rng = rand::rng();
        // Pool is empty, so league exploiter should self-play
        let opponent = league.get_opponent(0, &mut rng).unwrap();
        assert_eq!(opponent, 77);
    }
}
