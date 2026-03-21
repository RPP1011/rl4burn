pub mod base;
pub mod dreamer;
pub mod imitation;
pub mod multi_agent;
pub mod planning;

pub mod distributed;
pub mod privileged_critic;
pub mod z_conditioning;

pub mod loss;

// ---------------------------------------------------------------------------
// Loggable implementations
// ---------------------------------------------------------------------------

use rl4burn_core::log::{Loggable, Logger};

impl Loggable for crate::base::ppo::PpoStats {
    fn log(&self, logger: &mut dyn Logger, step: u64) {
        logger.log_scalar("train/policy_loss", self.policy_loss as f64, step);
        logger.log_scalar("train/value_loss", self.value_loss as f64, step);
        logger.log_scalar("train/entropy", self.entropy as f64, step);
        logger.log_scalar("train/approx_kl", self.approx_kl as f64, step);
    }
}

impl Loggable for crate::base::dqn::DqnStats {
    fn log(&self, logger: &mut dyn Logger, step: u64) {
        logger.log_scalar("train/loss", self.loss as f64, step);
        logger.log_scalar("train/mean_q", self.mean_q as f64, step);
        logger.log_scalar("train/epsilon", self.epsilon as f64, step);
    }
}
