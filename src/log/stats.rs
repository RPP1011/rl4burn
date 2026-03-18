//! [`Loggable`] trait and implementations for algorithm stats structs.

use super::Logger;
use crate::algo::base::dqn::DqnStats;
use crate::algo::base::ppo::PpoStats;

/// Types that can log their fields to a [`Logger`].
pub trait Loggable {
    fn log(&self, logger: &mut dyn Logger, step: u64);
}

impl Loggable for PpoStats {
    fn log(&self, logger: &mut dyn Logger, step: u64) {
        logger.log_scalar("train/policy_loss", self.policy_loss as f64, step);
        logger.log_scalar("train/value_loss", self.value_loss as f64, step);
        logger.log_scalar("train/entropy", self.entropy as f64, step);
        logger.log_scalar("train/approx_kl", self.approx_kl as f64, step);
    }
}

impl Loggable for DqnStats {
    fn log(&self, logger: &mut dyn Logger, step: u64) {
        logger.log_scalar("train/loss", self.loss as f64, step);
        logger.log_scalar("train/mean_q", self.mean_q as f64, step);
        logger.log_scalar("train/epsilon", self.epsilon as f64, step);
    }
}
