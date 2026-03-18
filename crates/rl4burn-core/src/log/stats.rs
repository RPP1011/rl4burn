//! [`Loggable`] trait for algorithm stats structs.

use super::Logger;

/// Types that can log their fields to a [`Logger`].
pub trait Loggable {
    fn log(&self, logger: &mut dyn Logger, step: u64);
}
