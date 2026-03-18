//! Logging infrastructure for training metrics.
//!
//! Provides a [`Logger`] trait with built-in implementations:
//! - [`PrintLogger`] — formatted output to stderr (always available)
//! - [`NoopLogger`] — discards all events
//! - [`CompositeLogger`] — fans out to multiple loggers
//!
//! Feature-gated implementations:
//! - [`TensorBoardLogger`] — writes TFEvent files (`tensorboard` feature)
//! - [`JsonLogger`] — writes JSONL to any `Write` sink (`json-log` feature)

mod stats;

#[cfg(feature = "tensorboard")]
mod tensorboard;

#[cfg(feature = "json-log")]
mod json;

#[cfg(feature = "video")]
mod video;

pub use stats::Loggable;

#[cfg(feature = "tensorboard")]
pub use tensorboard::TensorBoardLogger;

#[cfg(feature = "json-log")]
pub use json::JsonLogger;

#[cfg(feature = "video")]
pub use video::write_gif;

// ---------------------------------------------------------------------------
// Logger trait
// ---------------------------------------------------------------------------

/// Trait for logging training metrics.
pub trait Logger {
    /// Log a single scalar value.
    fn log_scalar(&mut self, key: &str, value: f64, step: u64);

    /// Log multiple related scalars under a group key.
    fn log_scalars(&mut self, key: &str, values: &[(&str, f64)], step: u64);

    /// Log a text string.
    fn log_text(&mut self, key: &str, text: &str, step: u64);

    /// Log a histogram from raw values.
    fn log_histogram(&mut self, key: &str, values: &[f32], step: u64);

    /// Flush any buffered data.
    fn flush(&mut self);
}

// ---------------------------------------------------------------------------
// PrintLogger
// ---------------------------------------------------------------------------

/// Logs scalars to stderr in a formatted table.
pub struct PrintLogger {
    /// Minimum step interval between prints (0 = print every call).
    pub every: u64,
    last_step: u64,
    first: bool,
}

impl PrintLogger {
    /// Create a new `PrintLogger` that prints every `every` steps.
    pub fn new(every: u64) -> Self {
        Self {
            every,
            last_step: 0,
            first: true,
        }
    }
}

impl Default for PrintLogger {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Logger for PrintLogger {
    fn log_scalar(&mut self, key: &str, value: f64, step: u64) {
        if self.every > 0 && !self.first && step < self.last_step + self.every {
            return;
        }
        self.first = false;
        self.last_step = step;
        eprintln!("[step {step:>8}] {key}: {value:.4}");
    }

    fn log_scalars(&mut self, key: &str, values: &[(&str, f64)], step: u64) {
        if self.every > 0 && !self.first && step < self.last_step + self.every {
            return;
        }
        self.first = false;
        self.last_step = step;
        let parts: Vec<String> = values.iter().map(|(k, v)| format!("{k}={v:.4}")).collect();
        eprintln!("[step {step:>8}] {key} | {}", parts.join(" | "));
    }

    fn log_text(&mut self, key: &str, text: &str, step: u64) {
        eprintln!("[step {step:>8}] {key}: {text}");
    }

    fn log_histogram(&mut self, _key: &str, _values: &[f32], _step: u64) {
        // PrintLogger does not render histograms
    }

    fn flush(&mut self) {}
}

// ---------------------------------------------------------------------------
// NoopLogger
// ---------------------------------------------------------------------------

/// Discards all logged events.
pub struct NoopLogger;

impl Logger for NoopLogger {
    fn log_scalar(&mut self, _key: &str, _value: f64, _step: u64) {}
    fn log_scalars(&mut self, _key: &str, _values: &[(&str, f64)], _step: u64) {}
    fn log_text(&mut self, _key: &str, _text: &str, _step: u64) {}
    fn log_histogram(&mut self, _key: &str, _values: &[f32], _step: u64) {}
    fn flush(&mut self) {}
}

// ---------------------------------------------------------------------------
// CompositeLogger
// ---------------------------------------------------------------------------

/// Fans out log events to multiple loggers.
pub struct CompositeLogger {
    loggers: Vec<Box<dyn Logger>>,
}

impl CompositeLogger {
    pub fn new(loggers: Vec<Box<dyn Logger>>) -> Self {
        Self { loggers }
    }
}

impl Logger for CompositeLogger {
    fn log_scalar(&mut self, key: &str, value: f64, step: u64) {
        for logger in &mut self.loggers {
            logger.log_scalar(key, value, step);
        }
    }

    fn log_scalars(&mut self, key: &str, values: &[(&str, f64)], step: u64) {
        for logger in &mut self.loggers {
            logger.log_scalars(key, values, step);
        }
    }

    fn log_text(&mut self, key: &str, text: &str, step: u64) {
        for logger in &mut self.loggers {
            logger.log_text(key, text, step);
        }
    }

    fn log_histogram(&mut self, key: &str, values: &[f32], step: u64) {
        for logger in &mut self.loggers {
            logger.log_histogram(key, values, step);
        }
    }

    fn flush(&mut self) {
        for logger in &mut self.loggers {
            logger.flush();
        }
    }
}
