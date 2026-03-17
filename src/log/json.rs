//! JSONL logger for integration with external tools (wandb, mlflow, etc.).

use super::Logger;

use std::io::{BufWriter, Write};
use std::time::{SystemTime, UNIX_EPOCH};

/// Writes one JSON object per event to a `Write` sink.
///
/// Output format (one per line):
/// ```json
/// {"type":"scalar","key":"train/loss","value":0.42,"step":1000,"wall_time":1234567890.123}
/// ```
pub struct JsonLogger {
    writer: BufWriter<Box<dyn Write>>,
}

impl JsonLogger {
    /// Create a logger writing to the given sink (file, stdout, etc.).
    pub fn new(writer: Box<dyn Write>) -> Self {
        Self {
            writer: BufWriter::new(writer),
        }
    }

    /// Create a logger writing to a file at the given path.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let file = std::fs::File::create(path)?;
        Ok(Self::new(Box::new(file)))
    }

    fn wall_time() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }

    fn write_line(&mut self, line: &str) {
        let _ = writeln!(self.writer, "{line}");
    }
}

impl Logger for JsonLogger {
    fn log_scalar(&mut self, key: &str, value: f64, step: u64) {
        let wt = Self::wall_time();
        // Hand-format to avoid serde_json dependency for simple cases
        let line = format!(
            r#"{{"type":"scalar","key":"{key}","value":{value},"step":{step},"wall_time":{wt:.3}}}"#
        );
        self.write_line(&line);
    }

    fn log_scalars(&mut self, key: &str, values: &[(&str, f64)], step: u64) {
        let wt = Self::wall_time();
        let pairs: Vec<String> = values
            .iter()
            .map(|(k, v)| format!(r#""{k}":{v}"#))
            .collect();
        let line = format!(
            r#"{{"type":"scalars","key":"{key}","values":{{{inner}}},"step":{step},"wall_time":{wt:.3}}}"#,
            inner = pairs.join(",")
        );
        self.write_line(&line);
    }

    fn log_text(&mut self, key: &str, text: &str, step: u64) {
        let wt = Self::wall_time();
        // Escape basic JSON characters in text
        let escaped = text
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        let line = format!(
            r#"{{"type":"text","key":"{key}","text":"{escaped}","step":{step},"wall_time":{wt:.3}}}"#
        );
        self.write_line(&line);
    }

    fn log_histogram(&mut self, key: &str, values: &[f32], step: u64) {
        let wt = Self::wall_time();
        let vals: Vec<String> = values.iter().map(|v| format!("{v}")).collect();
        let line = format!(
            r#"{{"type":"histogram","key":"{key}","values":[{inner}],"step":{step},"wall_time":{wt:.3}}}"#,
            inner = vals.join(",")
        );
        self.write_line(&line);
    }

    fn flush(&mut self) {
        let _ = self.writer.flush();
    }
}
