//! TensorBoard TFEvent file writer.
//!
//! Writes `events.out.tfevents.{timestamp}.{hostname}` files that TensorBoard
//! reads natively. Protobufs are hand-serialized to avoid pulling in `prost`.

use super::Logger;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// CRC32C (masked, as TFRecord requires)
// ---------------------------------------------------------------------------

fn masked_crc32c(data: &[u8]) -> u32 {
    let crc = crc32c::crc32c(data);
    // TFRecord mask: ((crc >> 15) | (crc << 17)) + 0xa282ead8
    ((crc >> 15) | (crc << 17)).wrapping_add(0xa282ead8)
}

// ---------------------------------------------------------------------------
// Protobuf encoding helpers (wire format, no schema needed)
// ---------------------------------------------------------------------------

/// Encode a varint (LEB128).
fn encode_varint(buf: &mut Vec<u8>, mut val: u64) {
    loop {
        let byte = (val & 0x7f) as u8;
        val >>= 7;
        if val == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

/// Encode a protobuf tag (field number + wire type).
fn encode_tag(buf: &mut Vec<u8>, field: u32, wire_type: u32) {
    encode_varint(buf, ((field as u64) << 3) | wire_type as u64);
}

/// Encode a double field (wire type 1 = 64-bit).
fn encode_double(buf: &mut Vec<u8>, field: u32, val: f64) {
    encode_tag(buf, field, 1);
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Encode a float field (wire type 5 = 32-bit).
fn encode_float(buf: &mut Vec<u8>, field: u32, val: f32) {
    encode_tag(buf, field, 5);
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Encode a bytes/string field (wire type 2 = length-delimited).
fn encode_bytes(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
    encode_tag(buf, field, 2);
    encode_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

/// Encode an int64 field (wire type 0 = varint).
fn encode_int64(buf: &mut Vec<u8>, field: u32, val: i64) {
    encode_tag(buf, field, 0);
    encode_varint(buf, val as u64);
}

// ---------------------------------------------------------------------------
// TFEvent construction
// ---------------------------------------------------------------------------

/// Build a serialized `Event` protobuf with a scalar `Summary`.
///
/// Event proto:
///   field 1 (double): wall_time
///   field 2 (int64): step
///   field 5 (message): summary
///
/// Summary proto:
///   field 1 (repeated message): value
///
/// Summary.Value proto:
///   field 1 (string): tag
///   field 2 (float): simple_value
fn build_scalar_event(wall_time: f64, step: i64, tag: &str, value: f32) -> Vec<u8> {
    // Inner: Summary.Value
    let mut sv = Vec::new();
    encode_bytes(&mut sv, 1, tag.as_bytes());
    encode_float(&mut sv, 2, value);

    // Summary
    let mut summary = Vec::new();
    encode_bytes(&mut summary, 1, &sv);

    // Event
    let mut event = Vec::new();
    encode_double(&mut event, 1, wall_time);
    encode_int64(&mut event, 2, step);
    encode_bytes(&mut event, 5, &summary);

    event
}

/// Build a serialized `Event` protobuf with a text `Summary`.
///
/// Uses the tensor-based text encoding that TensorBoard expects:
///   Summary.Value.tag = tag
///   Summary.Value.metadata.plugin_data.plugin_name = "text"
///   Summary.Value.tensor = string_val containing the text
fn build_text_event(wall_time: f64, step: i64, tag: &str, text: &str) -> Vec<u8> {
    // SummaryMetadata.PluginData: field 1 = plugin_name
    let mut plugin_data = Vec::new();
    encode_bytes(&mut plugin_data, 1, b"text");

    // SummaryMetadata: field 1 = plugin_data
    let mut metadata = Vec::new();
    encode_bytes(&mut metadata, 1, &plugin_data);

    // TensorProto with string_val (field 8) and dtype=DT_STRING (field 1, value 7)
    let mut tensor = Vec::new();
    encode_int64(&mut tensor, 1, 7); // DT_STRING
    encode_bytes(&mut tensor, 8, text.as_bytes());

    // Summary.Value: tag(1), metadata(9), tensor(8)
    let mut sv = Vec::new();
    encode_bytes(&mut sv, 1, tag.as_bytes());
    encode_bytes(&mut sv, 9, &metadata);
    encode_bytes(&mut sv, 8, &tensor);

    // Summary: value(1)
    let mut summary = Vec::new();
    encode_bytes(&mut summary, 1, &sv);

    // Event
    let mut event = Vec::new();
    encode_double(&mut event, 1, wall_time);
    encode_int64(&mut event, 2, step);
    encode_bytes(&mut event, 5, &summary);

    event
}

/// Build a serialized `Event` protobuf with a histogram `Summary`.
///
/// HistogramProto:
///   field 1 (double): min
///   field 2 (double): max
///   field 3 (double): num
///   field 4 (double): sum
///   field 5 (double): sum_squares
///   field 6 (repeated double): bucket_limit
///   field 7 (repeated double): bucket
fn build_histogram_event(wall_time: f64, step: i64, tag: &str, values: &[f32]) -> Vec<u8> {
    let (histo_min, histo_max, histo_sum, histo_sum_sq) = if values.is_empty() {
        (0.0f64, 0.0, 0.0, 0.0)
    } else {
        let mut mn = f64::MAX;
        let mut mx = f64::MIN;
        let mut s = 0.0f64;
        let mut ss = 0.0f64;
        for &v in values {
            let v = v as f64;
            mn = mn.min(v);
            mx = mx.max(v);
            s += v;
            ss += v * v;
        }
        (mn, mx, s, ss)
    };

    // Generate ~30 exponential buckets covering the data range
    let n_buckets = 30usize;
    let mut limits = Vec::with_capacity(n_buckets + 1);
    let mut counts = vec![0.0f64; n_buckets + 1];

    if !values.is_empty() && histo_max > histo_min {
        let step_size = (histo_max - histo_min) / n_buckets as f64;
        for i in 0..n_buckets {
            limits.push(histo_min + step_size * (i + 1) as f64);
        }
        limits.push(f64::MAX);

        for &v in values {
            let v = v as f64;
            let idx = limits.iter().position(|&l| v <= l).unwrap_or(n_buckets);
            counts[idx] += 1.0;
        }
    } else {
        limits.push(f64::MAX);
        counts = vec![values.len() as f64];
    }

    // HistogramProto
    let mut histo = Vec::new();
    encode_double(&mut histo, 1, histo_min);
    encode_double(&mut histo, 2, histo_max);
    encode_double(&mut histo, 3, values.len() as f64);
    encode_double(&mut histo, 4, histo_sum);
    encode_double(&mut histo, 5, histo_sum_sq);
    for &l in &limits {
        encode_double(&mut histo, 6, l);
    }
    for &c in &counts {
        encode_double(&mut histo, 7, c);
    }

    // Summary.Value: tag(1), histo(5)
    let mut sv = Vec::new();
    encode_bytes(&mut sv, 1, tag.as_bytes());
    encode_bytes(&mut sv, 5, &histo);

    // Summary
    let mut summary = Vec::new();
    encode_bytes(&mut summary, 1, &sv);

    // Event
    let mut event = Vec::new();
    encode_double(&mut event, 1, wall_time);
    encode_int64(&mut event, 2, step);
    encode_bytes(&mut event, 5, &summary);

    event
}

// ---------------------------------------------------------------------------
// TFRecord framing
// ---------------------------------------------------------------------------

/// Write a TFRecord: [u64 len][u32 crc(len)][data][u32 crc(data)]
fn write_record(writer: &mut impl Write, data: &[u8]) -> std::io::Result<()> {
    let len = data.len() as u64;
    let len_bytes = len.to_le_bytes();
    let len_crc = masked_crc32c(&len_bytes);
    let data_crc = masked_crc32c(data);

    writer.write_all(&len_bytes)?;
    writer.write_all(&len_crc.to_le_bytes())?;
    writer.write_all(data)?;
    writer.write_all(&data_crc.to_le_bytes())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// TensorBoardLogger
// ---------------------------------------------------------------------------

/// Writes TFEvent files readable by `tensorboard --logdir`.
pub struct TensorBoardLogger {
    writer: BufWriter<File>,
}

impl TensorBoardLogger {
    /// Create a new logger writing to `log_dir`.
    ///
    /// Creates `log_dir` if it doesn't exist. The event file is named
    /// `events.out.tfevents.{timestamp}.{hostname}`.
    pub fn new(log_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let log_dir = log_dir.as_ref();
        fs::create_dir_all(log_dir)?;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let hostname = hostname();
        let filename = format!("events.out.tfevents.{timestamp}.{hostname}");
        let path = log_dir.join(filename);

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write file_version event (required by TensorBoard)
        let mut event = Vec::new();
        encode_double(&mut event, 1, wall_time());
        encode_bytes(&mut event, 3, b"brain.Event:2");
        write_record(&mut writer, &event)?;

        Ok(Self { writer })
    }
}

fn wall_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "localhost".into())
}

impl Logger for TensorBoardLogger {
    fn log_scalar(&mut self, key: &str, value: f64, step: u64) {
        let event = build_scalar_event(wall_time(), step as i64, key, value as f32);
        let _ = write_record(&mut self.writer, &event);
    }

    fn log_scalars(&mut self, key: &str, values: &[(&str, f64)], step: u64) {
        for (sub_key, value) in values {
            let tag = format!("{key}/{sub_key}");
            let event = build_scalar_event(wall_time(), step as i64, &tag, *value as f32);
            let _ = write_record(&mut self.writer, &event);
        }
    }

    fn log_text(&mut self, key: &str, text: &str, step: u64) {
        let event = build_text_event(wall_time(), step as i64, key, text);
        let _ = write_record(&mut self.writer, &event);
    }

    fn log_histogram(&mut self, key: &str, values: &[f32], step: u64) {
        let event = build_histogram_event(wall_time(), step as i64, key, values);
        let _ = write_record(&mut self.writer, &event);
    }

    fn flush(&mut self) {
        let _ = self.writer.flush();
    }
}
