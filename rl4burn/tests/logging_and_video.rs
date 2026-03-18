//! Tests for logging infrastructure and video recording.
//!
//! Run: `cargo test --test logging_and_video --features "ndarray,tensorboard,json-log,video"`

use rl4burn::log::{CompositeLogger, Logger, NoopLogger, PrintLogger};
use rl4burn::{DqnStats, Loggable, PpoStats};

// ---------------------------------------------------------------------------
// Tier 1: Core logging
// ---------------------------------------------------------------------------

#[test]
fn print_logger_basic() {
    let mut logger = PrintLogger::new(0);
    logger.log_scalar("train/loss", 0.42, 100);
    logger.log_scalars("train", &[("lr", 0.001), ("eps", 0.1)], 200);
    logger.log_text("info", "hello world", 300);
    logger.log_histogram("weights", &[0.1, 0.2, 0.3, 0.4], 400);
    logger.flush();
}

#[test]
fn print_logger_throttle() {
    let mut logger = PrintLogger::new(100);
    // First call always prints
    logger.log_scalar("x", 1.0, 0);
    // These should be throttled (step < 0 + 100)
    logger.log_scalar("x", 2.0, 50);
    logger.log_scalar("x", 3.0, 99);
    // This should print (step >= 0 + 100)
    logger.log_scalar("x", 4.0, 100);
}

#[test]
fn noop_logger() {
    let mut logger = NoopLogger;
    logger.log_scalar("anything", 999.0, 0);
    logger.flush();
}

#[test]
fn composite_logger() {
    let mut logger = CompositeLogger::new(vec![
        Box::new(PrintLogger::new(0)),
        Box::new(NoopLogger),
    ]);
    logger.log_scalar("train/loss", 0.5, 1);
    logger.log_scalars("train", &[("a", 1.0), ("b", 2.0)], 2);
    logger.flush();
}

// ---------------------------------------------------------------------------
// Loggable trait on stats structs
// ---------------------------------------------------------------------------

#[test]
fn ppo_stats_loggable() {
    let stats = PpoStats {
        policy_loss: 0.123,
        value_loss: 4.56,
        entropy: 0.69,
        approx_kl: 0.01,
    };
    let mut logger = PrintLogger::new(0);
    stats.log(&mut logger, 1000);
}

#[test]
fn dqn_stats_loggable() {
    let stats = DqnStats {
        loss: 0.42,
        mean_q: 12.5,
        epsilon: 0.1,
    };
    let mut logger = PrintLogger::new(0);
    stats.log(&mut logger, 5000);
}

// ---------------------------------------------------------------------------
// Tier 2: TensorBoard logger
// ---------------------------------------------------------------------------

#[cfg(feature = "tensorboard")]
mod tensorboard_tests {
    use rl4burn::log::{Logger, TensorBoardLogger};
    use rl4burn::{Loggable, PpoStats};

    #[test]
    fn tensorboard_writes_event_file() {
        let dir = std::env::temp_dir().join("rl4burn_test_tb");
        let _ = std::fs::remove_dir_all(&dir);

        let mut logger = TensorBoardLogger::new(&dir).expect("failed to create TensorBoardLogger");
        logger.log_scalar("train/loss", 0.5, 1);
        logger.log_scalar("train/loss", 0.3, 2);
        logger.log_scalars("train", &[("lr", 0.001), ("eps", 0.1)], 3);
        logger.log_text("info", "test run started", 0);
        logger.log_histogram("weights", &[0.1, -0.2, 0.3, 0.0, 0.5, -0.1], 1);

        let stats = PpoStats {
            policy_loss: 0.1,
            value_loss: 2.0,
            entropy: 0.6,
            approx_kl: 0.005,
        };
        stats.log(&mut logger, 100);
        logger.flush();

        // Verify the event file was created and has content
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1, "should have exactly one event file");

        let path = entries[0].path();
        let filename = path.file_name().unwrap().to_str().unwrap();
        assert!(
            filename.starts_with("events.out.tfevents."),
            "unexpected filename: {filename}"
        );

        let size = std::fs::metadata(&path).unwrap().len();
        assert!(size > 100, "event file too small: {size} bytes");

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ---------------------------------------------------------------------------
// Tier 3: JSON logger
// ---------------------------------------------------------------------------

#[cfg(feature = "json-log")]
mod json_tests {
    use rl4burn::log::{JsonLogger, Logger};
    use rl4burn::{DqnStats, Loggable};

    #[test]
    fn json_logger_writes_valid_jsonl() {
        let path = std::env::temp_dir().join("rl4burn_test_json.jsonl");

        {
            let mut logger = JsonLogger::from_path(&path).expect("failed to create JsonLogger");
            logger.log_scalar("train/loss", 0.42, 100);
            logger.log_scalars("train", &[("lr", 0.001)], 200);
            logger.log_text("info", "hello \"world\"\nnewline", 300);
            logger.log_histogram("weights", &[0.1, 0.2], 400);

            let stats = DqnStats {
                loss: 0.5,
                mean_q: 10.0,
                epsilon: 0.05,
            };
            stats.log(&mut logger, 500);
            logger.flush();
        }

        let output = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = output.lines().collect();

        // 4 direct calls + 3 from DqnStats.log = 7 lines
        assert_eq!(lines.len(), 7, "expected 7 JSONL lines, got {}:\n{output}", lines.len());

        // Each line should be valid JSON with expected fields
        for line in &lines {
            assert!(line.starts_with('{'), "line should be JSON object: {line}");
            assert!(line.contains("\"step\""), "line should contain step: {line}");
            assert!(line.contains("\"wall_time\""), "line should contain wall_time: {line}");
        }

        // First line should be a scalar
        assert!(lines[0].contains("\"type\":\"scalar\""));
        assert!(lines[0].contains("\"train/loss\""));

        let _ = std::fs::remove_file(&path);
    }
}

// ---------------------------------------------------------------------------
// Video: CartPole rendering + GIF
// ---------------------------------------------------------------------------

#[test]
fn cartpole_render_produces_rgb_frame() {
    use rand::SeedableRng;
    use rl4burn::env::render::{Renderable, RgbFrame};
    use rl4burn::envs::CartPole;
    use rl4burn::Env;

    let mut env = CartPole::new(rand::rngs::SmallRng::seed_from_u64(42));
    env.reset();

    let frame: RgbFrame = env.render();
    assert_eq!(frame.width, 600);
    assert_eq!(frame.height, 400);
    assert_eq!(frame.data.len(), 600 * 400 * 3);

    // After a few steps the frame should still be valid
    for _ in 0..10 {
        env.step(1);
    }
    let frame2 = env.render();
    assert_eq!(frame2.data.len(), 600 * 400 * 3);
}

#[cfg(feature = "video")]
mod video_tests {
    use rand::SeedableRng;
    use rl4burn::envs::CartPole;
    use rl4burn::write_gif;
    use rl4burn::Env;

    #[test]
    fn record_cartpole_gif() {
        let mut env = CartPole::new(rand::rngs::SmallRng::seed_from_u64(42));
        env.reset();

        let mut frames = Vec::new();
        frames.push(env.render());

        // Run an episode with alternating actions (won't balance, but shows movement)
        for i in 0..80 {
            let action = if i % 4 < 2 { 1 } else { 0 };
            let step = env.step(action);
            frames.push(env.render());
            if step.done() {
                break;
            }
        }

        let path = std::env::temp_dir().join("rl4burn_test_cartpole.gif");
        write_gif(&path, &frames, 4).expect("failed to write GIF");

        let size = std::fs::metadata(&path).unwrap().len();
        assert!(size > 1000, "GIF too small: {size} bytes");
        eprintln!("CartPole GIF written to: {}", path.display());
        eprintln!("  frames: {}, size: {} bytes", frames.len(), size);
    }
}
