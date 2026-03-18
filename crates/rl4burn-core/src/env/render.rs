//! Rendering types and trait for environment visualization.

/// An RGB frame (row-major, 3 bytes per pixel).
pub struct RgbFrame {
    pub width: u16,
    pub height: u16,
    /// Row-major RGB pixels, length = width * height * 3.
    pub data: Vec<u8>,
}

/// An environment that can render its current state to an RGB frame.
///
/// Implement this trait to enable GIF recording via `write_gif`.
pub trait Renderable {
    /// Render the current state to an RGB pixel buffer.
    fn render(&self) -> RgbFrame;
}
