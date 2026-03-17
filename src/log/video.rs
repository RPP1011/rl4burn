//! GIF video writer for recording episodes.

use crate::env::render::RgbFrame;

use std::fs::File;
use std::path::Path;

/// Write a sequence of RGB frames to a GIF file.
///
/// `delay_cs` is the delay between frames in centiseconds (e.g. 2 = 20ms ≈ 50fps).
pub fn write_gif(path: impl AsRef<Path>, frames: &[RgbFrame], delay_cs: u16) -> std::io::Result<()> {
    if frames.is_empty() {
        return Ok(());
    }

    let width = frames[0].width;
    let height = frames[0].height;

    let file = File::create(path)?;
    let mut encoder = gif::Encoder::new(file, width, height, &[])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    encoder
        .set_repeat(gif::Repeat::Infinite)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    for frame in frames {
        let mut rgba: Vec<u8> = Vec::with_capacity(frame.data.len() / 3 * 4);
        for chunk in frame.data.chunks_exact(3) {
            rgba.push(chunk[0]);
            rgba.push(chunk[1]);
            rgba.push(chunk[2]);
            rgba.push(255);
        }

        let mut gif_frame = gif::Frame::from_rgba_speed(width, height, &mut rgba, 10);
        gif_frame.delay = delay_cs;
        encoder
            .write_frame(&gif_frame)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    }

    Ok(())
}
