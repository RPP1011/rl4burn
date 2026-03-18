//! Shared drawing primitives for environment rendering.

/// Fill a rectangle with the given color (coordinates are clamped to canvas bounds).
pub(crate) fn draw_rect(
    pixels: &mut [u8],
    canvas_w: u16,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 3],
) {
    let w = canvas_w as i32;
    let h = (pixels.len() / 3 / w as usize) as i32;
    for y in y0.max(0)..y1.min(h) {
        for x in x0.max(0)..x1.min(w) {
            let idx = (y * w + x) as usize * 3;
            pixels[idx..idx + 3].copy_from_slice(&color);
        }
    }
}

/// Fill a circle at (cx, cy) with the given radius and color.
pub(crate) fn draw_circle(
    pixels: &mut [u8],
    canvas_w: u16,
    canvas_h: u16,
    cx: i32,
    cy: i32,
    r: i32,
    color: [u8; 3],
) {
    let w = canvas_w as i32;
    let h = canvas_h as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    let idx = (py * w + px) as usize * 3;
                    pixels[idx..idx + 3].copy_from_slice(&color);
                }
            }
        }
    }
}

/// Draw a thick line using Bresenham's algorithm with square brush.
pub(crate) fn draw_thick_line(
    pixels: &mut [u8],
    canvas_w: u16,
    canvas_h: u16,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    thickness: i32,
    color: [u8; 3],
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut cx = x0;
    let mut cy = y0;
    let w = canvas_w as i32;
    let h = canvas_h as i32;

    loop {
        for oy in -(thickness / 2)..=(thickness / 2) {
            for ox in -(thickness / 2)..=(thickness / 2) {
                let px = cx + ox;
                let py = cy + oy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    let idx = (py * w + px) as usize * 3;
                    pixels[idx..idx + 3].copy_from_slice(&color);
                }
            }
        }
        if cx == x1 && cy == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            cx += sx;
        }
        if e2 <= dx {
            err += dx;
            cy += sy;
        }
    }
}

/// Set a single pixel (bounds-checked).
pub(crate) fn set_pixel(pixels: &mut [u8], canvas_w: u16, x: i32, y: i32, color: [u8; 3]) {
    let w = canvas_w as i32;
    let h = pixels.len() as i32 / 3 / w;
    if x >= 0 && x < w && y >= 0 && y < h {
        let idx = (y * w + x) as usize * 3;
        pixels[idx..idx + 3].copy_from_slice(&color);
    }
}

/// Fill a convex or simple polygon using scanline rasterization.
///
/// `vertices` is a list of (x, y) integer coordinates defining the polygon.
pub(crate) fn draw_filled_polygon(
    pixels: &mut [u8],
    canvas_w: u16,
    canvas_h: u16,
    vertices: &[(i32, i32)],
    color: [u8; 3],
) {
    if vertices.len() < 3 {
        return;
    }
    let w = canvas_w as i32;
    let h = canvas_h as i32;

    let min_y = vertices.iter().map(|v| v.1).min().unwrap().max(0);
    let max_y = vertices.iter().map(|v| v.1).max().unwrap().min(h - 1);

    for y in min_y..=max_y {
        // Collect x-intersections with all edges
        let mut intersections = Vec::new();
        let n = vertices.len();
        for i in 0..n {
            let (x0, y0) = vertices[i];
            let (x1, y1) = vertices[(i + 1) % n];
            if (y0 <= y && y1 > y) || (y1 <= y && y0 > y) {
                let t = (y - y0) as f32 / (y1 - y0) as f32;
                intersections.push((x0 as f32 + t * (x1 - x0) as f32) as i32);
            }
        }
        intersections.sort_unstable();

        for pair in intersections.chunks(2) {
            if pair.len() == 2 {
                let xa = pair[0].max(0);
                let xb = pair[1].min(w - 1);
                for x in xa..=xb {
                    let idx = (y * w + x) as usize * 3;
                    pixels[idx..idx + 3].copy_from_slice(&color);
                }
            }
        }
    }
}
