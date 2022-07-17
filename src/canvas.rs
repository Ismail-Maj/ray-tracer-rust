use crate::color::Color;

use image::{ImageBuffer, ImageResult, Rgb};

pub struct Canvas {
    pub width: u32,
    pub height: u32,
    pub buffer: Vec<Color>,
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            buffer: vec![Default::default(); (width * height) as usize],
        }
    }

    pub fn save(self, path: &str) -> ImageResult<()> {
        let buffer_u8 = self
            .buffer
            .into_iter()
            .flatten()
            .into_iter()
            .map(|pixel| (pixel.min(1.0).max(0.0) * 255.0) as u8)
            .collect::<Vec<u8>>();

        dbg!(buffer_u8.len());

        let converted: ImageBuffer<Rgb<u8>, _> =
            ImageBuffer::from_raw(self.width, self.height, buffer_u8).unwrap();

        converted.save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canvas() {
        let output = Canvas::new(10, 20);

        output
            .buffer
            .iter()
            .for_each(|&pixel| assert_eq!(pixel, Color::new(0., 0., 0.)));

        assert_eq!(output.buffer.len(), 200);
    }

    #[test]
    fn test_canvas_png() {
        assert!(Canvas::new(10, 20).save("black.png").is_ok());
    }
}
