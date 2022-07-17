use crate::{color::Color, point::Point};

use core::ops::{Index, IndexMut};
use image::{ImageBuffer, ImageResult, Rgb};

pub struct Canvas {
    pub width: u32,
    pub height: u32,
    pub buffer: Vec<Color>,
}

impl Index<(usize, usize)> for Canvas {
    type Output = Color;

    fn index(&self, index: (usize, usize)) -> &Color {
        &self.buffer[index.1 * self.width as usize + index.0]
    }
}

impl IndexMut<(usize, usize)> for Canvas {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Color {
        &mut self.buffer[index.1 * self.width as usize + index.0]
    }
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            buffer: vec![Default::default(); (width * height) as usize],
        }
    }

    pub fn draw_2d(&mut self, point: Point, color: Color) -> Result<(), &str> {
        if point.x < 0.0
            || point.x >= self.width as f32
            || point.y < 0.0
            || point.y >= self.height as f32
        {
            Err("point out of bound")
        } else {
            self[(point.x as usize, point.y as usize)] = color;
            Ok(())
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

        let converted: ImageBuffer<Rgb<u8>, _> =
            ImageBuffer::from_raw(self.width, self.height, buffer_u8).unwrap();

        converted.save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point::Point, vector::Vector};

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

    #[test]
    fn test_canvas_projectile() {
        let mut point: Point = Point::new(0., 1., 0.);
        let mut velocity: Vector = Vector::new(1., 1.8, 0.).normalize() * 11.25;
        let gravity: Vector = Vector::new(0., -0.1, 0.);
        let wind: Vector = Vector::new(-0.01, 0., 0.);
        let mut canvas = Canvas::new(900, 550);
        while point.y > 0. {
            canvas.draw_2d(point, Color::RED).ok();
            point += velocity;
            velocity += gravity + wind;
        }
        canvas.save("projectile.png").unwrap();
    }
}
