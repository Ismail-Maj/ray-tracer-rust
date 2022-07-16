use crate::EPSILON;

use std::cmp::PartialEq;
use std::ops::*;

#[derive(Debug, Copy, Clone, Default)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Add for Color {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

impl AddAssign for Color {
    fn add_assign(&mut self, other: Self) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
    }
}

impl Sub for Color {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            r: self.r - other.r,
            g: self.g - other.g,
            b: self.b - other.b,
        }
    }
}

impl SubAssign for Color {
    fn sub_assign(&mut self, other: Self) {
        self.r -= other.r;
        self.g -= other.g;
        self.b -= other.b;
    }
}

impl Neg for Color {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            r: -self.r,
            g: -self.g,
            b: -self.b,
        }
    }
}

impl Mul for Color {
    type Output = Self;
    fn mul(self, other: Color) -> Self::Output {
        Self {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }
}

impl MulAssign for Color {
    fn mul_assign(&mut self, other: Self) {
        self.r *= other.r;
        self.g *= other.g;
        self.b *= other.b;
    }
}

impl Mul<f32> for Color {
    type Output = Self;
    fn mul(self, other: f32) -> Self::Output {
        Self {
            r: self.r * other,
            g: self.g * other,
            b: self.b * other,
        }
    }
}

impl MulAssign<f32> for Color {
    fn mul_assign(&mut self, other: f32) {
        self.r *= other;
        self.g *= other;
        self.b *= other;
    }
}

impl Div for Color {
    type Output = Self;
    fn div(self, other: Color) -> Self::Output {
        Self {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }
}

impl DivAssign for Color {
    fn div_assign(&mut self, other: Self) {
        self.r /= other.r;
        self.g /= other.g;
        self.b /= other.b;
    }
}

impl Div<f32> for Color {
    type Output = Self;
    fn div(self, other: f32) -> Self::Output {
        Self {
            r: self.r / other,
            g: self.g / other,
            b: self.b / other,
        }
    }
}

impl DivAssign<f32> for Color {
    fn div_assign(&mut self, other: f32) {
        self.r /= other;
        self.g /= other;
        self.b /= other;
    }
}

impl PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        (self.r - other.r).abs() <= EPSILON
            && (self.g - other.g).abs() <= EPSILON
            && (self.b - other.b).abs() <= EPSILON
    }
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color() {
        let color = Color::new(1.0, 0.2, 0.4);
        assert_eq!(color.r, 1.0);
        assert_eq!(color.g, 0.2);
        assert_eq!(color.b, 0.4);
    }

    #[test]
    fn test_color_add() {
        let output = Color::new(1.0, 0.2, 0.4) + Color::new(0.9, 1.0, 0.1);

        let expected = Color::new(1.9, 1.2, 0.5);

        assert_eq!(output, expected);
    }

    #[test]
    fn test_color_sub() {
        let output = Color::new(1.0, 0.2, 0.4) - Color::new(0.9, 1.0, 0.1);

        let expected = Color::new(0.1, -0.8, 0.3);

        assert_eq!(output, expected);
    }

    #[test]
    fn test_color_mul_scalar() {
        let output = Color::new(1.0, 0.2, 0.4) * 2.0;

        let expected = Color::new(2.0, 0.4, 0.8);

        assert_eq!(output, expected);
    }

    #[test]
    fn test_color_mul() {
        let output = Color::new(1.0, 0.2, 0.4) * Color::new(0.9, 1.0, 0.1);

        let expected = Color::new(0.9, 0.2, 0.04);

        assert_eq!(output, expected);
    }
}
