use crate::{matrix::Matrix, EPSILON};

use std::cmp::PartialEq;
use std::ops::*;

#[derive(Debug, Copy, Clone, Default)]
pub struct Vector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn from_matrix(matrix: Matrix) -> Self {
        Self {
            x: matrix[(0, 0)],
            y: matrix[(0, 1)],
            z: matrix[(0, 2)],
        }
    }

    pub fn to_matrix(self) -> Matrix {
        Matrix::from_vector(self)
    }

    pub fn norm(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(self) -> Self {
        let norm = self.norm();
        Self {
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl Add for Vector {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl Sub for Vector {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl Neg for Vector {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul for Vector {
    type Output = Self;
    fn mul(self, other: Vector) -> Self::Output {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl MulAssign for Vector {
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
    }
}

impl Mul<f32> for Vector {
    type Output = Self;
    fn mul(self, other: f32) -> Self::Output {
        Self {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl MulAssign<f32> for Vector {
    fn mul_assign(&mut self, other: f32) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
    }
}

impl Div for Vector {
    type Output = Self;
    fn div(self, other: Vector) -> Self::Output {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl DivAssign for Vector {
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
        self.z /= other.z;
    }
}

impl Div<f32> for Vector {
    type Output = Self;
    fn div(self, other: f32) -> Self::Output {
        Self {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl DivAssign<f32> for Vector {
    fn div_assign(&mut self, other: f32) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() <= EPSILON
            && (self.y - other.y).abs() <= EPSILON
            && (self.z - other.z).abs() <= EPSILON
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector() {
        let output = Vector {
            x: 1.1,
            y: 2.2,
            z: -3.3,
        };

        assert_eq!(output.x, 1.1);
        assert_eq!(output.y, 2.2);
        assert_eq!(output.z, -3.3);
    }
    #[test]
    fn test_vector_eq() {
        let output = Vector {
            x: 0.0,
            y: 2.2,
            z: -3.3,
        };

        assert_eq!(output, output.clone());
        assert_eq!(
            output,
            Vector {
                x: -0.0,
                y: 2.2,
                z: -3.3,
            }
        );
        assert_ne!(
            output,
            Vector {
                x: 0.0,
                y: 2.2,
                z: 0.0,
            }
        );
    }

    #[test]
    fn test_vector_add() {
        let output = Vector {
            x: 1.1,
            y: 2.2,
            z: -3.3,
        }
        .add(Vector {
            x: 2.2,
            y: -1.1,
            z: 3.3,
        });

        let expected = Vector {
            x: 3.3,
            y: 1.1,
            z: 0.0,
        };

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector_sub() {
        let output = Vector {
            x: 1.1,
            y: 2.2,
            z: -3.3,
        }
        .sub(Vector {
            x: 2.2,
            y: -1.1,
            z: 3.3,
        });

        let expected = Vector {
            x: -1.1,
            y: 3.3,
            z: -6.6,
        };

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector_neg() {
        let output = -Vector {
            x: 1.1,
            y: 2.2,
            z: -3.3,
        };

        let expected = Vector {
            x: -1.1,
            y: -2.2,
            z: 3.3,
        };

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector_mul_scalar() {
        let output = Vector {
            x: 1.1,
            y: 2.2,
            z: -3.3,
        } * 2.0;

        let expected = Vector {
            x: 2.2,
            y: 4.4,
            z: -6.6,
        };

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector_div_scalar() {
        let output = Vector {
            x: 1.1,
            y: 2.2,
            z: -3.3,
        } / 2.0;

        let expected = Vector {
            x: 0.55,
            y: 1.1,
            z: -1.65,
        };

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector_norm() {
        let output = Vector {
            x: 0.0,
            y: 3.0,
            z: 4.0,
        }
        .norm();

        let expected = 5.0;

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector_normalize() {
        let output = Vector {
            x: 0.0,
            y: 3.0,
            z: 4.0,
        }
        .normalize();

        let expected = Vector {
            x: 0.0 / 5.0,
            y: 3.0 / 5.0,
            z: 4.0 / 5.0,
        };

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector_dot() {
        let output = Vector {
            x: 2.0,
            y: 5.0,
            z: -4.0,
        }
        .dot(Vector {
            x: 1.1,
            y: 2.2,
            z: 3.3,
        });

        let expected = 0.0;

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector_cross() {
        let output = Vector {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
        .cross(Vector {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        });

        let expected = Vector {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };

        assert_eq!(output, expected);
    }
}
