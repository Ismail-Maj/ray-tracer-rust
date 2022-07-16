use crate::EPSILON;

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Copy, Clone, Default)]
struct Vector3 {
    x: f32,
    y: f32,
    z: f32,
}

impl PartialEq for Vector3 {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() <= EPSILON
            && (self.y - other.y).abs() <= EPSILON
            && (self.z - other.z).abs() <= EPSILON
    }
}

impl Add for Vector3 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vector3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Neg for Vector3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul<f32> for Vector3 {
    type Output = Self;
    fn mul(self, other: f32) -> Self::Output {
        Self {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl Div<f32> for Vector3 {
    type Output = Self;
    fn div(self, other: f32) -> Self::Output {
        Self {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl Vector3 {
    fn norm(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn normalize(self) -> Self {
        let norm = self.norm();
        Self {
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector3() {
        let output = Vector3 {
            x: 1.1,
            y: 2.2,
            z: -3.3,
        };

        assert_eq!(output.x, 1.1);
        assert_eq!(output.y, 2.2);
        assert_eq!(output.z, -3.3);
    }
    #[test]
    fn test_vector3_eq() {
        let output = Vector3 {
            x: 0.0,
            y: 2.2,
            z: -3.3,
        };

        assert_eq!(output, output.clone());
        assert_eq!(
            output,
            Vector3 {
                x: -0.0,
                y: 2.2,
                z: -3.3,
            }
        );
        assert_ne!(
            output,
            Vector3 {
                x: 0.0,
                y: 2.2,
                z: 0.0,
            }
        );
    }

    #[test]
    fn test_vector3_add() {
        let output = Vector3 {
            x: 1.1,
            y: 2.2,
            z: -3.3,
        }
        .add(Vector3 {
            x: 2.2,
            y: -1.1,
            z: 3.3,
        });

        let expected = Vector3 {
            x: 3.3,
            y: 1.1,
            z: 0.0,
        };

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector3_sub() {
        let output = Vector3 {
            x: 1.1,
            y: 2.2,
            z: -3.3,
        }
        .sub(Vector3 {
            x: 2.2,
            y: -1.1,
            z: 3.3,
        });

        let expected = Vector3 {
            x: -1.1,
            y: 3.3,
            z: -6.6,
        };

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector3_neg() {
        let output = -Vector3 {
            x: 1.1,
            y: 2.2,
            z: -3.3,
        };

        let expected = Vector3 {
            x: -1.1,
            y: -2.2,
            z: 3.3,
        };

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector3_mul_scalar() {
        let output = Vector3 {
            x: 1.1,
            y: 2.2,
            z: -3.3,
        } * 2.0;

        let expected = Vector3 {
            x: 2.2,
            y: 4.4,
            z: -6.6,
        };

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector3_div_scalar() {
        let output = Vector3 {
            x: 1.1,
            y: 2.2,
            z: -3.3,
        } / 2.0;

        let expected = Vector3 {
            x: 0.55,
            y: 1.1,
            z: -1.65,
        };

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector3_norm() {
        let output = Vector3 {
            x: 0.0,
            y: 3.0,
            z: 4.0,
        }
        .norm();

        let expected = 5.0;

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector3_normalize() {
        let output = Vector3 {
            x: 0.0,
            y: 3.0,
            z: 4.0,
        }
        .normalize();

        let expected = Vector3 {
            x: 0.0 / 5.0,
            y: 3.0 / 5.0,
            z: 4.0 / 5.0,
        };

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector3_dot() {
        let output = Vector3 {
            x: 2.0,
            y: 5.0,
            z: -4.0,
        }
        .dot(Vector3 {
            x: 1.1,
            y: 2.2,
            z: 3.3,
        });

        let expected = 0.0;

        assert_eq!(output, expected);
    }

    #[test]
    fn test_vector3_cross() {
        let output = Vector3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
        .cross(Vector3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        });

        let expected = Vector3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };

        assert_eq!(output, expected);
    }
}
