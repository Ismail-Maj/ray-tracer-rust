use crate::EPSILON;
use crate::{point::Point, vector::Vector};

use core::ops::{Index, IndexMut};
use std::cmp::PartialEq;
use std::iter;
use std::ops::*;

#[derive(Debug, Clone, Default)]
pub struct Matrix {
    pub width: u32,
    pub height: u32,
    buffer: Vec<f32>,
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &f32 {
        &self.buffer[index.0 * self.width as usize + index.1]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f32 {
        &mut self.buffer[index.0 * self.width as usize + index.1]
    }
}

impl Mul for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Matrix {
        assert_eq!(self.width, other.height);
        let mut result = Matrix::new(other.width, self.height);
        for i in 0..result.height {
            for j in 0..result.width {
                let mut sum = 0.0;
                for k in 0..self.width {
                    sum += self[(i as usize, k as usize)] * other[(k as usize, j as usize)];
                }
                result[(i as usize, j as usize)] = sum;
            }
        }
        result
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        return self.width == other.width
            && self.height == other.height
            && self
                .buffer
                .iter()
                .zip(other.buffer.iter())
                .all(|(lhs, rhs)| (lhs - rhs).abs() < EPSILON);
    }
}

impl Matrix {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            buffer: vec![Default::default(); (width * height) as usize],
        }
    }

    pub fn from_buffer(width: u32, height: u32, buffer: Vec<f32>) -> Self {
        if (width * height) as usize != buffer.len() {
            panic!(
                "Buffer length must be {} != {}",
                width * height,
                buffer.len(),
            );
        }
        Self {
            width,
            height,
            buffer,
        }
    }

    pub fn from_point(point: Point) -> Self {
        Self::from_buffer(1, 4, vec![point.x, point.y, point.z, 1.0])
    }

    pub fn to_point(self) -> Point {
        Point::from_matrix(self)
    }

    pub fn from_vector(vector: Vector) -> Self {
        Self::from_buffer(1, 4, vec![vector.x, vector.y, vector.z, 0.0])
    }

    pub fn to_vector(self) -> Vector {
        Vector::from_matrix(self)
    }

    pub fn new_identity(size: u32) -> Self {
        let mut buffer = vec![];
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    buffer.push(1.0);
                } else {
                    buffer.push(0.0);
                }
            }
        }
        Self::from_buffer(size, size, buffer)
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.height, self.width);
        for i in 0..self.height {
            for j in 0..self.width {
                result[(j as usize, i as usize)] = self[(i as usize, j as usize)];
            }
        }
        result
    }

    pub fn set(mut self, x: usize, y: usize, value: f32) -> Self {
        self[(x, y)] = value;
        self
    }

    pub fn translation_matrix(x: f32, y: f32, z: f32) -> Self {
        Self::new_identity(4).set(0, 3, x).set(1, 3, y).set(2, 3, z)
    }

    pub fn translate(self, x: f32, y: f32, z: f32) -> Self {
        Self::translation_matrix(x, y, z) * self
    }

    pub fn scaling_matrix(x: f32, y: f32, z: f32) -> Self {
        Self::new_identity(4).set(0, 0, x).set(1, 1, y).set(2, 2, z)
    }

    pub fn scale(self, point: &Point) -> Self {
        Self::scaling_matrix(point.x, point.y, point.z) * self
    }

    pub fn shearing_matrix(xy: f32, xz: f32, yx: f32, yz: f32, zx: f32, zy: f32) -> Self {
        Self::new_identity(4)
            .set(0, 1, xy)
            .set(0, 2, xz)
            .set(1, 0, yx)
            .set(1, 2, yz)
            .set(2, 0, zx)
            .set(2, 1, zy)
    }

    //TODO: rotate

    pub fn submatrix(&self, row: usize, col: usize) -> Self {
        let mut buffer = Vec::with_capacity(((self.width - 1) * (self.height - 1)) as usize);
        self.buffer.iter().enumerate().for_each(|(i, v)| {
            if i % self.width as usize != col as usize && i / self.width as usize != row as usize {
                buffer.push(*v);
            }
        });
        Self::from_buffer(self.width - 1, self.height - 1, buffer)
    }

    pub fn minor(&self, row: usize, col: usize) -> f32 {
        self.submatrix(row, col).determinant()
    }

    pub fn cofactor(&self, row: usize, col: usize) -> f32 {
        let minor = self.minor(row, col);
        if (row + col) % 2 == 0 {
            minor
        } else {
            -minor
        }
    }

    pub fn determinant(&self) -> f32 {
        if self.width != self.height {
            panic!("Matrix must be square");
        }
        if self.width == 1 {
            return self[(0, 0)];
        }
        if self.width == 2 {
            return self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)];
        }
        let mut result = 0.0;
        for i in 0..self.width {
            let i = i as usize;
            result += self.cofactor(0, i) * self[(0, i)];
        }
        result
    }

    pub fn inverse(&self) -> Self {
        let det = self.determinant();
        if det == 0.0 {
            panic!("Matrix is not invertible");
        }

        let buffer = (0..self.width)
            .flat_map(|i| {
                iter::repeat(i)
                    .zip(0..self.height)
                    .map(|(i, j)| self.cofactor(j as usize, i as usize) / det)
            })
            .collect();

        Self::from_buffer(self.width, self.height, buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix() {
        let matrix = Matrix::new(10, 20);
        for i in 0..matrix.width {
            for j in 0..matrix.height {
                assert_eq!(matrix[(i as usize, j as usize)], 0.0);
            }
        }

        let matrix = Matrix::from_buffer(2, 2, vec![-3., 5., 1., -2.]);
        assert_eq!(matrix[(0, 0)], -3.);
        assert_eq!(matrix[(0, 1)], 5.);
        assert_eq!(matrix[(1, 0)], 1.);
        assert_eq!(matrix[(1, 1)], -2.);
    }

    #[test]
    fn test_matrix_eq() {
        let output = Matrix::new(10, 20);
        let eq = Matrix::from_buffer(10, 20, vec![0.0; 200]);
        let ne = Matrix::new(20, 10);

        assert_eq!(output, eq);
        assert_ne!(output, ne);
    }

    #[test]
    fn test_matrix_mul() {
        let output = Matrix::from_buffer(2, 2, vec![1., 2., 3., 4.])
            * Matrix::from_buffer(2, 2, vec![-2., 1., 3., -1.]);

        let expected = Matrix::from_buffer(2, 2, vec![4., -1., 6., -1.]);

        assert_eq!(output, expected);
    }
    #[test]
    fn test_matrix_identity() {
        let output = Matrix::new_identity(4);
        let expected = Matrix::from_buffer(
            4,
            4,
            vec![
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn test_matrix_transpose() {
        let output = Matrix::from_buffer(
            4,
            4,
            vec![
                0., 9., 3., 0., 9., 8., 0., 8., 1., 8., 5., 3., 0., 0., 5., 8.,
            ],
        )
        .transpose();
        let expected = Matrix::from_buffer(
            4,
            4,
            vec![
                0., 9., 1., 0., 9., 8., 8., 0., 3., 0., 5., 5., 0., 8., 3., 8.,
            ],
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn test_matrix_submatrix() {
        let output =
            Matrix::from_buffer(3, 3, vec![1., 5., 0., -3., 2., 7., 0., 6., -3.]).submatrix(0, 2);
        let expected = Matrix::from_buffer(2, 2, vec![-3., 2., 0., 6.]);
        assert_eq!(output, expected);

        let output = Matrix::from_buffer(
            4,
            4,
            vec![
                -6., 1., 1., 6., -8., 5., 8., 6., -1., 0., 8., 2., -7., 1., -1., 1.,
            ],
        )
        .submatrix(2, 1);
        let expected = Matrix::from_buffer(3, 3, vec![-6., 1., 6., -8., 8., 6., -7., -1., 1.]);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_matrix_minor() {
        let output =
            Matrix::from_buffer(3, 3, vec![3., 5., 0., 2., -1., -7., 6., -1., 5.]).minor(1, 0);
        let expected = 25.;

        assert_eq!(output, expected);
    }

    #[test]
    fn test_matrix_cofactor() {
        let matrix = Matrix::from_buffer(3, 3, vec![3., 5., 0., 2., -1., -7., 6., -1., 5.]);
        assert_eq!(matrix.cofactor(0, 0), -12.);
        assert_eq!(matrix.cofactor(1, 0), -25.);
    }

    #[test]
    fn test_matrix_determinant() {
        let matrix = Matrix::from_buffer(3, 3, vec![1., 2., 6., -5., 8., -4., 2., 6., 4.]);
        assert_eq!(matrix.determinant(), -196.);

        let matrix = Matrix::from_buffer(
            4,
            4,
            vec![
                -2., -8., 3., 5., -3., 1., 7., 3., 1., 2., -9., 6., -6., 7., 7., -9.,
            ],
        );
        assert_eq!(matrix.determinant(), -4071.);
    }

    #[test]
    fn test_matrix_inverse() {
        let matrix = Matrix::from_buffer(
            4,
            4,
            vec![
                -5., 2., 6., -8., 1., -5., 1., 8., 7., 7., -6., -7., 1., -3., 7., 4.,
            ],
        );
        let expected = Matrix::from_buffer(
            4,
            4,
            vec![
                0.21805, 0.45113, 0.24060, -0.04511, -0.80827, -1.45677, -0.44361, 0.52068,
                -0.07895, -0.22368, -0.05263, 0.19737, -0.52256, -0.81391, -0.30075, 0.30639,
            ],
        );
        assert_eq!(matrix.inverse(), expected);
    }

    #[test]
    fn test_matrix_inverse_identity() {
        let matrix = Matrix::from_buffer(
            4,
            4,
            vec![
                -5., 2., 6., -8., 1., -5., 1., 8., 7., 7., -6., -7., 1., -3., 7., 4.,
            ],
        );
        let inverse = matrix.inverse();
        let output = matrix * inverse;
        assert_eq!(output, Matrix::new_identity(4));
    }

    #[test]
    fn test_matrix_translate() {
        let output = Point::new(-3., 4., 5.)
            .to_matrix()
            .translate(5., -3., 2.)
            .to_point();
        let expected = Point::new(2., 1., 7.);
        assert_eq!(output, expected);

        let reverse_translation = Matrix::translation_matrix(5., -3., 2.).inverse();
        let output = (reverse_translation * output.to_matrix()).to_point();
        let expected = Point::new(-3., 4., 5.);
        assert_eq!(output, expected);

        let vector = Vector::new(-3., 4., 5.);

        assert_eq!(
            vector.to_matrix().translate(5., -3., 2.).to_vector(),
            vector
        );
    }
}
