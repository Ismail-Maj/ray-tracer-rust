use crate::EPSILON;

use core::ops::{Index, IndexMut};
use std::cmp::PartialEq;
use std::ops::*;

#[derive(Debug, Clone, Default)]
struct Matrix {
    width: u32,
    height: u32,
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

    pub fn submatrix(&self, row: usize, col: usize) -> Self {
        let mut buffer = Vec::with_capacity(((self.width - 1) * (self.height - 1)) as usize);
        self.buffer
            .iter()
            .enumerate()
            .for_each(|(i, v)| {
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
        let output = Matrix::from_buffer(
            4,
            4,
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 8., 7., 6., 5., 4., 3., 2.,
            ],
        ) * Matrix::from_buffer(
            4,
            4,
            vec![
                -2., 1., 2., 3., 3., 2., 1., -1., 4., 3., 6., 5., 1., 2., 7., 8.,
            ],
        );

        let expected = Matrix::from_buffer(4, 4, vec![20., 22., 50., 48., 44., 54., 114., 108., 40., 58.,
            110., 102., 16., 26., 46., 42.]);

        assert_eq!(output, expected);
    }
    #[test]
    fn test_matrix_identity() {
        let output = Matrix::new_identity(4);
        let expected = Matrix::from_buffer(4, 4, vec![1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_matrix_transpose() {
        let output = Matrix::from_buffer(4, 4, vec![0., 9., 3., 0., 9., 8., 0., 8., 1., 8., 5., 3., 0., 0., 5., 8.]).transpose();
        let expected = Matrix::from_buffer(4, 4, vec![0., 9., 1., 0., 9., 8., 8., 0., 3., 0., 5., 5., 0., 8., 3., 8.]);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_matrix_submatrix() {
        let output = Matrix::from_buffer(3, 3, vec![1., 5., 0., -3., 2., 7., 0., 6., -3.]).submatrix(0, 2);
        let expected = Matrix::from_buffer(2, 2, vec![-3., 2., 0., 6.]);
        assert_eq!(output, expected);

        let output = Matrix::from_buffer(4, 4, vec![-6., 1., 1., 6., -8., 5., 8., 6., -1., 0., 8., 2., -7., 1., -1., 1.]).submatrix(2, 1);
        let expected = Matrix::from_buffer(3, 3, vec![-6., 1., 6., -8., 8., 6., -7., -1., 1.]);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_matrix_minor() {
        let output = Matrix::from_buffer(3, 3, vec![3., 5., 0., 2., -1., -7., 6., -1., 5.]).minor(1, 0);
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

        let matrix = Matrix::from_buffer(4, 4, vec![-2., -8., 3., 5., -3., 1., 7., 3., 1., 2., -9., 6., -6., 7., 7., -9.]);
        assert_eq!(matrix.determinant(), -4071.);
    }
}
