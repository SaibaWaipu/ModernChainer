use std::fmt;

use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Shape(dims.into())
    }

    pub fn scalar() -> Self {
        Shape(Vec::new())
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    pub fn contiguous_strides(&self) -> Strides {
        Strides::contiguous_for(self)
    }

    /// NumPy-style broadcasting.  Dimensions are aligned from the right.
    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape> {
        let rank = self.rank().max(other.rank());
        let mut out = vec![1; rank];
        for d in (0..rank).rev() {
            let a = self.dim_from_right(rank - 1 - d);
            let b = other.dim_from_right(rank - 1 - d);
            out[d] = match (a, b) {
                (1, x) | (x, 1) => x,
                (x, y) if x == y => x,
                _ => {
                    return Err(Error::Broadcast {
                        left: self.clone(),
                        right: other.clone(),
                    });
                }
            };
        }
        Ok(Shape(out))
    }

    fn dim_from_right(&self, from_right: usize) -> usize {
        let rank = self.rank();
        if from_right >= rank {
            1
        } else {
            self.0[rank - 1 - from_right]
        }
    }

    pub fn with_batch(&self, batch: usize) -> Shape {
        let mut dims = Vec::with_capacity(self.rank() + 1);
        dims.push(batch);
        dims.extend_from_slice(&self.0);
        Shape(dims)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("()");
        }
        let parts: Vec<String> = self.0.iter().map(usize::to_string).collect();
        f.write_str(&parts.join("x"))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Shape(value)
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Shape(value.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Shape(value.to_vec())
    }
}

impl From<Shape> for Vec<usize> {
    fn from(value: Shape) -> Self {
        value.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Strides(Vec<isize>);

impl Strides {
    pub fn new(strides: impl Into<Vec<isize>>) -> Self {
        Strides(strides.into())
    }

    pub fn as_slice(&self) -> &[isize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn contiguous_for(shape: &Shape) -> Self {
        let mut strides = vec![0isize; shape.rank()];
        let mut acc = 1isize;
        for d in (0..shape.rank()).rev() {
            strides[d] = acc;
            acc *= shape.dims()[d] as isize;
        }
        Strides(strides)
    }

    pub fn is_contiguous_for(&self, shape: &Shape) -> bool {
        *self == Strides::contiguous_for(shape)
    }
}

impl From<Vec<isize>> for Strides {
    fn from(value: Vec<isize>) -> Self {
        Strides(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_and_strides() {
        let s = Shape::new([2, 3, 4]);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.contiguous_strides(), Strides::new(vec![12, 4, 1]));
        assert_eq!(Shape::scalar().numel(), 1);
    }

    #[test]
    fn broadcasting() {
        assert_eq!(
            Shape::new([2, 3]).broadcast_with(&Shape::new([3])).unwrap(),
            Shape::new([2, 3])
        );
        assert_eq!(
            Shape::new([2, 1, 4])
                .broadcast_with(&Shape::new([3, 1]))
                .unwrap(),
            Shape::new([2, 3, 4])
        );
        assert!(
            Shape::new([2, 3])
                .broadcast_with(&Shape::new([2, 4]))
                .is_err()
        );
    }
}
