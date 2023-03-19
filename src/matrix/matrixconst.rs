use crate::matrix::{MatrixFull, BasicMatrix, MatFormat};
use crate::index::*; 
use crate::tensor_basic_operation::*;
use crate::matrix::matrixfullslice::*;
use crate::matrix::matrixupper::*;
use crate::matrix::submatrixfull::*;

pub struct DMatrix3x3 {
    pub size: [usize;2],
    pub indicing: [usize;2],
    pub data: [f64;9]
}

pub struct DMatrix5x6 {
    size: [usize;2],
    indicing: [usize;2],
    data: [f64;30]
}

pub struct DMatrix7x10 {
    size: [usize;2],
    indicing: [usize;2],
    data: [f64;70]
}

impl<'a> BasicMatrix<'a, f64> for DMatrix3x3 {
    fn size(&self) -> &[usize] {
        &self.size
    }

    fn indicing(&self) -> &[usize] {
        &self.indicing
    }

    fn data_ref(&self) -> Option<&[f64]> {
        Some(self.data.as_ref())
    }

    fn data_ref_mut(&mut self) -> Option<&mut [f64]> {
        Some(&mut self.data[..])
    }

    fn is_matr(&self) -> bool {
        self.size().len() == 2 && self.indicing().len() == 2
    }

    fn is_contiguous(&self) -> bool {true}
}