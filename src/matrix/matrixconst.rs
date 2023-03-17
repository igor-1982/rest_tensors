use crate::matrix::{MatrixFull, BasicMatrix, MatFormat};
use crate::index::*; 
use crate::tensor_basic_operation::*;
use crate::matrix::matrixfullslice::*;
use crate::matrix::matrixupper::*;
use crate::matrix::submatrixfull::*;

pub struct DMatrix3x3 {
    size: [usize;2],
    indicing: [usize;2],
    data: [f64;9]
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