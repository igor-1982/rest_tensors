//#![warn(missing_docs)]
use std::{fmt::Display, collections::binary_heap::Iter, iter::{Filter,Flatten, Map}, convert, slice::{ChunksExact,ChunksExactMut, self}, mem::ManuallyDrop, marker, cell::RefCell, ops::{IndexMut, RangeFull, MulAssign, DivAssign, Div, DerefMut, Deref}, thread::panicking};
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, Index, Range};
use libc::{CLOSE_RANGE_CLOEXEC, SYS_userfaultfd};
use typenum::{U2, Pow};
use rayon::{prelude::*, collections::btree_map::IterMut, iter::Enumerate};
use std::vec::IntoIter;

use crate::matrix::{MatrixFull, BasicMatrix, MatFormat, BasicMatrixOpt, MathMatrix, ParMathMatrix};
use crate::index::*; 
use crate::tensor_basic_operation::*;
use crate::matrix::matrixfullslice::*;
use crate::matrix::matrixupper::*;
//{Indexing,Tensors4D};

pub enum SubMatrixFull<'a,T> {
    Contiguous(MatrixFullSlice<'a, T>),
    Detached(MatrixFull<&'a T>),
}
pub enum SubMatrixFullMut<'a,T> {
    Contiguous(MatrixFullSliceMut<'a, T>),
    Detached(MatrixFull<&'a mut T>),
}

impl <'a, T: Copy + Clone> BasicMatrix<'a, T> for SubMatrixFull<'a, T> {
    #[inline]
    /// `matr_a.size()' return &matr_a.size;
    fn size(&self) -> &[usize] {
        match &self {
            Self::Contiguous(matr) => {matr.size},
            Self::Detached(matr) => {&matr.size}
        }
    }
    #[inline]
    /// `matr_a.indicing()' return &matr_a.indicing;
    fn indicing(&self) -> &[usize] {
        match &self {
            Self::Contiguous(matr) => {matr.indicing},
            Self::Detached(matr) => {&matr.indicing}
        }
    }

    fn is_contiguous(&self) -> bool {
        match &self {
            Self::Contiguous(_) => {true},
            Self::Detached(_) => {false},
        }
    }

    fn data_ref(&self) -> Option<&[T]> {
        match &self {
            Self::Contiguous(matr) => {Some(matr.data)},
            Self::Detached(_) => {None},
        }
    }

    fn data_ref_mut(&mut self) -> Option<&mut [T]> {None}

}

impl<'a, T> BasicMatrixOpt<'a, T> for SubMatrixFull<'a, T> where T: Copy + Clone {}

impl<'a, T> MathMatrix<'a, T> for SubMatrixFull<'a, T> where T: Copy + Clone {}

impl<'a, T> ParMathMatrix<'a, T> for SubMatrixFull<'a, T> where T: Copy + Clone + Send + Sync {}


impl <'a, T> BasicMatrix<'a, T> for SubMatrixFullMut<'a, T> {
    #[inline]
    /// `matr_a.size()' return &matr_a.size;
    fn size(&self) -> &[usize] {
        match &self {
            Self::Contiguous(matr) => {matr.size},
            Self::Detached(matr) => {&matr.size}
        }
    }
    #[inline]
    /// `matr_a.indicing()' return &matr_a.indicing;
    fn indicing(&self) -> &[usize] {
        match &self {
            Self::Contiguous(matr) => {matr.indicing},
            Self::Detached(matr) => {&matr.indicing}
        }
    }

    fn data_ref(&self) -> Option<&[T]> {
        match &self {
            Self::Contiguous(matr) => {Some(matr.data)},
            Self::Detached(_) => {None},
        }
    }
    fn data_ref_mut(&mut self) -> Option<&mut [T]> {
        match self {
            Self::Contiguous(matr) => {Some(matr.data)},
            Self::Detached(_) => {None},
        }
    }
    fn is_contiguous(&self) -> bool {
        match &self {
            Self::Contiguous(_) => {true},
            Self::Detached(_) => {false},
        }
    }
}

impl<'a, T> BasicMatrixOpt<'a, T> for SubMatrixFullMut<'a, T> where T: Copy + Clone {}

impl<'a, T> MathMatrix<'a, T> for SubMatrixFullMut<'a, T> where T: Copy + Clone {}

impl<'a, T> ParMathMatrix<'a, T> for SubMatrixFullMut<'a, T> where T: Copy + Clone + Send + Sync {}

impl <'a, T: Copy + Clone> SubMatrixFull<'a, T> {
    pub fn data(&self) -> Vec<T> {
        match &self {
            SubMatrixFull::Contiguous(matr) => {matr.data.to_vec()},
            SubMatrixFull::Detached(matr) => {
                matr.data.iter().map(|x| *x.clone()).collect::<Vec<T>>()
            },
        }
    }
    pub fn c2d(self) -> SubMatrixFull<'a, T> {
        if let SubMatrixFull::Contiguous(matr) = self {
            let size = matr.size();
            let size = [size[0],size[1]];
            let indc = matr.indicing();
            let indicing = [indc[0],indc[1]];
            SubMatrixFull::Detached(MatrixFull {
                size,
                indicing,
                data: matr.data.iter().collect::<Vec<&T>>()
            })
        } else {
            self
        }
    }
}
impl <'a, T: Copy + Clone> SubMatrixFullMut<'a, T> {
    pub fn data(&self) -> Vec<T> {
        match &self {
            Self::Contiguous(matr) => {matr.data.to_vec()},
            Self::Detached(matr) => {
                matr.data.iter().map(|x| **x.clone()).collect::<Vec<T>>()
            },
        }
    }
    pub fn c2d(self) -> SubMatrixFullMut<'a, T> {
        if let SubMatrixFullMut::Contiguous(matr) = self {
            let size = matr.size();
            let size = [size[0],size[1]];
            let indc = matr.indicing();
            let indicing = [indc[0],indc[1]];
            SubMatrixFullMut::Detached(MatrixFull {
                size,
                indicing,
                data: matr.data.iter_mut().collect::<Vec<&mut T>>()
            })
        } else {
            self
        }
    }
}