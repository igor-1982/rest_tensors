//#![warn(missing_docs)]
use std::{cell::RefCell, collections::binary_heap::Iter, convert, fmt::Display, iter::{Filter,Flatten, Map, StepBy}, marker, mem::ManuallyDrop, ops::{Deref, DerefMut, Div, DivAssign, IndexMut, MulAssign, RangeFull}, slice::{self, ChunksExact, ChunksExactMut}, thread::panicking};
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, Index, Range};
use libc::{CLOSE_RANGE_CLOEXEC, SYS_userfaultfd};
use typenum::{U2, Pow};
use rayon::{prelude::*, collections::btree_map::IterMut, iter::Enumerate};
use std::vec::IntoIter;

use crate::{MatrixFull, BasicMatrix, MatFormat, BasicMatrixOpt, MathMatrix, ParMathMatrix};
use crate::index::*; 
use crate::tensor_basic_operation::*;
use crate::matrix::matrixfull::*;
//use crate::matrix::matrixupper::*;

use super::matrix_trait::*;


#[derive(Debug,PartialEq)]
pub struct MatrixFullSliceMut<'a,T> {
    pub size : &'a [usize],
    pub indicing: &'a [usize],
    pub data : &'a mut [T]
    //pub data : [&'a mut T]
}

impl <'a, T> BasicMatrix<'a, T> for MatrixFullSliceMut<'a,T> {
    #[inline]
    /// `matr_a.size()' return &matr_a.size;
    fn size(&self) -> &[usize] {
        &self.size
    }
    #[inline]
    /// `matr_a.indicing()' return &matr_a.indicing;
    fn indicing(&self) -> &[usize] {
        &self.indicing
    }

    fn data_ref(&self) -> Option<&[T]> {
        Some(&self.data)
    }
    fn data_ref_mut(&mut self) -> Option<&mut [T]> {
        Some(self.data)
    }
}

impl<'a, T> BasicMatrixOpt<'a, T> for MatrixFullSliceMut<'a, T> where T: Copy + Clone {}

impl<'a, T> MathMatrix<'a, T> for MatrixFullSliceMut<'a, T> where T: Copy + Clone {}

impl<'a, T> ParMathMatrix<'a, T> for MatrixFullSliceMut<'a, T> where T: Copy + Clone + Send + Sync {}

impl <'a, T: Copy + Clone + Display + Send + Sync> MatrixFullSliceMut<'a, T> {
    //pub fn from_slice(sl: &[T]) -> MatrixFullSlice<'a, T> {
    //    MatrixFullSliceMut {
    //        size
    //    }
    //}
    #[inline]
    pub fn get_column_mut(&mut self, j: usize) -> &mut [T] {
        let start = self.size[0]*j;
        let end = start + self.size[0];
        &mut self.data[start..end]
    }
    #[inline]
    pub fn iter_column_mut(&mut self, j: usize) -> std::slice::IterMut<T> {
        let start = self.size[0]*j;
        let end = start + self.size[0];
        self.data[start..end].iter_mut()
    }
    #[inline]
    pub fn iter_mut_j(&mut self, j: usize) -> std::slice::IterMut<T> {
        let start = self.size[0]*j;
        let end = start + self.size[0];
        self.data[start..end].iter_mut()
    }
    #[inline]
    pub fn par_iter_mut_j(&mut self, j: usize) -> rayon::slice::IterMut<T> {
        let start = self.indicing[1]*j;
        let end = start + self.indicing[1];
        self.data[start..end].par_iter_mut()
    }
    #[inline]
    pub fn iter_columns_full(&self) -> ChunksExact<T>{
        self.data.chunks_exact(self.size[0])
    }
    #[inline]
    pub fn iter_mut_columns(&mut self,range_column: Range<usize>) -> Option<ChunksExactMut<T>>{
        if let Some(n_chunk) = self.size.get(0) {
            Some(self.data[n_chunk*range_column.start..n_chunk*range_column.end].chunks_exact_mut(*n_chunk))
        }  else {
            None
        }
    }
    #[inline]
    pub fn iter_submatrix_mut_old(& mut self, x: Range<usize>, y: Range<usize>) -> Flatten<IntoIter<&mut [T]>> {
        let mut tmp_slices: Vec<&mut [T]> = Vec::with_capacity(y.len());
        let mut dd = self.data.split_at_mut(0).1;
        let len_slices_x = x.len();
        let len_y = self.indicing[1];
        y.fold((dd,0_usize),|(ee, offset), y| {
            let start = x.start + y*len_y;
            let gg = ee.split_at_mut(start-offset).1.split_at_mut(len_slices_x);
            tmp_slices.push(gg.0);
            (gg.1,start+len_slices_x)
        });
        tmp_slices.into_iter().flatten()
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<T> {
        self.data.iter()
    }
    #[inline]
    pub fn iter_submatrix(&self, x: Range<usize>, y: Range<usize>) ->  SubMatrixStepBy<slice::Iter<T>>{
        self.iter().submatrix_step_by(x, y, [self.size[0],self.size[1]])
    }
    #[inline]
    pub fn iter_matrixupper_submatrix(&self, x: Range<usize>, y: Range<usize>) ->  SubMatrixInUpperStepBy<slice::Iter<T>>{
        self.iter().submatrix_in_upper_step_by(x, y, [self.size[0],self.size[1]])
    }
    #[inline]
    pub fn iter_diagonal(&self) -> Option<StepBy<std::slice::Iter<T>>> {
        //let [x,y] = self.size;
        let x = self.size[0];
        let y = self.size[1];
        if x==0 || y==0 || x!=y {
            return None
        } else {
            return Some(self.iter().step_by(x+1))
        }
    }
    #[inline]
    pub fn iter_matrixupper(&self) -> Option<MatrixUpperStepBy<std::slice::Iter<T>>> {
        //let [x,y] = self.size;
        let x = self.size[0];
        let y = self.size[1];
        if x==0 || y==0 || x!=y {
            return None
        } else {
            return Some(self.iter().matrixupper_step_by([x,y]))
        }
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        self.data.iter_mut()
    }
    #[inline]
    pub fn iter_submatrix_mut(&mut self, x: Range<usize>, y: Range<usize>) ->  SubMatrixStepBy<slice::IterMut<T>>{
        let size = [self.size[0],self.size[1]];
        self.iter_mut().submatrix_step_by(x, y, size)
    }
    #[inline]
    pub fn iter_matrixupper_submatrix_mut(&mut self, x: Range<usize>, y: Range<usize>) ->  SubMatrixInUpperStepBy<slice::IterMut<T>>{
        let size = [self.size[0],self.size[1]];
        self.iter_mut().submatrix_in_upper_step_by(x, y, size)
    }
    pub fn iter_diagonal_mut(&mut self) -> Option<StepBy<std::slice::IterMut<T>>> {
        //let [x,y] = self.size;
        let x = self.size[0];
        let y = self.size[1];
        if x==0 || y==0 || x!=y {
            return None
        } else {
            return Some(self.iter_mut().step_by(x+1))
        }
    }
    #[inline]
    pub fn iter_matrixupper_mut(& mut self) -> Option<MatrixUpperStepBy<std::slice::IterMut<T>>> {
        //let [x,y] = self.size;
        let x = self.size[0];
        let y = self.size[1];
        if x==0 || y==0 || x!=y {
            return None
        } else {
            return Some(self.iter_mut().matrixupper_step_by([x,y]))
        }
    }
}



#[derive(Clone,Debug,PartialEq)]
pub struct MatrixFullSlice<'a,T> {
    pub size : &'a [usize],
    pub indicing: &'a [usize],
    pub data : &'a [T]
}

impl <'a, T> BasicMatrix<'a, T> for MatrixFullSlice<'a,T> {
    #[inline]
    /// `matr_a.size()' return &matr_a.size;
    fn size(&self) -> &[usize] {
        &self.size
    }
    #[inline]
    /// `matr_a.indicing()' return &matr_a.indicing;
    fn indicing(&self) -> &[usize] {
        &self.indicing
    }

    fn data_ref(&self) -> Option<&[T]> {
        Some(&self.data)
    }

    fn data_ref_mut(&mut self) -> Option<&mut [T]> {None}
}

impl<'a, T> BasicMatrixOpt<'a, T> for MatrixFullSlice<'a, T> where T: Copy + Clone {}

impl<'a, T> MathMatrix<'a, T> for MatrixFullSlice<'a, T> where T: Copy + Clone {}

impl<'a, T> ParMathMatrix<'a, T> for MatrixFullSlice<'a, T> where T: Copy + Clone + Send + Sync {}

impl <'a, T: Copy + Clone + Display + Send + Sync> MatrixFullSlice<'a, T> {
    #[inline]
    pub fn iter(&self) -> slice::Iter<T> {
        self.data.iter()
    }
    #[inline]
    pub fn iter_submatrix(&self, x: Range<usize>, y: Range<usize>) ->  SubMatrixStepBy<slice::Iter<T>>{
        self.iter().submatrix_step_by(x, y, [self.size[0],self.size[1]])
    }
    #[inline]
    pub fn iter_matrixupper_submatrix(&self, x: Range<usize>, y: Range<usize>) ->  SubMatrixInUpperStepBy<slice::Iter<T>>{
        self.iter().submatrix_in_upper_step_by(x, y, [self.size[0],self.size[1]])
    }
    #[inline]
    pub fn iter_diagonal(&self) -> Option<StepBy<std::slice::Iter<T>>> {
        //let [x,y] = self.size;
        let x = self.size[0];
        let y = self.size[1];
        if x==0 || y==0 || x!=y {
            return None
        } else {
            return Some(self.iter().step_by(x+1))
        }
    }
    #[inline]
    pub fn iter_matrixupper(&self) -> Option<MatrixUpperStepBy<std::slice::Iter<T>>> {
        //let [x,y] = self.size;
        let x = self.size[0];
        let y = self.size[1];
        if x==0 || y==0 || x!=y {
            return None
        } else {
            return Some(self.iter().matrixupper_step_by([x,y]))
        }
    }
    #[inline]
    pub fn iter_matrixupper_shift(&self, shift: usize) -> Option<MatrixUpperStepBy<std::slice::Iter<T>>> {
        //let [x,y] = self.size;
        let x = self.size[0];
        let y = self.size[1];
        if x==0 || y==0 || x!=y {
            return None
        } else {
            return Some(self.iter().matrixupper_step_by_shift([x,y], shift))
        }
    }
    #[inline]
    pub fn iter_j(&self, j: usize) -> std::slice::Iter<T> {
        let start = self.size[0]*j;
        let end = start + self.size[0];
        self.data[start..end].iter()
    }
    #[inline]
    pub fn iter_columns(&self, range_column: Range<usize>) -> Option<ChunksExact<T>>{
        if let Some(n_chunk) = self.size.get(0) {
            Some(self.data[n_chunk*range_column.start..n_chunk*range_column.end].chunks_exact(*n_chunk))
        }  else {
            None
        }
    }
    #[inline]
    pub fn iter_columns_full(&self) -> ChunksExact<T>{
        self.data.chunks_exact(self.size[0])
    }
    #[inline]
    pub fn par_iter_columns_full(&self) -> rayon::slice::ChunksExact<T>{
        self.data.par_chunks_exact(self.size[0])
    }
    #[inline]
    pub fn par_iter_columns(&self, range_column: Range<usize>) -> Option<rayon::slice::ChunksExact<T>>{
        if let Some(n_chunk) = self.size.get(0) {
            Some(self.data[n_chunk*range_column.start..n_chunk*range_column.end].par_chunks_exact(*n_chunk))
        }  else {
            None
        }
    }
    #[inline]
    pub fn get_slice_x(&self, y: usize) -> & [T] {
        let start = self.indicing[1]*y;
        let end = self.indicing[1]*(y+1);
        & self.data[start..end]
    }
    #[inline]
    pub fn transpose(&self) -> MatrixFull<T> {
        let x_len = self.size[0];
        let y_len = self.size[1];
        //let [x_len,y_len] = *self.size;
        let mut trans_mat = MatrixFull {
            size: [y_len,x_len],
            indicing: [0usize;2],
            data: self.data.to_vec()
        };
        let mut len = trans_mat.size.iter()
            .zip(trans_mat.indicing.iter_mut())
            .fold(1usize,|len,(di,ii)| {
                *ii = len;
                len * di
            });
        self.iter_columns_full().enumerate().for_each(|(i,c)| {
            trans_mat.iter_submatrix_mut(i..i+1,0..x_len).zip(c)
            .for_each(|(to,from)| {*to = *from})
        });
        trans_mat
    }
    #[inline]
    pub fn transpose_and_drop(self) -> MatrixFull<T> {
        let x_len = self.size[0];
        let y_len = self.size[1];
        //let [x_len,y_len] = *self.size;
        let mut trans_mat = MatrixFull {
            size: [y_len,x_len],
            indicing: [0usize;2],
            data: self.data.to_vec()
        };
        let mut len = trans_mat.size.iter()
            .zip(trans_mat.indicing.iter_mut())
            .fold(1usize,|len,(di,ii)| {
                *ii = len;
                len * di
            });
        self.iter_columns_full().enumerate().for_each(|(i,c)| {
            trans_mat.iter_submatrix_mut(i..i+1,0..x_len).zip(c)
            .for_each(|(to,from)| {*to = *from})
        });
        trans_mat
    }
}

#[derive(Debug,PartialEq)]
pub struct SubMatrixFullSlice<'a,T> {
    pub size : [usize;2],
    pub indicing: [usize;2],
    pub data : &'a [T]
    //pub data : [&'a mut T]
}

impl <'a, T> BasicMatrix<'a, T> for SubMatrixFullSlice<'a,T> {
    #[inline]
    /// `matr_a.size()' return &matr_a.size;
    fn size(&self) -> &[usize] {
        &self.size
    }
    #[inline]
    /// `matr_a.indicing()' return &matr_a.indicing;
    fn indicing(&self) -> &[usize] {
        &self.indicing
    }

    fn data_ref(&self) -> Option<&[T]> {
        Some(&self.data)
    }
    fn data_ref_mut(&mut self) -> Option<&mut [T]> {
        None
    }
}