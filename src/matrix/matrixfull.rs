//#![warn(missing_docs)]
use std::{fmt::Display, collections::binary_heap::Iter, iter::{Filter,Flatten, Map, StepBy}, convert, slice::{ChunksExact,ChunksExactMut, self}, mem::ManuallyDrop, marker, cell::RefCell, ops::{IndexMut, RangeFull, MulAssign, DivAssign, Div, DerefMut, Deref}, thread::panicking};
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, Index, Range};
use libc::{CLOSE_RANGE_CLOEXEC, SYS_userfaultfd};
use typenum::{U2, Pow};
use rayon::{prelude::*, collections::btree_map::IterMut, iter::Enumerate};
use std::vec::IntoIter;
use lapack::dgesv;
use regex::Regex;

use crate::matrix::{MatrixFull, BasicMatrix, MatFormat, MathMatrix, ParMathMatrix, BasicMatrixOpt};
use crate::{external_libs::matr_copy, check_shape};
use crate::index::*; 
use crate::tensor_basic_operation::*;
use crate::matrix::matrixfullslice::*;
use crate::matrix::matrixupper::*;
use crate::matrix::submatrixfull::*;
//{Indexing,Tensors4D};


impl <'a, T> BasicMatrix<'a, T> for MatrixFull<T> {
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
    #[inline]
    fn data_ref(&self) -> Option<&[T]> {
        Some(&self.data[..])
    }
    #[inline]
    fn data_ref_mut(&mut self) -> Option<&mut [T]> {
        Some(&mut self.data[..])
    }
}

impl<'a, T> BasicMatrixOpt<'a, T> for MatrixFull<T> where T: Copy + Clone {}

/// # more math operations for the rest package
/// 
///  These operators are provided by the trait of [`MathMatrix`](MathMatrix).  
///  Here, we illustrate the usage mainly using the [`MatrixFull`] struct. You can perform these operations for [`MatrixFullSlice`], 
///  [`MatrixFullSliceMut`], [`SubMatrixFull`], and [`SubMatrixFullMut`].
/// 
///  We also provide the rayon parallel version of these operators by the trait of [`ParMathMatrix`](ParMathMatrix).
/// 
///  - For  C = A + B, use [`MathMatrix::add`] and [`ParMathMatrix::par_add`]
///  - For  C = A + c*B, use [`MathMatrix::scaled_add`] and [`ParMathMatrix::par_scaled_add`]
///  - For  A += B, use [`MathMatrix::self_add`] and [`ParMathMatrix::par_self_add`]
///  - For  A += c*B, using [`MathMatrix::self_scaled_add`] and [`ParMathMatrix::par_scaled_add`]
///  - For  a*A + b*B -> A, using [`MathMatrix::self_general_add`] and [`ParMathMatrix::par_self_general_add`]
///  - For  C = A -B, use [`MathMatrix::sub`] and [`ParMathMatrix::par_sub`]
///  - For  A -= B, use [`MathMatrix::self_sub`] and [`ParMathMatrix::par_self_sub`]
///  - For  A *= a, use [`MathMatrix::self_multiple`] and [`ParMathMatrix::par_self_multiple`]
/// 
/// Several examples are given as follow:
/// 
/// * 1) add and self_sub
/// ```
///   use rest_tensors::{MatrixFull, MathMatrix};
///   let vec_a = vec![
///          1.0,  2.0,  3.0, 
///          4.0,  5.0,  6.0, 
///          7.0,  8.0,  9.0, 
///         10.0, 11.0, 12.0];
///   let matr_a = MatrixFull::from_vec([3,4],vec_a).unwrap();
///   //          |  1.0 |  4.0 |  7.0 | 10.0 |
///   //matr_a =  |  2.0 |  5.0 |  8.0 | 11.0 |
///   //          |  3.0 |  6.0 |  9.0 | 12.0 |
/// 
///   let vec_b = (13..25).map(|x| x as f64).collect::<Vec<f64>>();
///   let matr_b = MatrixFull::from_vec([3,4],vec_b).unwrap();
///   //          | 13.0 | 16.0 | 19.0 | 22.0 |
///   //matr_b =  | 14.0 | 17.0 | 20.0 | 23.0 |
///   //          | 15.0 | 18.0 | 21.0 | 24.0 |
/// 
///   // matr_c = matr_a + matr_b;   
///   let mut matr_c = MatrixFull::add(&matr_a, &matr_b).unwrap();
///   //          | 14.0 | 20.0 | 26.0 | 32.0 |
///   //matr_c =  | 16.0 | 22.0 | 28.0 | 34.0 |
///   //          | 18.0 | 24.0 | 30.0 | 36.0 |
///   assert_eq!(matr_c[(..,3)], [32.0,34.0,36.0]);
/// 
///   // matr_c_ref: MatrixFullSliceMut<f64>
///   let mut matr_c_ref = matr_c.to_matrixfullslicemut();
///   // matr_c -= matr_b = matr_a
///   matr_c_ref.self_sub(&matr_b);
///   assert_eq!(matr_c, matr_a);
/// 
/// ```
///  * 2) scaled add and self_sub 
/// ```
///   use rest_tensors::{MatrixFull, MathMatrix, ParMathMatrix};
///   let vec_a = vec![
///          1.0,  2.0,  3.0, 
///          4.0,  5.0,  6.0, 
///          7.0,  8.0,  9.0, 
///         10.0, 11.0, 12.0];
///   let matr_a = MatrixFull::from_vec([3,4],vec_a).unwrap();
///   //          |  1.0 |  4.0 |  7.0 | 10.0 |
///   //matr_a =  |  2.0 |  5.0 |  8.0 | 11.0 |
///   //          |  3.0 |  6.0 |  9.0 | 12.0 |
/// 
///   let vec_b = (13..25).map(|x| x as f64).collect::<Vec<f64>>();
///   let mut matr_b = MatrixFull::from_vec([3,4],vec_b).unwrap();
///   //          | 13.0 | 16.0 | 19.0 | 22.0 |
///   //matr_b =  | 14.0 | 17.0 | 20.0 | 23.0 |
///   //          | 15.0 | 18.0 | 21.0 | 24.0 |
/// 
///   // matr_c = 1.0*matr_a + matr_b
///   let mut matr_c = matr_b.scaled_add(&matr_a, 1.0).unwrap();
///   //          | 14.0 | 20.0 | 26.0 | 32.0 |
///   //matr_c =  | 16.0 | 22.0 | 28.0 | 34.0 |
///   //          | 18.0 | 24.0 | 30.0 | 36.0 |
///   assert_eq!(matr_c[(..,3)], [32.0,34.0,36.0]);
/// 
///   // matr_c_ref: MatrixFullSliceMut<f64>
///   let mut matr_c_ref = matr_c.to_matrixfullslicemut();
///   // matr_c -= matr_b = matr_a, using the rayon parallel version
///   matr_c_ref.par_self_sub(&matr_b);
///   assert_eq!(matr_c, matr_a);
/// ```
impl<'a, T> MathMatrix<'a, T> for MatrixFull<T> where T: Copy + Clone {}

impl<'a, T> ParMathMatrix<'a, T> for MatrixFull<T> where T: Copy + Clone + Send + Sync {}


pub struct IncreaseStepBy<I> {
    pub iter: I,
    step: usize,
    increase: usize,
    first_take: bool, 
}

impl<I> IncreaseStepBy<I> {
    pub fn new(iter: I, step: usize, increase: usize) -> IncreaseStepBy<I> {
        assert!(step!=0);
        IncreaseStepBy {iter, step, first_take: true, increase }
    }
}

impl<I> Iterator for IncreaseStepBy<I> 
where I: Iterator,
{
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.first_take {
            self.first_take = false;
            //self.step -= self.increase;
            self.iter.next()
        } else {
            self.step += self.increase;
            self.iter.nth(self.step-self.increase)
        }
    }
}

pub struct SubMatrixStepBy<I> {
    pub iter: I,
    rows: Range<usize>,
    columns: Range<usize>,
    size: [usize;2],
    step: usize,
    max: usize,
    position: usize,
    first_take: bool,
}
impl<I> SubMatrixStepBy<I> {
    pub fn new(iter: I, rows: Range<usize>, columns: Range<usize>, size:[usize;2]) -> SubMatrixStepBy<I> {
        let position =columns.start*size[0] + rows.start;
        let step = size[0]-rows.end+rows.start;
        let max = (columns.end-1)*size[0] + rows.end;
        SubMatrixStepBy{iter, rows, columns, size, step, position,max,first_take: true}
    }
}


impl<I> Iterator for SubMatrixStepBy<I>
where I:Iterator,
{
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {

        let curr_row = self.position%self.size[0];
        let curr_column = self.position/self.size[0];
        
        let is_in_range = 
            curr_row >= self.rows.start && curr_row < self.rows.end &&
            curr_column >= self.columns.start && curr_column < self.columns.end;

        if self.position >= self.max {
            None
        } else if self.first_take {
            self.position += 1;
            self.first_take = false;
            self.iter.nth(self.position-1)
        } else if is_in_range {
            //self.step -= self.increase;
            self.position += 1;
            self.iter.next()
        } else {
            self.position += self.step+1;
            self.iter.nth(self.step)
        }
    }
}

pub struct MatrixUpperStepBy<I> {
    pub iter: I,
    size: [usize;2],
    step: usize,
    position: usize,
    first_take: bool,
}

impl<I> MatrixUpperStepBy<I> {
    pub fn new(iter: I, size:[usize;2]) -> MatrixUpperStepBy<I> {
        let position =0;
        let step = size[0];
        MatrixUpperStepBy{iter, size, step, position,first_take: true}
    }
}

impl<I> Iterator for MatrixUpperStepBy<I>
where I:Iterator,
{
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {

        let curr_row = self.position%self.size[0];
        let curr_column = self.position/self.size[0];
        
        let is_in_range = curr_row <= curr_column;

        if self.first_take {
            self.position = 1;
            self.first_take = false;
            self.iter.next()
        } else if is_in_range {
            //self.step -= self.increase;
            self.position += 1;
            self.iter.next()
        } else {
            let step = self.size[0]-curr_column;
            self.position += step;
            self.iter.nth(step-1)
        }
    }
}

trait MatrixIterator: Iterator {
    type Item;
    fn step_by_increase(self, step:usize, increase: usize) -> IncreaseStepBy<Self>
    where Self:Sized {
        IncreaseStepBy::new(self, step, increase)
    }
    //pub fn new(iter: I, rows: Range<usize>, columns: Range<usize>, size:[usize;2]) -> SubMatrixStepBy<I> {
    fn submatrix_step_by(self, rows: Range<usize>, columns: Range<usize>, size:[usize;2]) -> SubMatrixStepBy<Self> 
    where Self:Sized {
        SubMatrixStepBy::new(self, rows, columns, size)
    }
    fn matrixupper_step_by(self, size:[usize;2]) -> MatrixUpperStepBy<Self> 
    where Self:Sized {
        MatrixUpperStepBy::new(self, size)
    }
}


impl<'a,T> MatrixIterator for std::slice::Iter<'a,T> {
    type Item = T;
}
impl<'a,T> MatrixIterator for std::slice::IterMut<'a,T> {
    type Item = T;
}


impl <T> MatrixFull<T> {
    /// initialize an empty MatrixFull entity
    pub fn empty() -> MatrixFull<T> {
        unsafe{MatrixFull::from_vec_unchecked([0,0],Vec::new())}
    }
    /// generate a new MatrixFull entity, where all elemental values as "new_default"
    pub fn new(size: [usize;2], new_default: T) -> MatrixFull<T> 
    where T: Copy + Clone
    {
        let mut indicing = [0usize;2];
        let mut len = size.iter()
            .zip(indicing.iter_mut())
            .fold(1usize,|len,(di,ii)| {
            *ii = len;
            len * di
        });
        MatrixFull {
            size,
            indicing,
            data: vec![new_default.clone(); len]
        }
    }
    pub fn data(&self) -> Vec<T> where T: Copy + Clone {self.data.clone()}

    pub unsafe fn from_vec_unchecked(size: [usize;2], new_vec: Vec<T>) -> MatrixFull<T> {
        let mut indicing = [0usize;2];
        let mut len = size.iter().zip(indicing.iter_mut()).fold(1usize,|len,(di,ii)| {
            *ii = len;
            len * di
        });
        MatrixFull {
            size,
            indicing,
            data: new_vec
        }
    }
    pub fn from_vec(size: [usize;2], new_vec: Vec<T>) -> Option<MatrixFull<T>> {
        unsafe{
            let tmp_mat = MatrixFull::from_vec_unchecked(size, new_vec);
            let len = tmp_mat.size.iter().product::<usize>();
            if len>tmp_mat.data.len() {
                panic!("Error: inconsistency happens when formating a tensor from a given vector, (length from size, length of new vector) = ({},{})",len,tmp_mat.data.len());
                None
            } else {
                if len<tmp_mat.data.len() {println!("Waring: the vector size ({}) is larger for the size of the new tensor ({})", tmp_mat.data.len(), len)};
                Some(tmp_mat)
            }

        }
    }
    /// reshape the matrix without change the data and its ordering
    pub fn reshape(&mut self, size:[usize;2]) {
        if size.iter().product::<usize>() !=self.data.len() {
            panic!("Cannot reshape a matrix ({:?}) and change the data length in the meantime: {:?}", &self.size,&size);
        } else {
            self.size = size;
            self.indicing = [0usize;2];
            let _ = self.size.iter().zip(self.indicing.iter_mut()).fold(1usize,|len,(di,ii)| {
                *ii = len;
                len * di
            });
        }
    }
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.data.iter()
    }
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
    #[inline]
    /// `iter_submatrix` provides a home-made submatrix StepBy iterator for the elements in the sub-matrix one by one
    /// Example
    /// ```
    ///   use rest_tensors::MatrixFull;
    ///   let mut matr_a = MatrixFull::from_vec(
    ///     [3,4],
    ///     (1..13).collect::<Vec<i32>>()
    /// ).unwrap();
    ///   //          |  1 |  4 |  7 | 10 |
    ///   //matr_a =  |  2 |  5 |  8 | 11 |  with the type of MatrixFull<i32>
    ///   //          |  3 |  6 |  9 | 12 |
    ///   let mut tmp_iter = matr_a.iter_submatrix_mut(1..2,0..4).collect::<Vec<&mut i32>>();
    ///   assert_eq!(tmp_iter, vec![&2,&5,&8,&11])
    /// ```
    pub fn iter_submatrix(&self, x: Range<usize>, y: Range<usize>) ->  SubMatrixStepBy<slice::Iter<T>>{
        self.iter().submatrix_step_by(x, y, self.size.clone())
    }
    #[inline]
    pub fn get_submatrix<'a>(&'a self, x: Range<usize>, y: Range<usize>) -> SubMatrixFull<T> {
        let new_x_len = x.len();
        let new_y_len = y.len();
        let new_vec: Vec<&T> = self.iter_submatrix(x,y).collect();

        SubMatrixFull::Detached(
            MatrixFull::from_vec([new_x_len, new_y_len], new_vec).unwrap())
    }
    pub fn iter_submatrix_old(&self, x: Range<usize>, y: Range<usize>) -> Flatten<IntoIter<&[T]>> {
        let mut tmp_slices = vec![&self.data[..]; y.len()];
        let len_slices_x = x.len();
        let len_y = self.indicing[1];
        tmp_slices.iter_mut().zip(y).for_each(|(t,y)| {
            let start = x.start + y*len_y;
            *t = &self.data[start..start + len_slices_x];
        });
        tmp_slices.into_iter().flatten()
    }
    #[inline]
    /// `iter_submatrix_mut` provides a flatten iterator for the mutable elements in the sub-matrix one by one
    /// **NOTE:: the current implementation is not efficient.
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
    pub fn iter_submatrix_mut(&mut self, x: Range<usize>, y: Range<usize>) ->  SubMatrixStepBy<slice::IterMut<T>>{
        let size = [self.size[0],self.size[1]];
        self.iter_mut().submatrix_step_by(x, y, size)
    }


    #[inline]
    pub fn get_submatrix_mut<'a>(&'a mut self, x: Range<usize>, y: Range<usize>) -> SubMatrixFullMut<T> 
    {
        let new_x_len = x.len();
        let new_y_len = y.len();
        let data =  self.iter_submatrix_mut(x,y).collect::<Vec<&mut T>>();

        SubMatrixFullMut::Detached(
            MatrixFull {
                size: [new_x_len, new_y_len],
                indicing: [1, new_x_len],
                data})
    }
    #[inline]
    pub fn iter_column(&self, y: usize) -> std::slice::Iter<T> {
        let len = self.indicing[1];
        self.data[y*len..(y+1)*len].iter()
    }
    #[inline]
    pub fn iter_column_mut(&mut self, y: usize) -> std::slice::IterMut<T> {
        let len = self.indicing[1];
        self.data[y*len..(y+1)*len].iter_mut()
    }
    #[inline]
    pub fn iter_columns(&self, range_column: Range<usize>) -> ChunksExact<T>{
        let n_chunk = self.indicing[1];
        self.data[n_chunk*range_column.start..n_chunk*range_column.end]
           .chunks_exact(n_chunk)
    }
    #[inline]
    pub fn iter_columns_mut(&mut self,range_column: Range<usize>) -> ChunksExactMut<T>{
        let n_chunk = self.indicing[1];
        self.data[n_chunk*range_column.start..n_chunk*range_column.end]
            .chunks_exact_mut(n_chunk)
    }
    #[inline]
    pub fn iter_columns_full(&self) -> ChunksExact<T>{
        self.data.chunks_exact(self.size[0])
    }
    #[inline]
    pub fn iter_columns_full_mut(&mut self) -> ChunksExactMut<T>{
        self.data.chunks_exact_mut(self.size[0])
    }
    #[inline]
    pub fn iter_row_old(&self, x: usize) -> Flatten<IntoIter<&[T]>> {
        let y_len = self.size[1];
        self.iter_submatrix_old(x..x+1,0..y_len)
    }
    #[inline]
    pub fn iter_row(&self, x: usize) -> StepBy<std::slice::Iter<T>> {
        self.slice()[x..].iter().step_by(self.indicing[1])
    }
    #[inline]
    pub fn iter_row_mut_old(&mut self, x: usize) -> Flatten<IntoIter<&mut [T]>> {
        let y_len = self.size[1];
        self.iter_submatrix_mut_old(x..x+1,0..y_len)
    }
    #[inline]
    pub fn iter_row_mut(&mut self, x: usize) -> StepBy<std::slice::IterMut<T>> {
        let step = self.indicing[1];
        self.slice_mut()[x..].iter_mut().step_by(step)
    }
    #[inline]
    pub fn iter_rows_old(&self, x: Range<usize>) -> Flatten<IntoIter<&[T]>> {
        let y = 0..self.size[1];
        self.iter_submatrix_old(x,y)
    }
    #[inline]
    pub fn iter_rows(&self, x: Range<usize>) -> SubMatrixStepBy<slice::Iter<T>>  {
        let y = 0..self.size[1];
        self.iter_submatrix(x,y)
    }
    #[inline]
    pub fn iter_rows_mut_old(&mut self, x: Range<usize>) -> Flatten<IntoIter<&mut [T]>> {
        let y = 0..self.size[1];
        self.iter_submatrix_mut_old(x,y)
    }
    #[inline]
    pub fn iter_rows_mut(&mut self, x: Range<usize>) -> SubMatrixStepBy<slice::IterMut<T>>  {
        let y = 0..self.size[1];
        self.iter_submatrix_mut(x,y)
    }
    #[inline]
    pub fn iter_diagonal<'a>(&'a self) -> Option<StepBy<std::slice::Iter<T>>> {
        let [x,y] = self.size;
        if x==0 || y==0 || x!=y {
            return None
        } else {
            return Some(self.iter().step_by(x+1))
        }
    }
    pub fn iter_diagonal_mut<'a>(&'a mut self) -> Option<StepBy<std::slice::IterMut<T>>> {
        let [x,y] = self.size;
        if x==0 || y==0 || x!=y {
            return None
        } else {
            return Some(self.iter_mut().step_by(x+1))
        }
    }
    #[inline]
    pub fn iter_matrixupper<'a>(&'a self) -> Option<MatrixUpperStepBy<std::slice::Iter<T>>> {
        let [x,y] = self.size;
        if x==0 || y==0 || x!=y {
            return None
        } else {
            return Some(self.iter().matrixupper_step_by([x,y]))
        }
    }
    #[inline]
    pub fn iter_matrixupper_mut<'a>(&'a mut self) -> Option<MatrixUpperStepBy<std::slice::IterMut<T>>> {
        let [x,y] = self.size;
        if x==0 || y==0 || x!=y {
            return None
        } else {
            return Some(self.iter_mut().matrixupper_step_by([x,y]))
        }
    }
    #[inline]
    pub fn slice(&self) -> & [T] {
        &self.data[..]
    }
    #[inline]
    pub fn slice_mut(&mut self) -> &mut [T] {
        &mut self.data[..]
    }
    #[inline]
    pub fn slice_column(&self, y: usize) -> & [T] {
        let start = self.indicing[1]*y;
        let end = self.indicing[1]*(y+1);
        & self.data[start..end]
    }
    #[inline]
    pub fn slice_column_mut(&mut self, y: usize) -> &mut [T] {
        let start = self.indicing[1]*y;
        let end = self.indicing[1]*(y+1);
        &mut self.data[start..end]
    }
    #[inline]
    pub fn slice_columns(&self, y: Range<usize>) -> & [T] {
        let start = self.indicing[1]*y.start;
        let end = self.indicing[1]*y.end;
        & self.data[start..end]
    }
    #[inline]
    pub fn slice_columns_mut(&mut self, y: Range<usize>) -> &mut [T] {
        let start = self.indicing[1]*y.start;
        let end = self.indicing[1]*y.end;
        &mut self.data[start..end]
    }
}

/// # more matrix operations needed by the rest package
/// - For transpose A -> A^{T}, use [`MatrixFull::transpose`] and [`MatrixFull::transpose_and_drop`]
/// ```
///   use rest_tensors::MatrixFull;
///   let mut matr_a = MatrixFull::from_vec(
///     [3,4],
///     (1..13).collect::<Vec<i32>>()
/// ).unwrap();
///   //          |  1 |  4 |  7 | 10 |
///   //matr_a =  |  2 |  5 |  8 | 11 |  with the type of MatrixFull<i32>
///   //          |  3 |  6 |  9 | 12 |
/// 
///   // the second-row elements of matr_a
///   let row_2 = matr_a.iter_row(2).map(|x| *x).collect::<Vec<i32>>();
///   assert_eq!(row_2,vec![3,6,9,12]);
/// 
///   // transpose matr_a to matr_b
///   let mut matr_b = matr_a.transpose();
///   // the second-column elements of matr_b
///   let column_2 = matr_b.iter_column(2).map(|x| *x).collect::<Vec<i32>>();
///   assert_eq!(column_2,vec![3,6,9,12]);
///   
///   assert_eq!(column_2, row_2)
/// ```
/// - Collect the (mut)-refs of the diagonal elements in a vector, use [`MatrixFull::get_diagonal_terms`] and [`MatrixFull::get_diagonal_terms_mut`]  
///  **NOTE**: because this operation creates a new vector to store the (mut)-refs, it is not efficient. A better way to maniputate the diagonal terms
///  is to use the iterators of [`MatrixFull::iter_diagonal`] and [`MatrixFull::iter_diagonal_mut`] directly
/// ```
///   use rest_tensors::MatrixFull;
///   let mut matr_a = MatrixFull::from_vec(
///     [4,4],
///     (1..17).collect::<Vec<i32>>()
/// ).unwrap();
///   //          |  1 |  5 |  9 | 13 |
///   //matr_a =  |  2 |  6 | 10 | 14 |  with the type of MatrixFull<i32>
///   //          |  3 |  7 | 11 | 15 |
///   //          |  4 |  8 | 12 | 16 |
/// 
///   // the second-row elements of matr_a
///   let diagonal = matr_a.get_diagonal_terms().unwrap();
///   assert_eq!(diagonal,vec![&1,&6,&11,&16]);
/// ```
/// - Collect and copy the upper part of the matrix into the [`MatrixUpper`](MatrixUpper) Struct, use [`MatrixFull::to_matrixupper`]
/// - Get the (mutable) reference to the matrix, and encapsulate in various structs for different uses:  
/// -- [`MatrixFull::to_matrixfullslice`] to [`MatrixFullSlice`]  
/// -- [`MatrixFull::to_matrixfullslicemut`] to [`MatrixFullSliceMut`]  
/// -- [`MatrixFull::to_matrixfullslice_columns`] to [`SubMatrixFullSlice`]
impl <T: Copy + Clone> MatrixFull<T> {
    /// Transpose the matrix: A -> A^{T}
    #[inline]
    pub fn transpose(&self) -> MatrixFull<T> {
        let mut trans_mat = self.clone();
        let [x_len,y_len] = self.size;
        trans_mat.size=[y_len,x_len];
        trans_mat.indicing = [0usize;2];
        let mut len = trans_mat.size.iter()
            .zip(trans_mat.indicing.iter_mut())
            .fold(1usize,|len,(di,ii)| {
                *ii = len;
                len * di
            });
        self.iter_columns_full().enumerate().for_each(|(i,c)| {
            trans_mat.iter_row_mut(i).zip(c)
            .for_each(|(to,from)| {*to = *from})
        });
        trans_mat
    }
    /// Transpose the matrix: A -> A^{T}
    #[inline]
    pub fn transpose_and_drop(self) -> MatrixFull<T> {
        let mut trans_mat = self.clone();
        let [x_len,y_len] = self.size;
        trans_mat.size=[y_len,x_len];
        trans_mat.indicing = [0usize;2];
        let mut len = trans_mat.size.iter()
            .zip(trans_mat.indicing.iter_mut())
            .fold(1usize,|len,(di,ii)| {
                *ii = len;
                len * di
            });
        self.iter_columns_full().enumerate().for_each(|(i,c)| {
            trans_mat.iter_row_mut(i).zip(c)
            .for_each(|(to,from)| {*to = *from})
        });
        trans_mat
    }
    /// # Collect the reference of diagonal terms as a vector
    #[inline]
    pub fn get_diagonal_terms<'a>(&'a self) -> Option<Vec<&T>> {
        //let tmp_len = self.size;
        let [x,y] = self.size;
        if x==0 || y==0 || x!=y {
            return None
        } else {
            let tmp_vec = self.iter_diagonal().unwrap().collect::<Vec<&T>>();
            return Some(tmp_vec)
        }
    }
    #[inline]
    pub fn get_diagonal_terms_mut(&mut self) -> Option<Vec<&mut T>> {
        let [x,y] = self.size;
        if x==0 || y==0 || x!=y {
            return None
        } else {
            let tmp_v = self.iter_diagonal_mut().unwrap().collect::<Vec<&mut T>>();
            return Some(tmp_v)
        }
    }
    #[inline]
    pub fn to_matrixupper(&self) -> MatrixUpper<T> {
        if self.size[0]!=self.size[1] {
            panic!("Error: Nonsymmetric matrix cannot be converted to the upper format");
        }
        unsafe{MatrixUpper::from_vec_unchecked(
            self.size[0]*(self.size[0]+1)/2,
            self.iter_matrixupper().unwrap().map(|ad| ad.clone()).collect::<Vec<T>>()
        )}
    }
    #[inline]
    pub fn to_matrixfullslicemut(&mut self) -> MatrixFullSliceMut<T> {
        MatrixFullSliceMut {
            size: &self.size[0..2],
            indicing: &self.indicing[0..2],
            data: &mut self.data[..],
        }
    }
    #[inline]
    pub fn to_matrixfullslice(&self) -> MatrixFullSlice<T> {
        MatrixFullSlice {
            size: &self.size[0..2],
            indicing: &self.indicing[0..2],
            data: & self.data[..],
        }
    }
    #[inline]
    pub fn to_matrixfullslice_columns(&self,range_columns: Range<usize>) -> SubMatrixFullSlice<T> {
        let start = range_columns.start*self.indicing[1];
        let end = start + range_columns.len()*self.indicing[1];
        SubMatrixFullSlice {
            size: [self.size[0],range_columns.len()],
            indicing: [0,self.size[0]],
            data: & self.data[start..end],
        }
    }
}


/// Useful iterators with rayon parallization
impl <T: Copy + Clone + Send + Sync + Sized> MatrixFull<T> {
    #[inline]
    pub fn par_iter_column(&self, j: usize) -> rayon::slice::Iter<T> {
        let start = self.indicing[1]*j;
        let end = start + self.indicing[1];
        self.data[start..end].par_iter()
    }
    #[inline]
    pub fn par_iter_column_mut(&mut self, j: usize) -> rayon::slice::IterMut<T> {
        let start = self.indicing[1]*j;
        let end = start + self.indicing[1];
        self.data[start..end].par_iter_mut()
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
    pub fn par_iter_columns_mut(&mut self,range_column: Range<usize>) -> Option<rayon::slice::ChunksExactMut<T>>{
        if let Some(n_chunk) = self.size.get(0) {
            Some(self.data[n_chunk*range_column.start..n_chunk*range_column.end].par_chunks_exact_mut(*n_chunk))
        }  else {
            None
        }
    }
    #[inline]
    pub fn par_iter_columns_full(&self) -> rayon::slice::ChunksExact<T>{
        self.data.par_chunks_exact(self.size[0])
    }
    #[inline]
    pub fn par_iter_columns_full_mut(&mut self) -> rayon::slice::ChunksExactMut<T>{
        self.data.par_chunks_exact_mut(self.size[0])
    }
}


//==========================================================================
// Now implement the Index and IndexMut traits for MatrixFull
//==========================================================================
impl<T> Index<[usize;2]> for MatrixFull<T> {
    type Output = T;
    fn index(&self, p:[usize;2]) -> &Self::Output {
        self.get2d(p).unwrap()
    }
}

impl<T> IndexMut<[usize;2]> for MatrixFull<T> {
    fn index_mut(&mut self, p:[usize;2]) -> &mut Self::Output {
        self.get2d_mut(p).unwrap()
    }
}

impl<T> Index<(usize,usize)> for MatrixFull<T> {
    type Output = T;
    fn index(&self, p:(usize,usize)) -> &Self::Output {
        self.get2d([p.0,p.1]).unwrap()
    }
}
impl<T> IndexMut<(usize,usize)> for MatrixFull<T> {
    fn index_mut(&mut self, p:(usize,usize)) -> &mut Self::Output {
        self.get2d_mut([p.0,p.1]).unwrap()
    }
}
impl<T> Index<usize> for MatrixFull<T> {
    type Output = T;
    fn index(&self, position:usize) -> &Self::Output {
        self.data.get(position).unwrap()
    }
}
impl<T> IndexMut<usize> for MatrixFull<T> {
    /// refer to the slice with a given 
    fn index_mut(&mut self, position:usize) -> &mut Self::Output {
        self.data.get_mut(position).unwrap()
    }
}
impl<T> Index<(Range<usize>,usize)> for MatrixFull<T> {
    type Output = [T];
    fn index(&self, p:(Range<usize>,usize)) -> &Self::Output {
        self.get2d_slice([p.0.start,p.1],p.0.len()).unwrap()
    }
}
impl<T> IndexMut<(Range<usize>,usize)> for MatrixFull<T> {
    fn index_mut(&mut self, p:(Range<usize>,usize)) -> &mut Self::Output {
        self.get2d_slice_mut([p.0.start,p.1],p.0.len()).unwrap()
    }
}
impl<T> Index<(RangeFull,Range<usize>)> for MatrixFull<T> {
    type Output = [T];
    fn index(&self, p:(RangeFull,Range<usize>)) -> &Self::Output {
        self.get_slice(&[0,p.1.start],self.size[0]*p.1.len()).unwrap()
    }
}
impl<T> IndexMut<(RangeFull,Range<usize>)> for MatrixFull<T> {
    fn index_mut(&mut self, p:(RangeFull,Range<usize>)) -> &mut Self::Output {
        self.get_slice_mut(&[0,p.1.start],self.size[0]*p.1.len()).unwrap()
    }
}
impl<T> Index<(RangeFull,usize)> for MatrixFull<T> {
    type Output = [T];
    fn index(&self, p:(RangeFull,usize)) -> &Self::Output {
        self.get_slice(&[0,p.1],self.size[0]).unwrap()
    }
}
impl<T> IndexMut<(RangeFull,usize)> for MatrixFull<T> {
    fn index_mut(&mut self, p:(RangeFull,usize)) -> &mut Self::Output {
        self.get_slice_mut(&[0,p.1],self.size[0]).unwrap()
    }
}

//==========================================================================
// Now implement Add and Sub traits for MatrixFull and SubMatrixFull(Mut)
//==========================================================================

impl<T: Clone + Add + AddAssign> Add<MatrixFull<T>> for MatrixFull<T> {
    type Output = Self;

    fn add(self, other: MatrixFull<T>) -> MatrixFull<T> {
        if ! check_shape(&self, &other) {
            panic!("It is not allowed to add two matrices with different size: {:?}, {:?}", &self.size, &other.size);
        }
        let mut new_tensor: MatrixFull<T> = self.clone();
        new_tensor.data.iter_mut().zip(other.data.iter()).for_each(|(t,f)| {*t += f.clone()});
        new_tensor
    }
}
impl<T: Clone + Add + AddAssign> AddAssign<MatrixFull<T>> for MatrixFull<T> {
    fn add_assign(&mut self, other: MatrixFull<T>) {
        //let mut new_tensor: MatrixFull<T> = self.clone();
        if ! check_shape(self, &other) {
            panic!("It is not allowed to add two matrices with different size: {:?}, {:?}", &self.size, &other.size);
        }
        self.data.iter_mut().zip(other.data.iter()).for_each(|(t,f)| {*t += f.clone()});
    }
}
impl<T: Clone + Add + AddAssign> Add<T> for MatrixFull<T> {
    type Output = Self;

    fn add(self, factor: T) -> MatrixFull<T> {
        let mut new_tensor: MatrixFull<T> = self.clone();
        new_tensor.data.iter_mut().for_each(|t| {*t += factor.clone()});
        new_tensor
    }
}
impl<T: Clone + Add + AddAssign> AddAssign<T> for MatrixFull<T> {
    fn add_assign(&mut self, factor: T) {
        //let mut new_tensor: MatrixFull<T> = self.clone();
        self.data.iter_mut().for_each(|t| {*t += factor.clone()});
    }
}

impl<'a, T: Copy + Clone + Add + AddAssign> Add<T> for SubMatrixFull<'a, T> {
    type Output = MatrixFull<T>;

    fn add(self, factor: T) -> MatrixFull<T> {
        let size = self.size().clone();
        let size = [size[0],size[1]];
        match self {
            Self::Contiguous(matr) => {
                let mut new_vec = matr.data.iter().map(|x| x.clone()).collect::<Vec<T>>();
                MatrixFull::from_vec(size,new_vec).unwrap() + factor
            },
            Self::Detached(matr) => {
                let mut new_vec = matr.data.iter().map(|x| *x.clone()).collect::<Vec<T>>();
                MatrixFull::from_vec(size,new_vec).unwrap() + factor
            }
        }
    }
}
impl<'a, T: Copy + Clone + Add + AddAssign> AddAssign<T> for SubMatrixFullMut<'a, T> {
    fn add_assign(&mut self, factor: T) {
        let size = self.size().clone();
        match self {
            Self::Contiguous(matr) => {
                matr.data.iter_mut().for_each(|t| {*t += factor});
            },
            Self::Detached(matr) => {
                matr.data.iter_mut().for_each(|t| {**t += factor});
            }
        }
    }
}

impl<'a, T: Copy + Clone + Add + AddAssign> Add<SubMatrixFull<'a, T>> for MatrixFull<T> {
    type Output = Self;

    fn add(self, other: SubMatrixFull<T>) -> MatrixFull<T> {
        let o_size = other.size().clone();
        let check_shape = self.size.iter().zip(o_size.iter()).fold(true, |flag, (a,b)| flag && *a==*b);
        if ! check_shape {
            panic!("It is not allowed to add two matrices with different size: {:?}, {:?}", &self.size, &o_size);
        }
        let mut new_tensor: MatrixFull<T> = self.clone();
        match &other {
            SubMatrixFull::Contiguous(matr) => {
                new_tensor.data.iter_mut().zip(matr.data.iter()).for_each(|(t,f)| {*t += f.clone()});

            },
            SubMatrixFull::Detached(matr) => {
                new_tensor.data.iter_mut().zip(matr.data.iter()).for_each(|(t,f)| {*t += *f.clone()});
            },
        }
        new_tensor
    }
}
impl<'a, T: Copy + Clone + Add + AddAssign> AddAssign<SubMatrixFull<'a, T>> for MatrixFull<T> {
    fn add_assign(&mut self, other: SubMatrixFull<T>) {
        let o_size = other.size().clone();
        let check_shape = self.size.iter().zip(o_size.iter()).fold(true, |flag, (a,b)| flag && *a==*b);
        if ! check_shape {
            panic!("It is not allowed to add two matrices with different size: {:?}, {:?}", &self.size, &o_size);
        }
        match &other {
            SubMatrixFull::Contiguous(matr) => {
                self.data.iter_mut().zip(matr.data.iter()).for_each(|(t,f)| {*t += f.clone()});
            },
            SubMatrixFull::Detached(matr) => {
                self.data.iter_mut().zip(matr.data.iter()).for_each(|(t,f)| {*t += *f.clone()});
            }
        }
    }
}

impl<'a, T: Copy + Clone + Add + AddAssign> Add<MatrixFull<T>> for SubMatrixFull<'a,T> {
    type Output = MatrixFull<T>;

    fn add(self, other: MatrixFull<T>) -> MatrixFull<T> {
        let self_size = self.size().clone();
        let check_shape = self_size.iter().zip(other.size.iter()).fold(true, |flag, (a,b)| flag && *a==*b);
        if ! check_shape {
            panic!("It is not allowed to add two matrices with different size: {:?}, {:?}", &self_size, &other.size);
        }
        let mut new_tensor: MatrixFull<T> = other.clone();
        match &self {
            SubMatrixFull::Contiguous(matr) => {
                new_tensor.data.iter_mut().zip(matr.data.iter()).for_each(|(t,f)| {*t += f.clone()});
            },
            SubMatrixFull::Detached(matr) => {
                new_tensor.data.iter_mut().zip(matr.data.iter()).for_each(|(t,f)| {*t += *f.clone()});
            },
        }
        new_tensor
    }
}
impl<'a, T: Clone + Add + AddAssign> AddAssign<MatrixFull<T>> for SubMatrixFullMut<'a, T> {
    fn add_assign(&mut self, other: MatrixFull<T>) {
        let self_size = self.size().clone();
        let check_shape = self_size.iter().zip(other.size.iter()).fold(true, |flag, (a,b)| flag && *a==*b);
        if ! check_shape {
            panic!("It is not allowed to add two matrices with different size: {:?}, {:?}", &self_size, &other.size);
        }
        match self {
            SubMatrixFullMut::Contiguous(matr) => {
                matr.data.iter_mut().zip(other.data.iter()).for_each(|(t,f)| {*t += f.clone()});
            },
            SubMatrixFullMut::Detached(matr) => {
                matr.data.iter_mut().zip(other.data.iter()).for_each(|(t,f)| {**t += f.clone()});
            },
        }
    }
}
impl<'a, T: Copy + Clone + Add + AddAssign> AddAssign<SubMatrixFull<'a, T>> for SubMatrixFullMut<'a, T> {
    fn add_assign(&mut self, other: SubMatrixFull<T>) {
        let s_size = self.size().clone();
        let o_size = other.size().clone();
        let check_shape = s_size.iter().zip(o_size.iter()).fold(true, |flag, (a,b)| flag && *a==*b);
        if ! check_shape {
            panic!("It is not allowed to add two matrices with different size: {:?}, {:?}", &s_size, &o_size);
        }
        match (self, &other) {
            (SubMatrixFullMut::Contiguous(s_matr), SubMatrixFull::Contiguous(o_matr)) => {
                s_matr.data.iter_mut().zip(o_matr.data.iter()).for_each(|(t,f)| {*t += f.clone()});
            },
            (SubMatrixFullMut::Contiguous(s_matr), SubMatrixFull::Detached(o_matr)) => {
                s_matr.data.iter_mut().zip(o_matr.data.iter()).for_each(|(t,f)| {*t += *f.clone()});
            },
            (SubMatrixFullMut::Detached(s_matr), SubMatrixFull::Contiguous(o_matr)) => {
                s_matr.data.iter_mut().zip(o_matr.data.iter()).for_each(|(t,f)| {**t += f.clone()});
            },
            (SubMatrixFullMut::Detached(s_matr), SubMatrixFull::Detached(o_matr)) => {
                s_matr.data.iter_mut().zip(o_matr.data.iter()).for_each(|(t,f)| {**t += *f.clone()});
            },
        }
    }
}


impl<T: Clone + Sub + SubAssign> Sub<MatrixFull<T>> for MatrixFull<T> {
    type Output = Self;
    fn sub(self, other: MatrixFull<T>) -> MatrixFull<T> {
        if ! check_shape(&self, &other) {
            panic!("It is not allowed to subtract two matrices with different size: {:?}, {:?}", &self.size, &other.size);
        }
        let mut new_tensors: MatrixFull<T> = self.clone();
        new_tensors.data.iter_mut()
            .zip(other.data.iter())
            .for_each(|(t,f)| {
                *t -= f.clone()
        });
        new_tensors
    }
}
impl<T: Clone + Sub + SubAssign> SubAssign<MatrixFull<T>> for MatrixFull<T> {
    fn sub_assign(&mut self, other: MatrixFull<T>){
        if ! check_shape(self, &other) {
            panic!("It is not allowed to subtract two matrices with different size: {:?}, {:?}", &self.size, &other.size);
        }
        self.data.iter_mut()
            .zip(other.data.iter())
            .for_each(|(t,f)| {
                *t -= f.clone()
        });
    }
}
impl<T: Clone + Sub + SubAssign> Sub<T> for MatrixFull<T> {
    type Output = MatrixFull<T>;
    fn sub(self, f: T) -> MatrixFull<T> {
        let mut new_tensors: MatrixFull<T> = self.clone();
        new_tensors.data.iter_mut().for_each(|t| {*t -= f.clone()});
        new_tensors
    }
}
impl<T: Clone + Sub + SubAssign> SubAssign<T> for MatrixFull<T> {
    fn sub_assign(&mut self, f: T){
        self.data.iter_mut().for_each(|t| {*t -= f.clone()});
    }
}

impl<'a, T: Copy + Clone + Sub + SubAssign> Sub<T> for SubMatrixFull<'a,T> {
    type Output = MatrixFull<T>;
    fn sub(self, f: T) -> MatrixFull<T> {
        let s_size = self.size().clone();
        let s_size = [s_size[0],s_size[1]];
        let mut new_tensor: MatrixFull<T> = match &self {
            SubMatrixFull::Contiguous(matr) => {
                MatrixFull::from_vec(s_size, matr.data.iter().map(|x| x.clone()).collect::<Vec<T>>()).unwrap()
            },
            SubMatrixFull::Detached(matr) => {
                MatrixFull::from_vec(s_size, matr.data.iter().map(|x| *x.clone()).collect::<Vec<T>>()).unwrap()
            },
        };
        new_tensor.data.iter_mut().for_each(|t| {*t -= f.clone()});
        new_tensor
    }
}
impl<'a, T: Clone + Sub + SubAssign> SubAssign<T> for SubMatrixFullMut<'a, T> {
    fn sub_assign(&mut self, f: T){
        match self {
            SubMatrixFullMut::Contiguous(matr) => {
                matr.data.iter_mut().for_each(|t| {*t -= f.clone()});
            },
            SubMatrixFullMut::Detached(matr) => {
                matr.data.iter_mut().for_each(|t| {**t -= f.clone()});
            },
        }
    }
}

impl<'a, T: Copy + Clone + Sub + SubAssign> Sub<SubMatrixFull<'a, T>> for MatrixFull<T> {
    type Output = Self;
    fn sub(self, other: SubMatrixFull<T>) -> MatrixFull<T> {
        let o_size = other.size().clone();
        let check_shape = self.size.iter().zip(o_size.iter()).fold(true, |flag, (a,b)| flag && *a==*b);
        if ! check_shape {
            panic!("It is not allowed to subtract two matrices with different size: {:?}, {:?}", &self.size, &o_size);
        }
        let mut new_tensors: MatrixFull<T> = self.clone();
        match &other {
            SubMatrixFull::Contiguous(matr) => {
                new_tensors.data.iter_mut().zip(matr.data.iter()).for_each(|(t,f)| {*t -= f.clone()});
            },
            SubMatrixFull::Detached(matr) => {
                new_tensors.data.iter_mut().zip(matr.data.iter()).for_each(|(t,f)| {*t -= *f.clone()});
            },
        }
        new_tensors
    }
}
impl<'a, T: Copy + Clone + Sub + SubAssign> SubAssign<SubMatrixFull<'a, T>> for MatrixFull<T> {
    fn sub_assign(&mut self, other: SubMatrixFull<T>){
        let o_size = other.size().clone();
        let check_shape = self.size.iter().zip(o_size.iter()).fold(true, |flag, (a,b)| flag && *a==*b);
        if ! check_shape {
            panic!("It is not allowed to subtract two matrices with different size: {:?}, {:?}", &self.size, &o_size);
        }
        match &other {
            SubMatrixFull::Contiguous(matr) => {
                self.data.iter_mut().zip(matr.data.iter()).for_each(|(t,f)| {*t -= f.clone()});
            },
            SubMatrixFull::Detached(matr) => {
                self.data.iter_mut().zip(matr.data.iter()).for_each(|(t,f)| {*t -= *f.clone()});
            },
        }
    }
}
impl<'a, T: Copy + Clone + Sub<Output=T> + SubAssign> Sub<MatrixFull<T>> for SubMatrixFull<'a,T> {
    type Output = MatrixFull<T>;

    fn sub(self, other: MatrixFull<T>) -> MatrixFull<T> {
        let s_size = self.size().clone();
        let check_shape = s_size.iter().zip(other.size.iter()).fold(true, |flag, (a,b)| flag && *a==*b);
        if ! check_shape {
            panic!("It is not allowed to subtract two matrices with different size: {:?}, {:?}", &s_size, &other.size);
        }
        let mut new_tensor: MatrixFull<T> = other.clone();
        match &self {
            SubMatrixFull::Contiguous(matr) => {
                new_tensor.data.iter_mut().zip(matr.data.iter()).for_each(|(t,f)| {*t = (f.clone()-*t)});
            },
            SubMatrixFull::Detached(matr) => {
                new_tensor.data.iter_mut().zip(matr.data.iter()).for_each(|(t,f)| {*t = (*f.clone()-*t)});
            },
        }
        new_tensor
    }
}
impl<'a, T: Clone + Sub + SubAssign> SubAssign<MatrixFull<T>> for SubMatrixFullMut<'a, T> {
    fn sub_assign(&mut self, other: MatrixFull<T>) {
        let s_size = self.size().clone();
        let check_shape = s_size.iter().zip(other.size.iter()).fold(true, |flag, (a,b)| flag && *a==*b);
        if ! check_shape {
            panic!("It is not allowed to subtract two matrices with different size: {:?}, {:?}", &s_size, &other.size);
        }
        match self {
            SubMatrixFullMut::Contiguous(matr) => {
                matr.data.iter_mut().zip(other.data.iter()).for_each(|(t,f)| {*t -= f.clone()});
            },
            SubMatrixFullMut::Detached(matr) => {
                matr.data.iter_mut().zip(other.data.iter()).for_each(|(t,f)| {**t -= f.clone()});
            },
        }
    }
}
impl<'a, T: Copy + Clone + Sub + SubAssign> SubAssign<SubMatrixFull<'a, T>> for SubMatrixFullMut<'a, T> {
    fn sub_assign(&mut self, other: SubMatrixFull<T>) {
        let s_size = self.size().clone();
        let o_size = other.size().clone();
        let check_shape = s_size.iter().zip(o_size.iter()).fold(true, |flag, (a,b)| flag && *a==*b);
        if ! check_shape {
            panic!("It is not allowed to subtract two matrices with different size: {:?}, {:?}", &s_size, &o_size);
        }
        match (self, &other) {
            (SubMatrixFullMut::Contiguous(s_matr), SubMatrixFull::Contiguous(o_matr)) => {
                s_matr.data.iter_mut().zip(o_matr.data.iter()).for_each(|(t,f)| {*t -= f.clone()});
            },
            (SubMatrixFullMut::Contiguous(s_matr), SubMatrixFull::Detached(o_matr)) => {
                s_matr.data.iter_mut().zip(o_matr.data.iter()).for_each(|(t,f)| {*t -= *f.clone()});
            },
            (SubMatrixFullMut::Detached(s_matr), SubMatrixFull::Contiguous(o_matr)) => {
                s_matr.data.iter_mut().zip(o_matr.data.iter()).for_each(|(t,f)| {**t -= f.clone()});
            },
            (SubMatrixFullMut::Detached(s_matr), SubMatrixFull::Detached(o_matr)) => {
                s_matr.data.iter_mut().zip(o_matr.data.iter()).for_each(|(t,f)| {**t -= *f.clone()});
            },
        }
    }
}
impl<T: Clone + Mul + MulAssign> Mul<T> for MatrixFull<T> {
    type Output = MatrixFull<T>;
    fn mul(self, factor: T) -> MatrixFull<T> {
        let mut new_tensors: MatrixFull<T> = self.clone();
        new_tensors.data.iter_mut().for_each(|t| {*t *= factor.clone()});
        new_tensors
    }
}
impl<T: Clone + Mul + MulAssign> MulAssign<T> for MatrixFull<T> {
    fn mul_assign(&mut self, factor: T){
        self.data.iter_mut().for_each(|t| {*t *= factor.clone()});
    }
}

impl<'a, T: Copy + Clone + Mul + MulAssign> Mul<T> for SubMatrixFull<'a,T> {
    type Output = MatrixFull<T>;
    fn mul(self, f: T) -> MatrixFull<T> {
        let s_size = self.size().clone();
        let s_size = [s_size[0],s_size[1]];
        let mut new_tensor: MatrixFull<T> = match &self {
            SubMatrixFull::Contiguous(matr) => {
                MatrixFull::from_vec(s_size, matr.data.to_vec()).unwrap()
            },
            SubMatrixFull::Detached(matr) => {
                MatrixFull::from_vec(s_size, matr.data.iter().map(|x| *x.clone()).collect::<Vec<T>>()).unwrap()
            },
        };
        new_tensor.data.iter_mut().for_each(|t| {*t *= f.clone()});
        new_tensor
    }
}
impl<'a, T: Clone + Mul + MulAssign> MulAssign<T> for SubMatrixFullMut<'a, T> {
    fn mul_assign(&mut self, f: T){
        match self {
            SubMatrixFullMut::Contiguous(matr) => {
                matr.data.iter_mut().for_each(|t| {*t *= f.clone()});
            },
            SubMatrixFullMut::Detached(matr) => {
                matr.data.iter_mut().for_each(|t| {**t *= f.clone()});
            },
        }
    }
}
impl<T: Clone + Div + DivAssign> Div<T> for MatrixFull<T> {
    type Output = MatrixFull<T>;
    fn div(self, factor: T) -> MatrixFull<T> {
        let mut new_tensors: MatrixFull<T> = self.clone();
        new_tensors.data.iter_mut().for_each(|t| {*t /= factor.clone()});
        new_tensors
    }
}
impl<T: Clone + Div + DivAssign> DivAssign<T> for MatrixFull<T> {
    fn div_assign(&mut self, factor: T){
        self.data.iter_mut().for_each(|t| {*t /= factor.clone()});
    }
}
impl<'a, T: Copy + Clone + Div + DivAssign> Div<T> for SubMatrixFull<'a,T> {
    type Output = MatrixFull<T>;
    fn div(self, f: T) -> MatrixFull<T> {
        let s_size = self.size().clone();
        let s_size = [s_size[0],s_size[1]];
        let mut new_tensor: MatrixFull<T> = match &self {
            SubMatrixFull::Contiguous(matr) => {
                MatrixFull::from_vec(s_size, matr.data.to_vec()).unwrap()
            },
            SubMatrixFull::Detached(matr) => {
                MatrixFull::from_vec(s_size, matr.data.iter().map(|x| *x.clone()).collect::<Vec<T>>()).unwrap()
            },
        };
        new_tensor.data.iter_mut().for_each(|t| {*t /= f.clone()});
        new_tensor
    }
}
impl<'a, T: Clone + Div + DivAssign> DivAssign<T> for SubMatrixFullMut<'a, T> {
    fn div_assign(&mut self, f: T){
        match self {
            SubMatrixFullMut::Contiguous(matr) => {
                matr.data.iter_mut().for_each(|t| {*t /= f.clone()});
            },
            SubMatrixFullMut::Detached(matr) => {
                matr.data.iter_mut().for_each(|t| {**t /= f.clone()});
            },
        }
    }
}

//==========================================================================
// Iter traits for MatrixFull
//==========================================================================
impl <'a, T> IntoIterator for &'a MatrixFull<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

/// Add by Tianyi Gao
impl<T> MatrixFull<T> 
where T: std::fmt::Debug,
{
    pub fn get_antidiag_terms(&self) -> Option<Vec<&T>> {
        //let tmp_len = self.size;
        let new_size = self.size.get(0).unwrap();
        let new_size_y = self.size.get(1).unwrap();
        if *new_size == 0 || new_size != new_size_y {
            return None
        } else if self.size[0] == self.size[1] {
           let tmp_v = 
               self.data.iter()
               .enumerate().filter(|(i,data)| {i%(new_size-1) == 0 && *i!=0 && *i!=(new_size-1)*(new_size+1)})
               .map(|(i,data)| data).collect::<Vec<&T>>();
            return Some(tmp_v)
        } else {
            return None
        }

    }

    pub fn get_sub_antidiag_terms(&self, ijsum: usize) -> Option<Vec<T>> //{
    where T: Copy + Clone {
        
        let mut return_vec: Vec<T> = vec![];
        let order = &self.size[0];

        let result = 
            if (ijsum == 0) {
                Some(vec![self.data[0]])
            } else if (ijsum > 0 && ijsum <= order-1 ) {
                let submat = MatrixFull::from_vec([ijsum+1,ijsum+1],self.iter_submatrix(0..(ijsum+1), 0..(ijsum+1)).map(|x| *x).collect::<Vec<T>>()).unwrap();
                //println!("submat = {:?}", submat);
                let return_vec = submat.get_antidiag_terms().unwrap().iter().map(|x| **x).collect::<Vec<T>>();
                //return_vec = new_vec.into_iter().map(|v| *v).collect();
                Some(return_vec)
            } else if (ijsum < (order*2 - 2) && ijsum > order-1) {
                let diff = ijsum - order + 1;
                //println!("order={}, diff = {}, new_order = {}",order,diff, 2*order-diff-1);
                let new_vec = self.iter_submatrix(diff..*order,diff..*order).map(|x| *x).collect::<Vec<T>>();
                //println!("new_vec = {:?}", new_vec);
                let submat = MatrixFull::from_vec([2*order-ijsum-1,2*order-ijsum-1],new_vec).unwrap();
                //println!("submatrix = {:?}", submat);
                let return_vec = submat.get_antidiag_terms().unwrap().iter().map(|x| **x).collect::<Vec<T>>();
                //let new_vec = submat.get_antidiag_terms().unwrap();
                //return_vec = new_vec.into_iter().map(|v| *v).collect();
                Some(return_vec)
            } else if (ijsum == (order*2 - 2)) {
                Some(vec![self.data[&self.data.len()-1]])
            } else {
                None
            };
        //new_vec.into_iter().map(|v| return_vec.push(*v));
        result
    }
}


/// More math operations for T: f64
impl MatrixFull<f64> {
    pub fn print_debug(&self, x: Range<usize>, y: Range<usize>)  {
        let length = x.len()*y.len();
        let mut tmp_s:String = format!("debug: ");
        self.iter_submatrix(x,y).for_each(|x| {
            tmp_s = format!("{},{:16.8}", tmp_s, x);
        });
        println!("{}",tmp_s);
    }

    pub fn copy_from_matr<'a, T>(&mut self, x: Range<usize>, y: Range<usize>, matr_a: &T, f_x: Range<usize>, f_y: Range<usize>) 
    where T: BasicMatrix<'a,f64>
    {
        let size = self.size.clone();
        unsafe{matr_copy(
            matr_a.data_ref().unwrap(), matr_a.size(), f_x, f_y, 
            self.data_ref_mut().unwrap(), &size,x,y 
        )}
    }

    pub fn lapack_dgemm(&mut self, a: &mut MatrixFull<f64>, b: &mut MatrixFull<f64>, opa: char, opb: char, alpha: f64, beta: f64) {
        self.to_matrixfullslicemut().lapack_dgemm(
            &mut a.to_matrixfullslice(),
            &mut b.to_matrixfullslice(), 
            opa, 
            opb, 
            alpha, 
            beta);
    }

    pub fn lapack_dgesv(&mut self, b: &mut MatrixFull<f64>, n: i32) -> MatrixFull<f64>{

        /// for x : A * x = B

        let flag = true; 
        let mut info = 0;
        let mut ipiv = vec![0 as i32; n as usize];
        if flag {
            unsafe {
                dgesv(n,
                      b.size[1] as i32,
                      &mut self.data,
                      n,
                      &mut ipiv,
                      &mut b.data,
                      n,
                      &mut info);
            }
        } else {
            panic!("Error: Inconsistency happens to perform dgesv");
        }
        let result = b.data.clone();
        MatrixFull::from_vec([n as usize, result.len()/n as usize], result).unwrap()
    }

    pub fn ddot(&self, b: &mut MatrixFull<f64>) -> Option<MatrixFull<f64>> {
        if let Some(tmp_mat) = self.to_matrixfullslice().ddot(&b.to_matrixfullslice()) {
            Some(tmp_mat)
        } else {
            None
        }
    }

    pub fn lapack_inverse(&mut self) -> Option<MatrixFull<f64>> {
        if let Some(tmp_mat) = self.to_matrixfullslicemut().lapack_inverse() {
            Some(tmp_mat)
        } else {
            None
        }
    }
    pub fn lapack_power(&mut self,p:f64, threshold: f64) -> Option<MatrixFull<f64>> {
        if let Some(tmp_mat) = self.to_matrixfullslicemut().lapack_power(p,threshold) {
            Some(tmp_mat)
        } else {
            None
        }
    }
    pub fn formated_output_e(&self, n_len: usize, mat_form: &str) {
        let mat_format = if mat_form.to_lowercase()==String::from("full") {MatFormat::Full
        } else if mat_form.to_lowercase()==String::from("upper") {MatFormat::Upper
        } else if mat_form.to_lowercase()==String::from("lower") {MatFormat::Lower
        } else {
            panic!("Error in determing the layout format of the matrix: {}", mat_form)
        };
        let n_row = self.size[0];
        let n_column = self.size[1];
        //let n_row = self.size[0];
        //let n_column = self.size[1];
        let n_block = if n_column%n_len==0 {n_column/n_len} else {n_column/n_len+1};
        let mut index:usize = 0;
        //println!("{}",n_block);
        (0..n_block).into_iter().for_each(|i_block| {
            let t_len = if (i_block+1)*n_len<=n_column {n_len} else {n_column%n_len};
            //println!("{},{}",i_block,t_len);
            let mut tmp_s:String = format!("{:5}","");
            for i in 0..t_len {
                if tmp_s.len()==5 {
                    tmp_s = format!("{} {:16}",tmp_s,i+i_block*n_len);
                } else {
                    tmp_s = format!("{},{:16}",tmp_s,i+i_block*n_len);
                }
            }
            println!("{}",tmp_s);
            for i in 0..n_row as usize {
                let mut tmp_s = format!("{:5}",i);
                let j_start = i_block*n_len;
                let mut turn_off_comma = true;
                for j in (j_start..j_start+t_len) {
                    match &mat_format {
                        MatFormat::Full => {
                            let tmp_f = self.get(&[i,j]).unwrap();
                            if tmp_s.len()==5 {
                                let tmp_fform = format!("{:+16.8e}", tmp_f);
                                tmp_s = format!("{} {}",tmp_s,c2f(&tmp_fform));
                            } else {
                                let tmp_fform = format!("{:+16.8e}", tmp_f);
                                tmp_s = format!("{},{}",tmp_s,c2f(&tmp_fform));
                            }
                        },
                        MatFormat::Upper => {
                            if i<=j {
                                let tmp_f = self.get(&[i,j]).unwrap();
                                if turn_off_comma {
                                    let tmp_fform = format!("{:+16.8e}", tmp_f);
                                    tmp_s = format!("{},{}",tmp_s,c2f(&tmp_fform));
                                    turn_off_comma = false;
                                } else {
                                    let tmp_fform = format!("{:+16.8e}", tmp_f);
                                    tmp_s = format!("{},{}",tmp_s,c2f(&tmp_fform));
                                }
                            } else {
                                tmp_s = format!("{} {:16}",tmp_s,String::from(" "));
                            }
                        },
                        MatFormat::Lower => {
                            if i>=j {
                                let tmp_f = self.get(&[i,j]).unwrap();
                                if tmp_s.len()==5 {
                                    let tmp_fform = format!("{:+16.8e}", tmp_f);
                                    tmp_s = format!("{},{}",tmp_s,c2f(&tmp_fform));
                                } else {
                                    let tmp_fform = format!("{:+16.8e}", tmp_f);
                                    tmp_s = format!("{},{}",tmp_s,c2f(&tmp_fform));
                                }
                            }
                        }
                    };
                    //println!("{},{}",tmp_i,j);
                };
                if tmp_s.len()>5 {println!("{}",tmp_s)};
            }
        });
    }

    pub fn formated_output_e_with_threshold(&self, n_len: usize, mat_form: &str, threshold: f64) {
        let mat_format = if mat_form.to_lowercase()==String::from("full") {MatFormat::Full
        } else if mat_form.to_lowercase()==String::from("upper") {MatFormat::Upper
        } else if mat_form.to_lowercase()==String::from("lower") {MatFormat::Lower
        } else {
            panic!("Error in determing the layout format of the matrix: {}", mat_form)
        };
        let n_row = self.size[0];
        let n_column = self.size[1];
        //let n_row = self.size[0];
        //let n_column = self.size[1];
        let n_block = if n_column%n_len==0 {n_column/n_len} else {n_column/n_len+1};
        let mut index:usize = 0;
        //println!("{}",n_block);
        (0..n_block).into_iter().for_each(|i_block| {
            let t_len = if (i_block+1)*n_len<=n_column {n_len} else {n_column%n_len};
            //println!("{},{}",i_block,t_len);
            let mut tmp_s:String = format!("{:5}","");
            for i in 0..t_len {
                if tmp_s.len()==5 {
                    tmp_s = format!("{} {:16}",tmp_s,i+i_block*n_len);
                } else {
                    tmp_s = format!("{},{:16}",tmp_s,i+i_block*n_len);
                }
            }
            println!("{}",tmp_s);
            for i in 0..n_row as usize {
                let mut tmp_s = format!("{:5}",i);
                let j_start = i_block*n_len;
                let mut turn_off_comma = true;
                for j in (j_start..j_start+t_len) {
                    match &mat_format {
                        MatFormat::Full => {
                            let tmp_f = if self.get(&[i,j]).unwrap().abs() > threshold {
                                self.get(&[i,j]).unwrap()
                            } else{
                                &0.0_f64
                            };
                            if tmp_s.len()==5 {
                                let tmp_fform = format!("{:+16.8e}", tmp_f);
                                tmp_s = format!("{} {}",tmp_s,c2f(&tmp_fform));
                            } else {
                                let tmp_fform = format!("{:+16.8e}", tmp_f);
                                tmp_s = format!("{},{}",tmp_s,c2f(&tmp_fform));
                            }
                        },
                        MatFormat::Upper => {
                            if i<=j {
                                let tmp_f = if self.get(&[i,j]).unwrap().abs() > threshold {
                                    self.get(&[i,j]).unwrap()
                                } else{
                                    &0.0_f64
                                };
                                if turn_off_comma {
                                    let tmp_fform = format!("{:+16.8e}", tmp_f);
                                    tmp_s = format!("{},{}",tmp_s,c2f(&tmp_fform));
                                    turn_off_comma = false;
                                } else {
                                    let tmp_fform = format!("{:+16.8e}", tmp_f);
                                    tmp_s = format!("{},{}",tmp_s,c2f(&tmp_fform));
                                }
                            } else {
                                tmp_s = format!("{} {:16}",tmp_s,String::from(" "));
                            }
                        },
                        MatFormat::Lower => {
                            if i>=j {
                                let tmp_f = if self.get(&[i,j]).unwrap().abs() > threshold {
                                    self.get(&[i,j]).unwrap()
                                } else{
                                    &0.0_f64
                                };
                                if tmp_s.len()==5 {
                                    let tmp_fform = format!("{:+16.8e}", tmp_f);
                                    tmp_s = format!("{},{}",tmp_s,c2f(&tmp_fform));
                                } else {
                                    let tmp_fform = format!("{:+16.8e}", tmp_f);
                                    tmp_s = format!("{},{}",tmp_s,c2f(&tmp_fform));
                                }
                            }
                        }
                    };
                    //println!("{},{}",tmp_i,j);
                };
                if tmp_s.len()>5 {println!("{}",tmp_s)};
            }
        });
    }
    pub fn formated_output(&self, n_len: usize, mat_form: &str) {
        let mat_format = if mat_form.to_lowercase()==String::from("full") {MatFormat::Full
        } else if mat_form.to_lowercase()==String::from("upper") {MatFormat::Upper
        } else if mat_form.to_lowercase()==String::from("lower") {MatFormat::Lower
        } else {
            panic!("Error in determing the layout format of the matrix: {}", mat_form)
        };
        let n_row = self.size[0];
        let n_column = self.size[1];
        //let n_row = self.size[0];
        //let n_column = self.size[1];
        let n_block = if n_column%n_len==0 {n_column/n_len} else {n_column/n_len+1};
        let mut index:usize = 0;
        //println!("{}",n_block);
        (0..n_block).into_iter().for_each(|i_block| {
            let t_len = if (i_block+1)*n_len<=n_column {n_len} else {n_column%n_len};
            //println!("{},{}",i_block,t_len);
            let mut tmp_s:String = format!("{:5}","");
            for i in 0..t_len {
                if tmp_s.len()==5 {
                    tmp_s = format!("{} {:12}",tmp_s,i+i_block*n_len);
                } else {
                    tmp_s = format!("{},{:12}",tmp_s,i+i_block*n_len);
                }
            }
            println!("{}",tmp_s);
            for i in 0..n_row as usize {
                let mut tmp_s = format!("{:5}",i);
                let j_start = i_block*n_len;
                let mut turn_off_comma = true;
                for j in (j_start..j_start+t_len) {
                    match &mat_format {
                        MatFormat::Full => {
                            let tmp_f = self.get(&[i,j]).unwrap();
                            if tmp_s.len()==5 {
                                tmp_s = format!("{} {:12.6}",tmp_s,tmp_f);
                            } else {
                                tmp_s = format!("{},{:12.6}",tmp_s,tmp_f);
                            }
                        },
                        MatFormat::Upper => {
                            if i<=j {
                                let tmp_f = self.get(&[i,j]).unwrap();
                                if turn_off_comma {
                                    tmp_s = format!("{} {:12.6}",tmp_s,tmp_f);
                                    turn_off_comma = false;
                                } else {
                                    tmp_s = format!("{},{:12.6}",tmp_s,tmp_f);
                                }
                            } else {
                                tmp_s = format!("{} {:12}",tmp_s,String::from(" "));
                            }
                        },
                        MatFormat::Lower => {
                            if i>=j {
                                let tmp_f = self.get(&[i,j]).unwrap();
                                if tmp_s.len()==5 {
                                    tmp_s = format!("{} {:12.6}",tmp_s,tmp_f);
                                } else {
                                    tmp_s = format!("{},{:12.6}",tmp_s,tmp_f);
                                }
                            }
                        }
                    };
                    //println!("{},{}",tmp_i,j);
                };
                if tmp_s.len()>5 {println!("{}",tmp_s)};
            }
        });
    }
    
}



fn convert_scientific_notation_to_fortran_format(n: &String) -> String {
    let re = Regex::new(r"(?P<num> *[-+]?\d.\d*)[E|e](?P<exp>-?\d{1,2})").unwrap();
    let o_len = n.len();

    if let Some(cap) = re.captures(n) {
        let main_part = cap["num"].to_string();
        let exp_part = cap["exp"].to_string();
        let exp: i32 = exp_part.parse().unwrap();
        let out_str = if exp>=0 {
            format!("{}E+{:0>2}",main_part,exp)
        } else {

            format!("{}E-{:0>2}",main_part,exp.abs())
        };
        let n_len = out_str.len();
        return out_str[n_len-o_len..n_len].to_string()
    } else {
        panic!("Error: the input string is not a standard scientific notation")
    }

}

fn c2f(n: &String) -> String {
    convert_scientific_notation_to_fortran_format(n)
}