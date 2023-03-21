//#![warn(missing_docs)]
use std::{fmt::Display, collections::binary_heap::Iter, iter::{Filter,Flatten, Map}, convert, slice::{ChunksExact,ChunksExactMut, self}, mem::ManuallyDrop, marker, cell::RefCell, ops::{IndexMut, RangeFull, MulAssign, DivAssign, Div, DerefMut, Deref}, thread::panicking};
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, Index, Range};
use libc::{CLOSE_RANGE_CLOEXEC, SYS_userfaultfd};
use typenum::{U2, Pow};
use rayon::{prelude::*, collections::btree_map::IterMut, iter::Enumerate};
use std::vec::IntoIter;

use crate::{index::{TensorIndex, TensorIndexUncheck}, TensorOpt, TensorOptMut, TensorSlice, TensorSliceMut, TensorOptUncheck, TensorSliceUncheck, TensorSliceMutUncheck, TensorOptMutUncheck};
use crate::matrix;

pub mod matrixfull;
pub mod matrixfullslice;
pub mod matrixupper;
pub mod submatrixfull;
pub mod matrixconst;
pub mod matrix_blas_lapack;


use crate::matrix::matrixfull::*;
use crate::matrix::matrixconst::*;
use crate::matrix::matrixfullslice::*;
use crate::matrix::matrixupper::*;
use crate::matrix::submatrixfull::*;
use crate::matrix::matrix_blas_lapack::*;


/// **MatrixFull** is a `column-major` 2D array designed for quantum chemistry calculations.
/// 
///  #### Basic Usage for General-purpose Use
/// 
/// - [Construction](#construction)
/// 
/// - [Indexing](#indexing)
/// 
/// - [Mathatics](#math-operations)
/// 
/// - [Iterators](#iterators)
/// 
/// - [Slicing](#slicing)
/// 
///  #### Usage for Advanced and/or Specific Uses
/// 
/// - [Mathmatic operations <span style="float:right;"> `MathMatrix::scaled_add`...</span>](#more-math-operations-for-the-rest-package)
/// 
/// - [Matrix operations <span style="float:right;"> `MatrixFull::transpose`...</span>](#more-matrix-operations-needed-by-the-rest-package)  
/// 
/// - Also provides wrappers to lapack and blas functions, including:<span style="float:right;"> [`matrix::matrix_blas_lapack`]</span>
/// 1) perform the matrix-matrix operation for C = alpha\*op( A )\*op( B ) + beta\*C: <span style="float:right;"> [`_dgemm`]</span>  
/// 2) compute the eigenvalues and, optionally, eigenvectors: <span style="float:right;"> [`_dsyev`]</span>  
/// 3) compute the Cholesky factorization of a real symmetric positive definite matrix A:<span style="float:right;"> [`_dpotrf`]</span>  
/// 4) many others ...  
/// **NOTE**:: all functions in lapack and blas libraries can be imported in the similar way.
/// 
/// 
/// # Construction 
///   There are several ways to construct a matrix from different sources
/// 
///   1. Create a new matrix filled with a given element.
/// ```
///   use rest_tensors::MatrixFull;
///   let matr = MatrixFull::new([3,4],1.0f64);
///   //| 1.0 | 1.0 | 1.0 | 1.0 |
///   //| 1.0 | 1.0 | 1.0 | 1.0 |
///   //| 1.0 | 1.0 | 1.0 | 1.0 |
/// ```
///   2. Cenerate a new matrix from a vector. For example, a 3x4 matrix from a vector with 12 elements
/// ```
///   use rest_tensors::MatrixFull;
///   let new_vec = vec![
///          1.0,  2.0,  3.0, 
///          4.0,  5.0,  6.0, 
///          7.0,  8.0,  9.0, 
///         10.0, 11.0, 12.0];
///   let matr = MatrixFull::from_vec([3,4],new_vec).unwrap();
///   assert_eq!(matr[(1,2)],8.0)
///   //| 1.0 | 4.0 | 7.0 |10.0 |
///   //| 2.0 | 5.0 | 8.0 |11.0 |
///   //| 3.0 | 6.0 | 9.0 |12.0 |
/// ```
/// # Indexing
///   The [`MatrixFull`](MatrixFull) struct allows to access values by index, based on the [`Index`](Index) trait.
///   1. For any matrix element, it is accessable via `[usize;2]` or `(usize, usize)` in the order of `row and column`
/// ```
///   use rest_tensors::MatrixFull;
///   let mut matr = MatrixFull::new([2,2],0.0);
///   matr[[0,0]] = 1.0;
///   matr[(1,1)] = 1.0;
///   assert_eq!(matr, MatrixFull::from_vec([2,2],vec![
///      1.0,0.0,
///      0.0,1.0]).unwrap());
/// ```
///   2. It is also accessable via `get2d(_mut)` and `set2d` in the traits of [`TensorOpt`](TensorOpt) and/or [`TensorOptMut`](TensorOptMut)
/// ```
///   use rest_tensors::MatrixFull;
///   use rest_tensors::TensorOptMut;
///   let mut matr = MatrixFull::new([2,2],0.0);
///   let mut mat00 = matr.get2d_mut([0,0]).unwrap();
///   *mat00 = 1.0;
///   matr.set2d([1,1],1.0);
///   assert_eq!(matr, MatrixFull::from_vec([2,2],vec![
///      1.0,0.0,
///      0.0,1.0]).unwrap());
/// ```
///   3. For all or part of elements in the a given column, they are accessable via `(Range<usize>,usize)` or `(RangeFull, usize)`, and return as a **slice**
/// ```
///   use rest_tensors::MatrixFull;
///   let new_vec = vec![
///          1.0,  2.0,  3.0, 
///          4.0,  5.0,  6.0, 
///          7.0,  8.0,  9.0, 
///         10.0, 11.0, 12.0];
///   let mut matr = MatrixFull::from_vec([3,4],new_vec).unwrap();
///   let mut part_column_2nd = &mut matr[(0..2,2)];
///   assert_eq!(part_column_2nd, &[7.0,8.0]);
///   let mut full_column_2nd = &mut matr[(..,2)];
///   assert_eq!(full_column_2nd, &[7.0,8.0,9.0]);
///   //             _______
///   //| 1.0 | 4.0 || 7.0 ||10.0 |
///   //| 2.0 | 5.0 || 8.0 ||11.0 |
///   //| 3.0 | 6.0 || 9.0 ||12.0 |
///   //             -------
/// ```
///   4. For the elements in several continued columns, they are accessable via `(RangeFull, Range<usize>)`, and return as a **slice**
/// ```
///   use rest_tensors::MatrixFull;
///   let new_vec = vec![
///          1.0,  2.0,  3.0, 
///          4.0,  5.0,  6.0, 
///          7.0,  8.0,  9.0, 
///         10.0, 11.0, 12.0];
///   let mut matr = MatrixFull::from_vec([3,4],new_vec).unwrap();
///   let mut columns23 = &mut matr[(..,1..3)];
///   assert_eq!(columns23, &[4.0,5.0,6.0,7.0,8.0,9.0]);
///   //       _____________
///   //| 1.0 || 4.0 | 7.0 ||10.0 |
///   //| 2.0 || 5.0 | 8.0 ||11.0 |
///   //| 3.0 || 6.0 | 9.0 ||12.0 |
///   //       -------------
/// ```
///   5. In general, a sub matrix in the area of `(Range<usize>, Range<usize>)` is accessable via  `get_submatrix()` and `get_submatrix_mut()`, 
///      and return as a [`SubMatrixFull<T>`] and [`SubMatrixFullMut<T>`]
/// ```
///   use rest_tensors::MatrixFull;
///   use rest_tensors::SubMatrixFull;
///   let new_vec = vec![
///          1.0,  2.0,  3.0, 
///          4.0,  5.0,  6.0, 
///          7.0,  8.0,  9.0, 
///         10.0, 11.0, 12.0];
///   let mut matr = MatrixFull::from_vec([3,4],new_vec).unwrap();
///   let mut sub_matr = matr.get_submatrix(0..2,2..4);
///   assert_eq!(sub_matr.data(), vec![7.0,8.0,10.0,11.0]);
///   //             _____________ 
///   //| 1.0 | 4.0 || 7.0 |10.0 ||
///   //| 2.0 | 5.0 || 8.0 |11.0 ||
///   //              -----------  
///   //| 3.0 | 6.0 |  9.0  |12.0 |
/// ```
/// # Math Operations
///   The [`MatrixFull`](MatrixFull) struct enables the basic mathmatic operations, including `+`, `+=`, `-`, `-=`, `*`, `*=`, `/`, and `/=`  
///   based on the traits of [`Add`](Add), [`AddAssign`](AddAssign), [`Sub`](Sub), [`SubAssign`](SubAssign), 
///   [`Mul`](Mul), [`MulAssign`](MulAssign), [`Div`](Div), [`DivAssign`](DivAssign), respectively
///   1. Add or subtract for two matrices: `MatrixFull<T>` +/- `MatrixFull<T>`. 
///   **NOTE**: 1) The size of two matrices should be the same. Otherwise, the program stops with **panic!**
/// 
/// ```
///   use rest_tensors::MatrixFull;
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
///   // NOTE: both matr_a and matr_b are consumed after `+` and `-` operations, 
///   let mut matr_c = matr_a.clone() + matr_b.clone();
///   //          | 14.0 | 20.0 | 26.0 | 32.0 |
///   //matr_c =  | 16.0 | 22.0 | 28.0 | 34.0 |
///   //          | 18.0 | 24.0 | 30.0 | 36.0 |
///   assert_eq!(matr_c[(..,3)], [32.0,34.0,36.0]);
/// 
///   // matr_c = matr_c - matr_b = matr_a
///   // NOTE: matr_b is consumed after `+` and `-` operations, 
///   matr_c -= matr_b;
///   assert_eq!(matr_c, matr_a);
/// ```
///   2. Add or subtract between two matrices with different types: `MatrixFull<T> +/- (Sub)MatrixFull<T>`
///      NOTE: the matrix: `MatrixFull<&T>` should be used after the operators of ‘+’ and ‘-’.
/// ```
///   use rest_tensors::MatrixFull;
///   let vec_a = vec![
///          1.0,  2.0,  3.0, 
///          4.0,  5.0,  6.0, 
///          7.0,  8.0,  9.0, 
///         10.0, 11.0, 12.0];
///   let matr_a = MatrixFull::from_vec([3,4],vec_a).unwrap();
///   //          |  1.0 |  4.0 |  7.0 | 10.0 |
///   //matr_a =  |  2.0 |  5.0 |  8.0 | 11.0 |  with the type of MatrixFull<f64>   
///   //          |  3.0 |  6.0 |  9.0 | 12.0 |
/// 
///   let matr_b = matr_a.get_submatrix(0..2,1..3);
///   //matr_b =  |  4.0 |  7.0 |       
///   //          |  5.0 |  8.0 | with the type of SubMatrixFull<f64>(MatrixFull<&f64>)
/// 
///   let vec_c = (5..9).map(|x| x as f64).collect::<Vec<f64>>();
///   let matr_c = MatrixFull::from_vec([2,2],vec_c).unwrap();
///   //matr_c =  |  5.0 |  7.0 |       
///   //          |  6.0 |  8.0 | with the type of MatrixFull<f64>
/// 
///   // matr_d = matr_b: `SubMatrixFull<f64>` + matr_c: `MatrixFull<f64>`;  
///   // NOTE: both matr_c and matr_b are dropped after the add operation.
///   let mut matr_d = matr_b + matr_c.clone();
///   //matr_d =  |  9.0 | 14.0 |       
///   //          | 11.0 | 16.0 | with the type of MatrixFull<f64>
///   assert_eq!(matr_d.data(), vec![9.0,11.0,14.0,16.0]);
/// 
///   // matr_d: `MatrixFull<f64>` -= matr_c: `SubMatrixFull<f64>` = matr_c;  
///   // NOTE: both matr_c and matr_b are dropped after the add operation.
///   let matr_b = matr_a.get_submatrix(0..2,1..3);
///   //matr_b =  |  4.0 |  7.0 |       
///   //          |  5.0 |  8.0 | with the type of SubMatrixFull<&f64>
///   matr_d -= matr_b;
///   assert_eq!(matr_d, matr_c)
/// ```
///   3. Enable `SubMatrixFullMut<T>` the operations of '+=' and '-=' with `(Sub)MatrixFull<T>`
/// 
/// ```
///   use rest_tensors::MatrixFull;
///   let vec_a = vec![
///          1.0,  2.0,  3.0, 
///          4.0,  5.0,  6.0, 
///          7.0,  8.0,  9.0, 
///         10.0, 11.0, 12.0];
///   let mut matr_a = MatrixFull::from_vec([3,4],vec_a).unwrap();
///   //          |  1.0 |  4.0 |  7.0 | 10.0 |
///   //matr_a =  |  2.0 |  5.0 |  8.0 | 11.0 |  with the type of MatrixFull<f64>   
///   //          |  3.0 |  6.0 |  9.0 | 12.0 |
/// 
///   let mut matr_b = matr_a.get_submatrix_mut(0..2,1..3);
///   //matr_b =  |  4.0 |  7.0 |       
///   //          |  5.0 |  8.0 | with the type of MatrixFull<&f64>
/// 
///   let vec_c = (5..9).map(|x| x as f64).collect::<Vec<f64>>();
///   let matr_c = MatrixFull::from_vec([2,2],vec_c).unwrap();
///   //matr_c =  |  5.0 |  7.0 |       
///   //          |  6.0 |  8.0 | with the type of MatrixFull<f64>
/// 
///   // matr_a[(0..2, 1..3)] += matr_c
///   matr_b += matr_c;
///   //          |  1.0 |  9.0 | 14.0 | 10.0 |
///   //matr_a =  |  2.0 | 11.0 | 16.0 | 11.0 |  with the type of MatrixFull<f64>   
///   //          |  3.0 |  6.0 |  9.0 | 12.0 |
///   assert_eq!(matr_a.get_submatrix(0..2,1..3).data(), vec![9.0,11.0,14.0,16.0]);
/// ```
/// 
///   4. Enable `(Sub)MatrixFull<T>` +/- `<T>`, and `(Sub)MatrixFull(Mut)<T>` +=/-= `<T>`
/// ```
///   use rest_tensors::MatrixFull;
///   let vec_a = vec![
///          1.0,  2.0,  3.0, 
///          4.0,  5.0,  6.0, 
///          7.0,  8.0,  9.0, 
///         10.0, 11.0, 12.0];
///   let mut matr_a = MatrixFull::from_vec([3,4],vec_a).unwrap();
///   // matr_b = matr_a + 2.0
///   let mut matr_b = matr_a.clone() + 2.0;
///   assert_eq!(matr_b[(..,0)], [3.0, 4.0, 5.0]);
///   // matr_b = matr_b - 2.0 = matr_a
///   matr_b -= 2.0;
///   assert_eq!(matr_b, matr_a);
/// 
///   let mut matr_b = matr_a.get_submatrix_mut(0..2,1..3);
///   //matr_b =  |  4.0 |  7.0 |       
///   //          |  5.0 |  8.0 | with the type of MatrixFull<&f64>
///   matr_b += 2.0;
///   //          |  1.0 |  6.0 |  9.0 | 10.0 |
///   //matr_a =  |  2.0 |  7.0 | 10.0 | 11.0 |  with the type of MatrixFull<f64>   
///   //          |  3.0 |  6.0 |  9.0 | 12.0 |
///   assert_eq!(matr_a.get_submatrix(0..2,1..3).data(), vec![6.0,7.0,9.0,10.0]);
/// ```
///   5. Enable `(Sub)MatrixFull<T>` *(/) `<T>`, and `(Sub)MatrixFull(Mut)<T>` +=(/=) `<T>`
/// ```
///   use rest_tensors::MatrixFull;
///   let vec_a = vec![
///          1.0,  2.0,  3.0, 
///          4.0,  5.0,  6.0, 
///          7.0,  8.0,  9.0, 
///         10.0, 11.0, 12.0];
///   let matr_a = MatrixFull::from_vec([3,4],vec_a).unwrap();
///   // matr_b = matr_a * 2.0  
///   // NOTE: 2.0 should be located after the operator '*' and '/'
///   let mut matr_b = matr_a.clone() * 2.0;
///   assert_eq!(matr_b[(..,0)], [2.0, 4.0, 6.0]);
///   // matr_b = matr_b / 2.0 = matr_a
///   matr_b /= 2.0;
///   assert_eq!(matr_b, matr_a);
/// 
///   //          |  1.0 |  4.0 |  7.0 | 10.0 |
///   //matr_b =  |  2.0 |  5.0 |  8.0 | 11.0 |  with the type of MatrixFull<f64>   
///   //          |  3.0 |  6.0 |  9.0 | 12.0 |
///   let mut matr_c = matr_b.get_submatrix_mut(0..2,1..3);
///   matr_c *= 2.0;
///   // after the multiply operation
///   //          |  1.0 |  8.0 | 14.0 | 10.0 |
///   //matr_b =  |  2.0 | 10.0 | 16.0 | 11.0 |  with the type of MatrixFull<f64>   
///   //          |  3.0 |  6.0 |  9.0 | 12.0 |
///   assert_eq!(matr_b.get_submatrix(0..2,1..3).data(), vec![8.0,10.0,14.0,16.0]);
/// 
/// ```
/// # Iterators
///   1. The [`MatrixFull`](MatrixFull) struct implements the standard iterators of `iter()`, `iter_mut()`, and `into_iter()`,
///       which are nothing but the wrappers to the iterators of `MatrixFull<T>.data: Vec<T>`
/// ```
///    use rest_tensors::MatrixFull;
///    let mut matr_a = MatrixFull::from_vec(
///      [3,4],
///      (1..13).collect::<Vec<i32>>()
///    ).unwrap();
/// 
///    let mut vec_a = (1..13).collect::<Vec<i32>>();
///    matr_a.into_iter().zip(vec_a.iter()).for_each(|(m_item, v_item)| {
///        assert_eq!(m_item, v_item);
///    });
///    matr_a.iter_mut().zip(vec_a.iter()).for_each(|(m_item, v_item)| {
///        assert_eq!(m_item, v_item)
///    });
/// ```
///  As a column-major 2-dimention tensor, [`MatrixFull`](MatrixFull) also provides special iterators with
///  respect to rows and/or columns:
/// 
///  2. `iter_column(j)` and `iter_column_mut(j)` provides the standard iterators for 
///     the immutable and mutable elements, respectively, in the `j`th column.
/// ```
///   use rest_tensors::MatrixFull;
///   let matr_a = MatrixFull::from_vec(
///     [3,4],
///     (1..13).collect::<Vec<i32>>()
///   ).unwrap();
///   //                _____
///   //          |  1 || 4 ||  7 | 10 |
///   //matr_a =  |  2 || 5 ||  8 | 11 |  with the type of MatrixFull<i32>
///   //          |  3 || 6 ||  9 | 12 |
///   //                -----
///   let column_j = MatrixFull::from_vec([3,1],vec![4,5,6]).unwrap();
///   matr_a.iter_column(1).zip(column_j.iter()).for_each(|(m_item, c_item)| {
///        assert_eq!(m_item, c_item)
///   })
/// ```
///  3. `iter_columns(Range<usize>)` and `iter_columns_mut(Range<usize>)` provides the chunck iterators
///      for a set of columns within `Range<usize>`. The iteratior gives the elements of different columns
///      chunck by chunck.
/// 
/// `iter_columns_full()` and `iter_columns_full_mut()` are the specific cases, which iterate over all columns
/// ```
///   use rest_tensors::MatrixFull;
///   let matr_a = MatrixFull::from_vec(
///     [3,4],
///     (1..13).collect::<Vec<i32>>()
///   ).unwrap();
///   //                __________
///   //          |  1 || 4 |  7 || 10 |
///   //matr_a =  |  2 || 5 |  8 || 11 |  with the type of MatrixFull<i32>
///   //          |  3 || 6 |  9 || 12 |
///   //                ----------
///   let columns = vec![[4,5,6],[7,8,9]];
///   matr_a.iter_columns(1..3).zip(columns.iter()).for_each(|(m_item, c_item)| {
///        assert_eq!(m_item, c_item)
///   })
/// ```
///   4. `iter_submatrix()` and `iter_submatrix_mut()` provide home-made StepBy iterators in the column-major order for 
///     the immutable and mutable elements in the sub-matrix, respectively.
/// 
/// ```
///   use rest_tensors::MatrixFull;
///   let matr_a = MatrixFull::from_vec(
///     [3,4],
///     (1..13).collect::<Vec<i32>>()
///   ).unwrap();
///   //                __________
///   //          |  1 || 4 |  7 || 10 |
///   //matr_a =  |  2 || 5 |  8 || 11 |  with the type of MatrixFull<i32>
///   //                ----------
///   //          |  3 |  6 |  9  | 12 |
///   let smatr_a = MatrixFull::from_vec([2,2],vec![4,5,7,8]).unwrap();
///   matr_a.iter_submatrix(0..2,1..3).zip(smatr_a.iter()).for_each(|(m_item, sm_item)| {
///        assert_eq!(m_item, sm_item)
///   })
/// ```
///   5. Based on `iter_submatrix()` and `iter_submatrix_mut()`, [`MatrixFull`] provides flatten iterators for rows, i.e.
///   `iter_row()`, `iter_row_mut()`, `iter_rows()`, `iter_rows_mut()`
/// ```
///   use rest_tensors::MatrixFull;
///   let matr_a = MatrixFull::from_vec(
///     [3,4],
///     (1..13).collect::<Vec<i32>>()
///   ).unwrap();
///   //          |  1 |  4 |  7 | 10 |
///   //          _____________________
///   //matr_a =  |  2 |  5 |  8 | 11 |  with the type of MatrixFull<i32>
///   //          ------ --------------
///   //          |  3 |  6 |  9 | 12 |
///   //
///   let row_1 = vec![&2,&5,&8,&11];
///   let from_iter_row =  matr_a.iter_row(1).collect::<Vec<&i32>>();
///   assert_eq!(row_1, from_iter_row);
/// ```
///  6. Iterate the diagonal terms using `iter_diagonal().unwrap()` and `iter_diagonal_mut().unwrap()`
/// ```
///   use rest_tensors::MatrixFull;
///   let matr_a = MatrixFull::from_vec(
///     [4,4],
///     (1..17).collect::<Vec<i32>>()
///   ).unwrap();
///   //          |  1 |  5 |  9 | 13 |
///   //matr_a =  |  2 |  6 | 10 | 14 |  with the type of MatrixFull<i32>
///   //          |  3 |  7 | 11 | 15 |
///   //          |  4 |  8 | 12 | 16 |
///   //
///   let diagonal = vec![&1,&6,&11,&16];
///   let from_diagonal_iter = matr_a.iter_diagonal().unwrap().collect::<Vec<&i32>>();
///   assert_eq!(from_diagonal_iter, diagonal);
/// ```
///   7. Iterate the upper part of the matrix using `iter_matrixupper().unwrap()` and `iter_matrixupper_mut().unwrap()`
/// ```
///   use rest_tensors::MatrixFull;
///   let matr_a = MatrixFull::from_vec(
///     [4,4],
///     (1..17).collect::<Vec<i32>>()
///   ).unwrap();
///   //          |  1 |  5 |  9 | 13 |
///   //matr_a =  |  2 |  6 | 10 | 14 |  with the type of MatrixFull<i32>
///   //          |  3 |  7 | 11 | 15 |
///   //          |  4 |  8 | 12 | 16 |
///   //
///   let upper = vec![&1,&5,&6,&9,&10,&11,&13,&14,&15,&16];
///   let from_upper_iter = matr_a.iter_matrixupper().unwrap().collect::<Vec<&i32>>();
///   assert_eq!(from_upper_iter, upper);
/// ```
/// # Slicing
///    The [`MatrixFull<T>`](MatrixFull) struct provides the tools to slice the data for a given column or a set of continued columns:
///    `slice_column(j:usize)->&[T]`, `slice_column_mut(j:usize)->&mut[T]`, 
///     and `slice_columns(j:Range<usize>)->&[T]`, `slice_columns_mut(j:Range<usize>)->&mut[T]`
/// ```
///   use rest_tensors::MatrixFull;
///   let matr_a = MatrixFull::from_vec(
///     [3,4],
///     (1..13).collect::<Vec<i32>>()
///   ).unwrap();
///   //          |  1 |  4 |  7 | 10 |
///   //matr_a =  |  2 |  5 |  8 | 11 |  with the type of MatrixFull<i32>
///   //          |  3 |  6 |  9 | 12 |
/// 
///   let column_1 = matr_a.slice_column(1);
///   assert_eq!(column_1, &[4,5,6]);
///   let column_12 = matr_a.slice_columns(1..3);
///   assert_eq!(column_12, &[4,5,6,7,8,9]);
/// ```
#[derive(Clone,Debug, PartialEq)]
pub struct MatrixFull<T> {
    /// the number of row and column, column major
    pub size : [usize;2],
    /// indicing is defined to facilitate the element nevigation, in particular for 3-rank tensors, RIFull
    pub indicing: [usize;2],
    /// the data stored in the [`Vec`](Vec) struct
    pub data : Vec<T>,
}

#[derive(Clone, Copy,Debug, PartialEq)]
// MatFormat: used for matrix printing
pub enum MatFormat {
    Full,
    Upper,
    Lower
}
pub trait BasicMatrix<'a, T> {

    fn size(&self) -> &[usize];

    fn indicing(&self) -> &[usize];

    fn is_matr(&self) -> bool {self.size().len() == 2 && self.indicing().len() == 2}

    /// by default, the matrix should be contiguous, unless specify explicitly.
    fn is_contiguous(&self) -> bool {true}

    fn data_ref(&self) -> Option<&[T]>; 

    fn data_ref_mut(&mut self) -> Option<&mut [T]>; 

}

pub trait BasicMatrixOpt<'a, T> where Self: BasicMatrix<'a, T>, T: Copy + Clone {
    fn to_matrixfull(&self) -> Option<MatrixFull<T>> 
    where T: Copy + Clone {
        if let Some(data) = self.data_ref() {
            return Some(MatrixFull {
                size: [self.size()[0],self.size()[1]],
                indicing: [self.indicing()[0],self.indicing()[1]],
                data: data.iter().map(|x| *x).collect::<Vec<T>>(),
            })
        } else {
            None
        }
    }
}


pub fn basic_check_shape(size_a: &[usize], size_b: &[usize]) -> bool {
    size_a.iter().zip(size_b.iter()).fold(true, |check,size| {
        check && size.0==size.1
    })
}

pub fn check_shape<'a, Q,P,T>(matr_a:&'a Q, matr_b: &'a P) -> bool 
where Q: BasicMatrix<'a, T>,
      P: BasicMatrix<'a, T>
{
    matr_a.size().iter().zip(matr_b.size().iter()).fold(true, |check,size| {
        check && size.0==size.1
    })
}

pub fn general_check_shape<'a, Q,P,T>(matr_a:&'a Q, matr_b: &'a P,opa: char, opb: char) -> bool 
where Q: BasicMatrix<'a, T>,
      P: BasicMatrix<'a, T>
{
    crate::matrix::matrix_blas_lapack::general_check_shape(matr_a, matr_b, opa, opb)
}

pub trait MathMatrix<'a, T> where Self: BasicMatrix<'a, T> + BasicMatrixOpt<'a, T>, T: Copy + Clone {
    fn add<Q>(&'a self, other: &'a Q) -> Option<MatrixFull<T>> 
    where T: Add<Output=T> + AddAssign,
          Q: BasicMatrix<'a, T>, Self: Sized
    {
        if check_shape(self, other) {
            let mut new_tensors = self.to_matrixfull();
            if let Some(out_tensors) = &mut new_tensors {
                out_tensors.data_ref_mut().unwrap().iter_mut()
                    .zip(other.data_ref().unwrap().iter()).for_each(|(t,f)| {*t += *f});
                new_tensors
            } else {None}
        } else {
            None
        }
    }
    fn scaled_add<Q>(&'a self, other: &'a Q,scale_factor: T) -> Option<MatrixFull<T>> 
    where T: Add + AddAssign + Mul<Output=T>,
          Q: BasicMatrix<'a, T>, Self: Sized
    {
        if check_shape(self, other) {
            let mut new_tensors = self.to_matrixfull();
            if let Some(out_tensors) = &mut new_tensors {
                out_tensors.data_ref_mut().unwrap().iter_mut()
                    .zip(other.data_ref().unwrap().iter()).for_each(|(t,f)| {*t += scale_factor * (*f)});
                new_tensors
            } else {None}
        } else {
            None
        }
    }
    /// For A += B
    fn self_add<Q>(&'a mut self, bm: &'a Q) 
    where T: Add + AddAssign,
          Q: BasicMatrix<'a, T>, Self:Sized
    {
        let size_a = [self.size()[0], self.size()[1]];
        /// A = A + B 
        if ! basic_check_shape(&size_a, bm.size()) {
            panic!("Error: Shape inconsistency happens when plus two matrices");
        }
        self.data_ref_mut().unwrap().iter_mut()
            .zip(bm.data_ref().unwrap().iter())
            .for_each(|(c,p)| {*c += *p});
    }
    /// For A += c*B where c is a scale factor
    fn self_scaled_add<Q>(&'a mut self, bm: &'a Q, b: T) 
    where T: Add + AddAssign + Mul<Output=T>,
          Q: BasicMatrix<'a, T>, Self:Sized
    {
        let size_a = [self.size()[0], self.size()[1]];
        if basic_check_shape(&size_a, bm.size()) {
            self.data_ref_mut().unwrap().iter_mut()
                .zip(bm.data_ref().unwrap().iter())
                .for_each(|(c,p)| {*c +=*p*b});
        } else {
            panic!("Error: Shape inconsistency happens when plus two matrices");
        }
    }
    /// For a*A + b*B -> A
    fn self_general_add<Q>(&'a mut self, bm: &'a Q,a: T, b:T) 
    where T: Add<Output=T> + AddAssign + Mul<Output=T>,
          Q: BasicMatrix<'a, T>, Self:Sized
    {
        let size_a = [self.size()[0], self.size()[1]];
        if basic_check_shape(&size_a, bm.size()) {
            //let mut new_tensors: MatrixFull<f64> = self.clone();
            self.data_ref_mut().unwrap().iter_mut().zip(bm.data_ref().unwrap().iter()).for_each(|(c,p)| {*c =*c*a+(*p)*b});
        } else {
            panic!("Error: Shape inconsistency happens when plus two matrices");
        }
    }
    /// For A - B -> C
    fn sub<Q>(&'a self, other: &'a Q) -> Option<MatrixFull<T>> 
    where T: Sub + SubAssign, Q: BasicMatrix<'a, T>, Self: Sized
    {
        if check_shape(self,other) {
            let mut new_tensors = self.to_matrixfull();
            if let Some(out_tensors) = &mut new_tensors {
                out_tensors.data_ref_mut().unwrap().iter_mut().zip(other.data_ref().unwrap().iter()).for_each(|(c,p)| {*c -= *p});
                new_tensors
            } else {None}
        } else {None}
    }
    /// For A - B -> A
    fn self_sub<Q>(&'a mut self, bm: &'a Q) 
    where T: Sub + SubAssign, Q: BasicMatrix<'a, T>, Self: Sized
    {
        let size_a = [self.size()[0], self.size()[1]];
        if basic_check_shape(&size_a, bm.size()) {
            self.data_ref_mut().unwrap().iter_mut().zip(bm.data_ref().unwrap().iter()).for_each(|(c,p)| {*c -= *p});
        } else {
            panic!("Error: Shape inconsistency happens when subtract two matrices");
        }
    }
    /// For a*A -> A
    #[inline]
    fn self_multiple(&mut self, a: T) 
    where T: Mul<Output = T> + MulAssign,
    {
        self.data_ref_mut().unwrap().iter_mut().for_each(|c| {*c *= a});
    }

}

pub trait ParMathMatrix<'a, T> 
where Self: Sized + BasicMatrix<'a, T> + BasicMatrixOpt<'a, T>, 
      T: Copy + Clone + Send + Sync 
{
    /// Parallel version for C = A + B,
    fn par_add<Q>(&'a self, other: &'a Q) -> Option<MatrixFull<T>> 
    where T: Add<Output=T> + AddAssign,
          Q: BasicMatrix<'a, T>, Self: Sized
    {
        if check_shape(self, other) {
            let mut new_tensors = self.to_matrixfull();
            if let Some(out_tensors) = &mut new_tensors {
                out_tensors.data_ref_mut().unwrap().par_iter_mut()
                    .zip(other.data_ref().unwrap().par_iter()).for_each(|(t,f)| {*t += *f});
                new_tensors
            } else {None}
        } else {
            None
        }
    }
    /// Parallel version for C = A + c*B,
    fn par_scaled_add<Q>(&'a self, other: &'a Q, fac: T) -> Option<MatrixFull<T>> 
    where T: Add<Output=T> + AddAssign + Mul<Output=T> + MulAssign,
          Q: BasicMatrix<'a, T>, Self: Sized
    {
        if check_shape(self, other) {
            let mut new_tensors = self.to_matrixfull();
            if let Some(out_tensors) = &mut new_tensors {
                out_tensors.data_ref_mut().unwrap().par_iter_mut()
                    .zip(other.data_ref().unwrap().par_iter()).for_each(|(t,f)| {*t += *f*fac});
                new_tensors
            } else {None}
        } else {
            None
        }
    }
    /// Parallel version for A + B -> A
    fn par_self_add<Q>(&mut self, bm: &'a Q) 
    where T: Add + AddAssign,
          Q: BasicMatrix<'a, T>,
          Self: Sized 
    {
        let size_a = [self.size()[0], self.size()[1]];
        if basic_check_shape(&size_a, bm.size()) {
            self.data_ref_mut().unwrap().par_iter_mut().zip(bm.data_ref().unwrap().par_iter())
                .for_each(|(c,p)| {*c += *p});
        } else {
            panic!("Error: Shape inconsistency happens when plus two matrices");
        }
    }
    /// Parallel version for A + b*B -> A
    fn par_self_scaled_add<Q>(&mut self, bm: &'a Q, b: T) 
    where T: Add + AddAssign + Mul<Output=T> + MulAssign,
          Q: BasicMatrix<'a, T>
    {
        let size_a = [self.size()[0], self.size()[1]];
        if basic_check_shape(&size_a, bm.size()) {
            self.data_ref_mut().unwrap().par_iter_mut().zip(bm.data_ref().unwrap().par_iter()).for_each(|(c,p)| {*c += *p*b});
        } else {
            panic!("Error: Shape inconsistency happens when plus two matrices");
        }
    }
    /// Parallel version for a*A + b*B -> A
    fn par_self_general_add<Q>(&'a mut self, bm: &'a Q,a: T, b:T) 
    where T: Add<Output=T> + AddAssign + Mul<Output=T>,
          Q: BasicMatrix<'a, T>
    {
        let size_a = [self.size()[0], self.size()[1]];
        if basic_check_shape(&size_a, bm.size()) {
            //let mut new_tensors: MatrixFull<f64> = self.clone();
            self.data_ref_mut().unwrap().par_iter_mut().zip(bm.data_ref().unwrap().par_iter()).for_each(|(c,p)| {*c =*c*a+(*p)*b});
        } else {
            panic!("Error: Shape inconsistency happens when plus two matrices");
        }
    }
    /// Parallel version for A - B -> C
    fn par_sub<Q>(&'a self, other: &'a Q) -> Option<MatrixFull<T>> 
    where T: Sub + SubAssign, Q: BasicMatrix<'a, T>
    {
        if check_shape(self,other) {
            let mut new_tensors = self.to_matrixfull();
            if let Some(out_tensors) = &mut new_tensors {
                out_tensors.data_ref_mut().unwrap().par_iter_mut().zip(other.data_ref().unwrap().par_iter()).for_each(|(c,p)| {*c -= *p});
                new_tensors
            } else {None}
        } else {None}
    }
    /// Parallel version for A - B -> A
    fn par_self_sub<Q>(&mut self, bm: &'a Q) 
    where T: Sub + SubAssign,
          Q: BasicMatrix<'a, T>
    {
        let size_a = [self.size()[0], self.size()[1]];
        if basic_check_shape(&size_a, bm.size()) {
            self.data_ref_mut().unwrap().par_iter_mut().zip(bm.data_ref().unwrap().par_iter()).for_each(|(c,p)| {*c -= *p});
        } else {
            panic!("Error: Shape inconsistency happens when plus two matrices");
        }
    }
    /// Parallel version for a*A -> A
    fn par_self_multiple(&mut self, a: T) 
    where T: Mul<Output=T> + MulAssign
    {
        self.data_ref_mut().unwrap().par_iter_mut().for_each(|c| {*c *= a});
    }

}
