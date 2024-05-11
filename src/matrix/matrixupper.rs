//#![warn(missing_docs)]
use std::{fmt::Display, collections::binary_heap::Iter, iter::{Filter,Flatten, Map}, convert, slice::{ChunksExact,ChunksExactMut, self}, mem::ManuallyDrop, marker, cell::RefCell, ops::{IndexMut, RangeFull, MulAssign, DivAssign, Div, DerefMut, Deref}, thread::panicking};
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, Index, Range};
use libc::{CLOSE_RANGE_CLOEXEC, SYS_userfaultfd};
use typenum::{U2, Pow};
use rayon::{prelude::*, collections::btree_map::IterMut, iter::Enumerate};
use std::vec::IntoIter;

use crate::matrix::{MatrixFull, BasicMatrix, MatFormat};
use crate::index::*; 
use crate::tensor_basic_operation::*;
use crate::matrix::matrixfull::*;
use crate::matrix::matrixfullslice::*;

use super::matrix_trait::IncreaseStepBy;

pub struct SubMatrixUpperStepBy<'a, I> {
    pub iter: I,
    rows: Range<usize>,
    columns: Range<usize>,
    max: Option<usize>,
    position: usize,
    matrixupper_index: &'a MatrixUpper<[usize;2]>,
    first_take: bool,
}
impl<'a, I> SubMatrixUpperStepBy<'a, I> {
    pub fn new(iter: I, rows: Range<usize>, columns: Range<usize>, matrixupper_index: &'a MatrixUpper<[usize;2]>) -> SubMatrixUpperStepBy<I> {
        //let position =columns.start*size[0] + rows.start;
        let position =if rows.start<=columns.start {
            columns.start*(columns.start+1)/2 + rows.start
        } else {
            //(columns.start+1)*(columns.start+2)/2 + rows.start
            rows.start*(rows.start+1)/2 + rows.start
        };
        let max = if rows.start>columns.end-1 {
            None
        } else if columns.end >= rows.end {
            Some((columns.end-1)*columns.end/2 + rows.end-1)
        } else {
            Some((columns.end-1)*columns.end/2 + columns.end-1)
        };
        //println!("start: {}, end: {}", position, &max.unwrap());
        SubMatrixUpperStepBy{iter, rows, columns, position,max, matrixupper_index, first_take: true}
    }
}

impl<'a, I> Iterator for SubMatrixUpperStepBy<'a, I>
where I:Iterator,
{
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {

        //println!("debug: {} -> ({},{})", self.position, curr_row, curr_column);

        if let Some(max) = self.max {
            if self.position > max {
                None
            } else {
                let [curr_row, curr_column] = self.matrixupper_index[self.position];
                let is_in_row_range = curr_row >= self.rows.start && curr_row < self.rows.end;
                if self.first_take {
                    //println!("the first position is {}", self.position);
                    self.position += 1;
                    self.first_take = false;
                    self.iter.nth(self.position-1)
                } else if is_in_row_range {
                    //println!("the current position is {}", self.position);
                    self.position += 1;
                    self.iter.next()
                } else {
                    let step = if curr_row >= self.rows.end {
                        (curr_column+1)*(curr_column+2)/2 + self.rows.start - self.position
                    //} else if curr_row < self.rows.start {  ==> Identical, because the condition of is_in_row_range has been considered upper.
                    } else {
                        (curr_column+1)*curr_column/2 + self.rows.start - self.position
                    };
                    //println!("debug: ({},{})-> {}", curr_row, curr_column, self.position+ step);
                    self.position += step+1;
                    //println!("the current position is {}", self.position-1);
                    self.iter.nth(step)
                }
            }
        } else {
            None
        }
    }
}

trait MatrixUpperIterator: Iterator {
    type Item;
    fn matrixupper_step_by_increase(self, step:usize, increase: usize) -> IncreaseStepBy<Self>
    where Self:Sized {
        IncreaseStepBy::new(self, step, increase)
    }
    fn submatrixupper_step_by<'a>(self, rows: Range<usize>, columns: Range<usize>, matrixupper_index: &'a MatrixUpper<[usize;2]>) -> SubMatrixUpperStepBy<'a, Self> 
    where Self:Sized {
        SubMatrixUpperStepBy::new(self, rows, columns, matrixupper_index)
    }
}

impl<'a,T> MatrixUpperIterator for std::slice::Iter<'a,T> {
    type Item = T;
}
impl<'a,T> MatrixUpperIterator for std::slice::IterMut<'a,T> {
    type Item = T;
}

#[test]
fn test_matrixupper() {
    let dd = MatrixUpper::from_vec(435, (0..435).collect::<Vec<usize>>()).unwrap();
    let ff = dd.to_matrixfull().unwrap();
    ff.formated_output_general(20, "full");
    //dd.iter_diagonal().for_each(|x| {println!("{}",x)});
    //ff.iter_diagonal().unwrap().for_each(|x| {println!("{}",x)});


    let matrixupper_index = map_upper_to_full(435).unwrap();
    
    //println!("{:?}", &matrixupper_index);
    let mut output = String::new();
    matrixupper_index.iter().enumerate().for_each(|(i,u)| output = format!("{}; ({},{:?})", output, i, u));
    println!("{}", output);
    dd.iter_submatrix(26..29, 26..29, &matrixupper_index).for_each(|x| {println!("{}",x)});
}

//impl <'a, T> BasicMatrix<'a, T> for MatrixUpper<T> {
//    #[inline]
//    /// `matr_a.size()' return &matr_a.size;
//    fn size(&self) -> &[usize] {
//        let n = ((8.0*self.size.to_owned() as f32+1.0).sqrt()/2.0) as usize;
//        let full_size = [n,n];
//        &full_size
//    }
//    #[inline]
//    /// `matr_a.indicing()' return &matr_a.indicing;
//    fn indicing(&self) -> &[usize] {
//        &self.indicing
//    }
//    #[inline]
//    fn data_ref(&self) -> Option<&[T]> {
//        Some(&self.data[..])
//    }
//    #[inline]
//    fn data_ref_mut(&mut self) -> Option<&mut [T]> {
//        Some(&mut self.data[..])
//    }
//}
pub trait BasicMatUp<'a, T> {

    fn size(&self) -> [usize;2];

    fn len(&self) -> usize;

    fn indicing(&self) -> [usize;2];

    /// by default, the matrix should be contiguous, unless specify explicitly.
    fn is_contiguous(&self) -> bool {true}

    fn data_ref(&self) -> Option<&[T]>; 

    fn data_ref_mut(&mut self) -> Option<&mut [T]>; 

}

impl <'a, T> BasicMatUp<'a, T> for MatrixUpper<T> {
    #[inline]
    fn size(&self) -> [usize;2] {
        let n = ((8.0*self.size.to_owned() as f32+1.0).sqrt()/2.0) as usize;
        [n,n]
    }
    #[inline]
    fn len(&self) -> usize {self.data.len()}
    #[inline]
    fn indicing(&self) -> [usize;2] {
        let n = ((8.0*self.size.to_owned() as f32+1.0).sqrt()/2.0) as usize;
        [1,n]
    }
    #[inline]
    fn data_ref(&self) -> Option<&[T]> {Some(&self.data)}
    #[inline]
    fn data_ref_mut(&mut self) -> Option<&mut [T]> {Some(&mut self.data)}
}
impl <'a, T> BasicMatUp<'a, T> for MatrixUpperSlice<'a, T> {
    #[inline]
    fn size(&self) -> [usize;2] {
        let n = ((8.0*self.size.to_owned() as f32+1.0).sqrt()/2.0) as usize;
        [n,n]
    }
    #[inline]
    fn len(&self) -> usize {self.data.len()}
    #[inline]
    fn indicing(&self) -> [usize;2] {
        let n = ((8.0*self.size.to_owned() as f32+1.0).sqrt()/2.0) as usize;
        [1,n]
    }
    #[inline]
    fn data_ref(&self) -> Option<&[T]> {Some(&self.data)}
    #[inline]
    fn data_ref_mut(&mut self) -> Option<&mut [T]> {None}
}

impl <'a, T> BasicMatUp<'a, T> for MatrixUpperSliceMut<'a, T> {
    #[inline]
    fn size(&self) -> [usize;2] {
        let n = ((8.0*self.size.to_owned() as f32+1.0).sqrt()/2.0) as usize;
        [n,n]
    }
    #[inline]
    fn len(&self) -> usize {self.data.len()}
    #[inline]
    fn indicing(&self) -> [usize;2] {
        let n = ((8.0*self.size.to_owned() as f32+1.0).sqrt()/2.0) as usize;
        [1,n]
    }
    #[inline]
    fn data_ref(&self) -> Option<&[T]> {Some(&self.data)}
    #[inline]
    fn data_ref_mut(&mut self) -> Option<&mut [T]> {Some(&mut self.data)}
}


#[derive(Clone,Debug,PartialEq)]
pub struct MatrixUpper<T> {
    pub size :  usize,
    pub data : Vec<T>
}

impl <T> MatrixUpper<T> {
    pub fn empty() -> MatrixUpper<T> {
        unsafe{MatrixUpper::from_vec_unchecked(0,Vec::new())}
    }
    pub unsafe fn from_vec_unchecked(size: usize, new_vec: Vec<T>) -> MatrixUpper<T> {
        MatrixUpper {
            size,
            data: new_vec
        }
    }
    pub fn from_vec(size: usize, new_vec: Vec<T>) -> Option<MatrixUpper<T>> {
        unsafe{
            let tmp_mat = MatrixUpper::from_vec_unchecked(size, new_vec);
            let len = tmp_mat.size;
            if len>tmp_mat.data.len() {
                panic!("Error: inconsistency happens when formating a matrix from a given vector, (length from size, length of new vector) = ({},{})",len,tmp_mat.data.len());
                None
            } else {
                if len<tmp_mat.data.len() {println!("Waring: the vector size ({}) is larger for the size of the new tensor ({})", tmp_mat.data.len(), len)};
                Some(tmp_mat)
            }

        }
    }
    pub fn iter_diagonal(&self) -> IncreaseStepBy<std::slice::Iter<T>> {
        self.data.iter().matrixupper_step_by_increase(1,1)
    }

    pub fn iter_diagonal_mut(&mut self) -> IncreaseStepBy<std::slice::IterMut<T>> {
        self.data.iter_mut().matrixupper_step_by_increase(1,1)
    }

    pub fn iter_submatrix<'a> (&'a self, rows: Range<usize>, columns: Range<usize>, matrixupper_index: &'a MatrixUpper<[usize;2]>) 
        -> SubMatrixUpperStepBy<slice::Iter<'a, T>> {
        self.data.iter().submatrixupper_step_by(rows, columns, matrixupper_index)
    }
    pub fn iter_submatrix_mut<'a> (&'a mut self, rows: Range<usize>, columns: Range<usize>, matrixupper_index: &'a MatrixUpper<[usize;2]>) 
        -> SubMatrixUpperStepBy<slice::IterMut<'a, T>> {
        self.data.iter_mut().submatrixupper_step_by(rows, columns, matrixupper_index)
    }

    #[inline]
    pub fn to_slice_mut(&mut self) -> &mut [T] {
        &mut self.data[..]
    }
    #[inline]
    pub fn to_matrixupperslicemut(&mut self) -> MatrixUpperSliceMut<T> {
        MatrixUpperSliceMut {
            size: self.size,
            data: &mut self.data[..],
        }
    }

    pub fn size(&self) -> [usize;2] {
        let ndim = ((1.0+8.0*(self.size as f64)).sqrt()*0.5-0.5) as usize;
        [ndim, ndim]
    }

    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.data.iter()
    }
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    pub fn get_diagonal_terms(&self) -> Option<Vec<&T>> {
        let tmp_len = self.size as f64;
        let new_size = ((1.0+8.0*tmp_len).sqrt()*0.5-0.5) as usize;
        if new_size ==0 {
            return None
        } else if new_size*(new_size+1)/2 == self.size {
            let mut tmp_v = vec![&self.data[0]; new_size];
            (0..new_size).for_each(|i| {
                if let Some(to_v) =tmp_v.get_mut(i) {
                    if let Some(fm_v) = self.data.get(i*(i+1)/2+i) {
                        *to_v = fm_v
                    }
                }
            });
            return Some(tmp_v)
        } else {
            return None
        }

    }
}
impl<T: Copy + Clone> MatrixUpper<T> {
    pub fn new(size: usize, new_default: T) -> MatrixUpper<T> {
        MatrixUpper {
            size,
            data: vec![new_default.clone(); size]
        }
    }
    #[inline]
    pub fn to_matrixfull(&self) -> Option<MatrixFull<T>> {
        let tmp_len = self.size as f64;
        let new_size = ((1.0+8.0*tmp_len).sqrt()*0.5-0.5) as usize;
        //println!("{},{}",self.size,new_size);
        if new_size*(new_size+1)/2 == self.size {
            let mut conv_mat = MatrixFull::new([new_size,new_size],self.data[0].clone());
            // now fill the upper part of the matrix
            let mut tmp_len = 1usize;
            let mut tmp_index = 0usize;
            (0..new_size).into_iter().for_each(|j| {
                if let Some(to_slice) = conv_mat.get2d_slice_mut([0,j], tmp_len) {
                    if let Some(from_slice) = self.get1d_slice(tmp_index, tmp_len) {
                        to_slice.iter_mut().zip(from_slice.iter()).for_each(|(a,b)| {
                            *a = b.clone();
                        });
                    } else {
                        panic!("Error in getting a slice from a matrix in MatrixUpper format");
                    }
                } else {
                    panic!("Error in getting a slice from a matrix in MatrixFull format");
                };
                tmp_index += tmp_len;
                tmp_len += 1;
            });
            // now fill the lower part of the matrix
            (0..new_size).into_iter().for_each(|j| {
                (j+1..new_size).into_iter().for_each(|i| {
                    let value = conv_mat.get(&[j,i]).unwrap().clone();
                    if let Some(new_v) = conv_mat.get_mut(&[i,j]) {
                        *new_v = value;
                    } else {
                        panic!("Error in getting a slice from a matrix in MatrixFull format");
                    };
                });
            });
            return Some(conv_mat)
        } else {
            None
        }
    }

    pub fn map_to_matrixfull(&self) -> Option<MatrixUpper<[usize;2]>> {
        let tmp_len = self.size as f64;
        let new_size = ((1.0+8.0*tmp_len).sqrt()*0.5-0.5) as usize;
        //println!("{},{}",self.size,new_size);
        if new_size*(new_size+1)/2 == self.size {
            let mut tmp_index = MatrixUpper::new(self.size,[0_usize,0_usize]);
            let mut i_up = 0_usize;
            for j in (0..new_size) {
                for i in (0..j+1) {
                    tmp_index.data[i_up] = [i,j]
                }
            }
            Some(tmp_index)
        } else {
            None
        }

    }

}
impl<T: Copy + Clone + Display + Send + Sync + Add + AddAssign> Add for MatrixUpper<T> {
    type Output = MatrixUpper<T>;

    fn add(self, other: MatrixUpper<T>) -> MatrixUpper<T> {
        let mut new_tensors: MatrixUpper<T> = self.clone();
        new_tensors.data.iter_mut()
            .zip(other.data.iter())
            .for_each(|(t,f)| {
                *t += f.clone()
        });
        new_tensors
    }
}
impl<T: Clone + Display + Send + Sync + Sub + SubAssign> Sub for MatrixUpper<T> {
    type Output = MatrixUpper<T>;

    fn sub(self, other: MatrixUpper<T>) -> MatrixUpper<T> {
        let mut new_tensors: MatrixUpper<T> = self.clone();
        new_tensors.data.iter_mut()
            .zip(other.data.iter())
            .for_each(|(t,f)| {
                *t -= f.clone()
        });
        new_tensors
    }
}

impl MatrixUpper<f64> {
    pub fn formated_output(&self, n_len: usize, mat_form: &str) {
        let mat_format = if mat_form.to_lowercase()==String::from("full") {MatFormat::Full
        } else if mat_form.to_lowercase()==String::from("upper") {MatFormat::Upper
        } else if mat_form.to_lowercase()==String::from("lower") {MatFormat::Lower
        } else {
            panic!("Error in determing the layout format of the matrix: {}", mat_form)
        };
        let n_row = ((8.0*self.size.to_owned() as f32+1.0).sqrt()/2.0) as usize;
        let n_column = n_row;
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



#[derive(Debug,PartialEq)]
pub struct MatrixUpperSliceMut<'a,T> {
    pub size : usize,
    //pub indicing: &'a usize,
    pub data : &'a mut[T]
}


#[derive(Clone,Debug,PartialEq)]
pub struct MatrixUpperSlice<'a,T> {
    pub size : usize,
    //pub indicing: &'a usize,
    pub data : &'a [T]
}

impl<'a, T> MatrixUpperSlice<'a, T> {
    pub fn from_vec(new_vec: &'a [T]) -> MatrixUpperSlice<'a,T> {
        MatrixUpperSlice {
            size: new_vec.len(),
            data: new_vec
        }
    }
    #[inline]
    pub fn to_matrixfull(&self) -> Option<MatrixFull<T>> 
    where T: Clone + Copy {
        let tmp_len = self.size as f64;
        let new_size = ((1.0+8.0*tmp_len).sqrt()*0.5-0.5) as usize;
        //println!("{},{}",self.size,new_size);
        if new_size*(new_size+1)/2 == self.size {
            let mut conv_mat = MatrixFull::new([new_size,new_size],self.data[0].clone());
            // now fill the upper part of the matrix
            let mut tmp_len = 1usize;
            let mut tmp_index = 0usize;
            (0..new_size).into_iter().for_each(|j| {
                if let Some(to_slice) = conv_mat.get2d_slice_mut([0,j], tmp_len) {
                    if let Some(from_slice) = self.get1d_slice(tmp_index, tmp_len) {
                        to_slice.iter_mut().zip(from_slice.iter()).for_each(|(a,b)| {
                            *a = b.clone();
                        });
                    } else {
                        panic!("Error in getting a slice from a matrix in MatrixUpper format");
                    }
                } else {
                    panic!("Error in getting a slice from a matrix in MatrixFull format");
                };
                tmp_index += tmp_len;
                tmp_len += 1;
            });
            // now fill the lower part of the matrix
            (0..new_size).into_iter().for_each(|j| {
                (j+1..new_size).into_iter().for_each(|i| {
                    let value = conv_mat.get(&[j,i]).unwrap().clone();
                    if let Some(new_v) = conv_mat.get_mut(&[i,j]) {
                        *new_v = value;
                    } else {
                        panic!("Error in getting a slice from a matrix in MatrixFull format");
                    };
                });
            });
            return Some(conv_mat)
        } else {
            None
        }
    }
    pub fn get_diagonal_terms(&self) -> Option<Vec<&T>> {
        let tmp_len = self.size as f64;
        let new_size = ((1.0+8.0*tmp_len).sqrt()*0.5-0.5) as usize;
        if new_size ==0 {
            return None
        } else if new_size*(new_size+1)/2 == self.size {
            let mut tmp_v = vec![&self.data[0]; new_size];
            (0..new_size).for_each(|i| {
                if let Some(to_v) =tmp_v.get_mut(i) {
                    if let Some(fm_v) = self.data.get(i*(i+1)/2+i) {
                        *to_v = fm_v
                    }
                }
            });
            return Some(tmp_v)
        } else {
            return None
        }

    }
    pub fn iter_diagonal(&self) -> IncreaseStepBy<std::slice::Iter<T>> {
        self.data.iter().matrixupper_step_by_increase(1,1)
    }
}

pub fn map_upper_to_full(size: usize) -> Option<MatrixUpper<[usize;2]>> {
    let tmp_len = size as f64;
    let new_size = ((1.0+8.0*tmp_len).sqrt()*0.5-0.5) as usize;
    //println!("{},{}",self.size,new_size);
    if new_size*(new_size+1)/2 == size {
        let mut tmp_index = MatrixUpper::new(size,[0_usize,0_usize]);
        let mut i_up = 0_usize;
        for j in (0..new_size) {
            for i in (0..j+1) {
                tmp_index.data[i_up] = [i,j];
                i_up += 1;
            }
        }
        Some(tmp_index)
    } else {
        None
    }
}
pub fn map_full_to_upper(size: [usize;2]) -> Option<MatrixFull<usize>> {
    //let tmp_len = size as f64;
    //let new_size = ((1.0+8.0*tmp_len).sqrt()*0.5-0.5) as usize;
    //println!("{},{}",self.size,new_size);
    if size[0]==size[1]{
        let mut tmp_index = MatrixFull::new(size,0_usize);
        let mut i_up = 0_usize;
        tmp_index.iter_matrixupper_mut().unwrap().enumerate().for_each(|(i,to)| {*to = i});
        Some(tmp_index)
    } else {
        None
    }
}


#[derive(Debug,PartialEq)]
pub struct MatrixFullSliceMut2<'a,T> {
    pub size : &'a [usize],
    pub indicing: &'a [usize],
    pub data : Vec<&'a mut T>
}

impl<'a, T> MatrixFullSliceMut2<'a,T> {
    #[inline]
    pub fn iter_mut_columns_full(&mut self) -> std::slice::ChunksExactMut<&'a mut T>{
        self.data.chunks_exact_mut(self.size[0])
    }
    
}