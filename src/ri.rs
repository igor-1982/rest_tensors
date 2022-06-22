use std::iter::Flatten;
use std::slice::{ChunksExactMut,ChunksExact};
use std::vec::IntoIter;
use std::{fmt::Display, collections::binary_heap::Iter, iter::Filter, convert};
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, Index, Range};
use typenum::{U2, Pow};
use rayon::{prelude::*,slice};
use itertools::iproduct;


use crate::{MatrixFullSliceMut, MatrixFullSlice};
use crate::{index::{TensorIndex, TensorIndexUncheck}, Tensors4D, TensorOpt, TensorOptMut, TensorSlice, TensorSliceMut, TensorOptUncheck, TensorSliceUncheck, TensorSliceMutUncheck, TensorOptMutUncheck, MatFormat};

#[derive(Clone,Debug,PartialEq)]
pub struct RIFull<T:Clone+Display> {
    /// Coloum-major 4-D ERI designed for quantum chemistry calculations specifically.
    pub size : [usize;3],
    pub indicing: [usize;3],
    pub data : Vec<T>
}

impl <T: Clone + Display + Send + Sync> RIFull<T> {
    pub fn new(size: [usize;3], new_default: T) -> RIFull<T> {
        let mut indicing = [0usize;3];
        let mut len = size.iter().zip(indicing.iter_mut()).fold(1usize,|len,(sizei,indicing_i)| {
            *indicing_i = len;
            len * sizei
        });
        //if let Some(value)=size.get(2) {len *= value};
        //println!("{}", len);
        RIFull {
            size,
            indicing,
            data: vec![new_default; len]
        }
    }
    pub unsafe fn from_vec_unchecked(size: [usize;3], new_vec: Vec<T>) -> RIFull<T> {
        let mut indicing = [0usize;3];
        let mut len = size.iter().zip(indicing.iter_mut()).fold(1usize,|len,(sizei,indicing_i)| {
            *indicing_i = len;
            len * sizei
        });
        //if let Some(value)=size.get(2) {len *= value};
        RIFull {
            size,
            indicing,
            data: new_vec
        }
    }
    pub fn from_vec(size: [usize;3], new_vec: Vec<T>) -> Option<RIFull<T>> {
        unsafe{
            let tmp_tensor = RIFull::from_vec_unchecked(size, new_vec);
            let len = tmp_tensor.size.iter().fold(1_usize,|acc,x| {acc*x});
            if len>tmp_tensor.data.len() {
                panic!("Error: inconsistency happens when formating a tensor from a given vector, (length from size, length of new vector) = ({},{})",len,tmp_tensor.data.len());
                None
            } else {
                if len<tmp_tensor.data.len() {println!("Waring: the vector size ({}) is larger for the size of the new tensor ({})", tmp_tensor.data.len(), len)};
                Some(tmp_tensor)
            }

        }
    }
    pub fn get_reducing_matrix_mut(&mut self, i_reduced: usize) -> Option<MatrixFullSliceMut<T>> {
        let p_length = if let Some(value) = self.indicing.get(2) {value} else {return None};
        let p_start = p_length * i_reduced; 
        Some(MatrixFullSliceMut {
            size: &self.size[0..2],
            indicing: &self.indicing[0..2],
            data : &mut self.data[p_start..p_start+p_length]}
        )
    }

    #[inline]
    pub fn get_reducing_matrix(&self, i_reduced: usize) -> Option<MatrixFullSlice<T>> {
        let p_length = if let Some(value) = self.indicing.get(2) {value} else {return None};
        let p_start = p_length * i_reduced;
        Some(MatrixFullSlice {
            size: &self.size[0..2],
            indicing: &self.indicing[0..2],
            data : &self.data[p_start..p_start+p_length]}
        )
    }
    #[inline]
    pub fn get_slices(&self, x: Range<usize>, y: Range<usize>, z: Range<usize>) -> Flatten<IntoIter<&[T]>> {
        let mut tmp_slices = vec![&self.data[..]; y.len()*z.len()];
        let len_slices_x = x.len();
        let len_y = self.indicing[1];
        let len_z = self.indicing[2];
        tmp_slices.iter_mut().zip(iproduct!(z,y)).for_each(|(t,(z,y))| {
            let start = x.start + y*len_y + z*len_z;
            //println!("start: {}, end: {}, y:{}, z:{}", start, start+x.len(), y,z);
            *t = &self.data[start..start + len_slices_x];
        });
        tmp_slices.into_iter().flatten()
    }
    #[inline]
    pub fn get_slices_mut(& mut self, x: Range<usize>, y: Range<usize>, z: Range<usize>) -> Flatten<IntoIter<&mut [T]>> {
        let mut tmp_slices: Vec<&mut [T]> = vec![];
        let mut dd = self.data.split_at_mut(0).1;
        let len_slices_x = x.len();
        let len_y = self.indicing[1];
        let len_z = self.indicing[2];
        iproduct!(z,y).fold((dd,0_usize),|(ee, offset), (z,y)| {
            let start = x.start + y*len_y + z*len_z;
            let gg = ee.split_at_mut(start-offset).1.split_at_mut(len_slices_x);
            tmp_slices.push(gg.0);
            (gg.1,start+len_slices_x)
        });
        tmp_slices.into_iter().flatten()
    }
    #[inline]
    pub fn iter_mut_auxbas(&mut self, auxbas_range: Range<usize>) -> Option<ChunksExactMut<T>> {
        if let Some(x) = self.size.get(0) {
            if let Some(y) = self.size.get(1) {
                let chunk_size = x*y;
                Some(self.data[chunk_size*auxbas_range.start..chunk_size*auxbas_range.end]
                        .chunks_exact_mut(chunk_size))
            } else {None}
        } else {None}
    }
    #[inline]
    pub fn iter_auxbas(&self, auxbas_range: Range<usize>) -> Option<ChunksExact<T>> {
        if let Some(x) = self.size.get(0) {
            if let Some(y) = self.size.get(1) {
                let chunk_size = x*y;
                Some(self.data[chunk_size*auxbas_range.start..chunk_size*auxbas_range.end]
                        .chunks_exact(chunk_size))
            } else {None}
        } else {None}
    }
    #[inline]
    pub fn par_iter_mut_auxbas(&mut self, auxbas_range: Range<usize>) -> Option<rayon::slice::ChunksExactMut<T>> {
        if let Some(x) = self.size.get(0) {
            if let Some(y) = self.size.get(1) {
                let chunk_size = x*y;
                Some(self.data[chunk_size*auxbas_range.start..chunk_size*auxbas_range.end]
                        .par_chunks_exact_mut(chunk_size))
            } else {None}
        } else {None}
    }
    #[inline]
    pub fn par_iter_auxbas(&self, auxbas_range: Range<usize>) -> Option<rayon::slice::ChunksExact<T>> {
        if let Some(x) = self.size.get(0) {
            if let Some(y) = self.size.get(1) {
                let chunk_size = x*y;
                Some(self.data[chunk_size*auxbas_range.start..chunk_size*auxbas_range.end]
                        .par_chunks_exact(chunk_size))
            } else {None}
        } else {None}
    }
}
