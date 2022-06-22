use std::{fmt::Display, collections::binary_heap::Iter, ops::Range, iter::Flatten, vec::IntoIter};
use libc::P_PID;
use typenum::{U4};
use blas::dcopy;
use rayon::prelude::*;
use itertools::iproduct;

use crate::{index::{TensorIndex, TensorIndexUncheck}, Tensors4D, tensors_slice::{TensorsSliceMut, TensorsSlice}, matrix::{MatrixFullSliceMut, MatrixFullSlice}, MatrixFull, MatrixUpperSliceMut, MatrixUpperSlice, TensorOptUncheck, TensorOptMutUncheck, TensorSliceUncheck, TensorSliceMutUncheck};
use crate::tensor_basic_operation::{TensorOpt, TensorOptMut, TensorSlice, TensorSliceMut};
//use crate::matrix::
//{Indexing,Tensors4D};


#[derive(Clone, Copy,Debug, PartialEq)]
pub enum ERIFormat {
    Full,
    Fold4,
    Fold8
}
//#[derive(Clone,Debug,PartialEq)]
//pub struct ERIFull<T:Clone+Display> {
//    /// Coloum-major 4-D ERI designed for quantum chemistry calculations specifically.
//    //pub store_format : ERIFormat,
//    //pub rank: usize,
//    pub size : [usize;4],
//    pub indicing: [usize;4],
//    pub data : Vec<T>
//}

pub type ERIFull<T> = Tensors4D<T,U4>;
//pub type ERI4F<T,D1,D2,D3,D4> = Tensors4D<T,D1,D2,D3,D4>;
//pub type ERI8F<T,D1,D2,D3,D4> = Tensors4D<T,D1,D2,D3,D4>;

//impl ERI4F<T,D1,D2,D3,D4> {
//    pub fn
//}

impl <T: Clone + Display + Send + Sync> ERIFull<T> {
    pub fn new(size: [usize;4], new_default: T) -> ERIFull<T> {
        let mut indicing = [0usize;4];
        //let mut len = size.iter().enumerate().fold(1usize,|len,(di,i)| {
        //    indicing[*i] = len;
        //    len * di
        //});
        let mut len = size.iter().zip(indicing.iter_mut()).fold(1usize,|len,(sizei,indicing_i)| {
            *indicing_i = len;
            len * sizei
        });
        //len *= size[3];
        ERIFull {
            rank: U4::default(),
            size,
            indicing,
            data: vec![new_default.clone(); len]
        }
    }
    pub unsafe fn from_vec_unchecked(size: [usize;4], new_vec: Vec<T>) -> ERIFull<T> {
        let mut indicing = [0usize;4];
        //let mut len = size.iter().enumerate().fold(1usize,|len,(di,i)| {
        let mut len = size.iter().zip(indicing.iter_mut()).fold(1usize,|len,(sizei,indicing_i)| {
            *indicing_i = len;
            len * sizei
        });
        //len *= size[3];
        //if len>new_vec.len() {
        //    panic!("Error: inconsistency happens when formating a tensor from a given vector, (length from size, length of new vector) = ({},{})",len,new_vec.len());
        //} else if len<new_vec.len() {
        //    println!("Waring: the vector size ({}) is larger for the size of the new tensor ({})", new_vec.len(), len);
        //}
        ERIFull {
            rank: U4::default(),
            size,
            indicing,
            data: new_vec
        }
    }
    pub fn from_vec(size: [usize;4], new_vec: Vec<T>) -> Option<ERIFull<T>> {
        unsafe{
            let tmp_eri = ERIFull::from_vec_unchecked(size, new_vec);
            let len = tmp_eri.indicing[3]*tmp_eri.size[3];
            if len>tmp_eri.data.len() {
                panic!("Error: inconsistency happens when formating a tensor from a given vector, (length from size, length of new vector) = ({},{})",len,tmp_eri.data.len());
                None
            } else {
                if len<tmp_eri.data.len() {println!("Waring: the vector size ({}) is larger for the size of the new tensor ({})", tmp_eri.data.len(), len)};
                Some(tmp_eri)
            }

        }
    }
    pub fn get_reducing_matrix_mut(&mut self, i_reduced: &[usize;2]) -> MatrixFullSliceMut<T> {
        let mut position = [0; 4];
        position[2] = i_reduced[0];
        position[3] = i_reduced[1];
        let p_start = self.index4d(position).unwrap();
        let p_length = self.indicing[2];
        MatrixFullSliceMut {
            size: &self.size[0..2],
            indicing: &self.indicing[0..2],
            data : &mut self.data[p_start..p_start+p_length]}
    }

    #[inline]
    pub fn get_reducing_matrix(&self, i_reduced: &[usize;2]) -> MatrixFullSlice<T> {
        let mut position = [0; 4];
        position[2] = i_reduced[0];
        position[3] = i_reduced[1];
        let p_start = self.index4d(position).unwrap();
        let p_length = self.indicing[2];
        MatrixFullSlice {
            size: &self.size[0..2],
            indicing: &self.indicing[0..2],
            data : &self.data[p_start..p_start+p_length]}
    }
    #[inline]
    pub fn chrunk_copy(&mut self, range: [Range<usize>;4], buf: Vec<T>) {
        let mut len = [0usize;4];
        len.iter_mut().zip(range.iter()).for_each(|(i,j)| *i=j.len());
        let mat_local = unsafe{ERIFull::from_vec_unchecked(len, buf)};
        for (ll,l) in (0..len[3]).zip(range[3].clone()) {
            for (kk,k) in (0..len[2]).zip(range[2].clone()) {
                let mut mat_full_kl = self.get_reducing_matrix_mut(&[k,l]);
                let mat_local_kl = mat_local.get_reducing_matrix(&[kk,ll]);
                //for (jj,j) in (0..len[1]).zip(jrange.clone()) {
                for (jj,j) in (0..len[1]).zip(range[1].clone()) {
                    let mut mat_full_klj = mat_full_kl.get2d_slice_mut([range[0].start,j],len[0]).unwrap();
                    let mat_local_klj = mat_local_kl.get2d_slice([0,jj],len[0]).unwrap();
                    //unsafe{
                    //    dcopy(len[0] as i32, mat_local_klj, 1, mat_full_klj, 1);
                    //}
                    mat_full_klj.iter_mut().zip(mat_local_klj.iter()).for_each(|(t,f)| *t = f.clone());
                }
            }
        }
    }
    #[inline]
    pub fn chrunk_copy_transpose_ij(&mut self, range: [Range<usize>;4], buf: Vec<T>) {
        //let mut len = [irange.len(),jrange.len(),krange.len(),lrange.len()];
        let ilen = self.size[0];
        let mut len = [0usize;4];
        len.iter_mut().zip(range.iter()).for_each(|(i,j)| *i=j.len());
        let mat_local = unsafe{ERIFull::from_vec_unchecked(len, buf)};
        for (ll,l) in (0..len[3]).zip(range[3].clone()) {
            for (kk,k) in (0..len[2]).zip(range[2].clone()) {
                let mut mat_full_kl = self.get_reducing_matrix_mut(&[k,l]);
                let mat_local_kl = mat_local.get_reducing_matrix(&[kk,ll]);
                for (jj,j) in (0..len[1]).zip(range[1].clone()) {
                    let mut mat_full_klj = mat_full_kl.get2d_slice_mut([range[0].start,j],len[0]).unwrap();
                    let mat_local_klj = mat_local_kl.get2d_slice([0,jj],len[0]).unwrap();
                    let mut global_ij = j + range[0].start*ilen;
                    for ii in (0..len[0]) {
                        mat_full_kl.data[global_ij] = mat_local_klj[ii].clone();
                        global_ij += ilen;
                    }
                }
            }
        }
    }
}


#[derive(Clone,Debug,PartialEq)]
pub struct ERIFold4<T:Clone+Display> {
    /// Coloum-major 4-D ERI designed for quantum chemistry calculations specifically.
    pub size : [usize;2],
    pub indicing: [usize;2],
    pub data : Vec<T>
}

impl <T: Clone + Display> ERIFold4<T> {
    pub fn new(size: [usize;2], new_default: T) -> ERIFold4<T> {
        let mut indicing = [0usize;2];
        let mut len = size.iter().zip(indicing.iter_mut()).fold(1usize,|len,(sizei,indicing_i)| {
            *indicing_i = len;
            len * sizei
        });
        ERIFold4 {
            size,
            indicing,
            data: vec![new_default.clone(); len]
        }
    }
    pub unsafe fn from_vec_unchecked(size: [usize;2], new_vec: Vec<T>) -> ERIFold4<T> {
        let mut indicing = [0usize;2];
        let mut len = size.iter().zip(indicing.iter_mut()).fold(1usize,|len,(sizei,indicing_i)| {
            *indicing_i = len;
            len * sizei
        });
        ERIFold4 {
            size,
            indicing,
            data: new_vec
        }
    }
    pub fn from_vec(size: [usize;2], new_vec: Vec<T>) -> Option<ERIFold4<T>> {
        unsafe{
            let tmp_eri = ERIFold4::from_vec_unchecked(size, new_vec);
            let len = tmp_eri.indicing[1]*tmp_eri.size[1];
            if len>tmp_eri.data.len() {
                panic!("Error: inconsistency happens when formating a tensor from a given vector, (length from size, length of new vector) = ({},{})",len,tmp_eri.data.len());
                None
            } else {
                if len<tmp_eri.data.len() {println!("Waring: the vector size ({}) is larger for the size of the new tensor ({})", tmp_eri.data.len(), len)};
                Some(tmp_eri)
            }

        }
    }
    pub fn get_reducing_matrix_mut(&mut self, i_reduced: usize) -> MatrixUpperSliceMut<T> {
        let p_start = self.index2d([0,i_reduced]).unwrap();
        let p_length = self.indicing[1];
        MatrixUpperSliceMut {
            size: &self.size[0],
            //indicing: &self.indicing[0],
            data : &mut self.data[p_start..p_start+p_length]}
    }

    #[inline]
    pub fn get_reducing_matrix(&self, i_reduced: usize) -> MatrixUpperSlice<T> {
        let p_start = self.index2d([0,i_reduced]).unwrap();
        let p_length = self.indicing[1];
        MatrixUpperSlice {
            size: &self.size[0],
            //indicing: &self.indicing[0],
            data : &self.data[p_start..p_start+p_length]}
    }
    #[inline]
    pub fn get_slices_mut(&mut self, dim: usize, d1: Range<usize>, d2:Range<usize>, d3:Range<usize>, d4:Range<usize>) -> Flatten<IntoIter<&mut [T]>> {
        //let mut tmp_slices = vec![&self.data[..]; d2.len()*d3.len()*d4.len()];
        let mut tmp_slices: Vec<&mut [T]> = Vec::new();
        let len_slices_d1 = d1.len();
        let dd = self.data.split_at_mut(0).1;
        let len_slices_d1 = d1.len();
        let len_d12 = dim*(dim+1)/2;
        let len_d34 = len_d12;
        
        //let mut debug_total_len = 0_usize;

        let mut final_slice_len = 0_usize;
        let mut start = 0_usize;
        iproduct!(d4,d3,d2).fold((dd,0_usize),|(ee, offset),(d4,d3,d2)| {
            if d3 > d4 || d2 < d1.start {
                (ee,offset)
            } else {
                start = (d4*(d4+1)/2+d3)*len_d12 + d2*(d2+1)/2 + d1.start;
                // check the slice length to gather
                final_slice_len = if d2 >= d1.end {len_slices_d1} else {d2-d1.start+1};
                let gg = ee.split_at_mut(start-offset).1.split_at_mut(final_slice_len);
                tmp_slices.push(gg.0);
                //debug_total_len += final_slice_len;
                (gg.1,start+final_slice_len)
            }
        });

        //let len = tmp_slices.len();
        
        //println!("debug: the length of to_slices: {}", debug_total_len);
        tmp_slices.into_iter().flatten()

    }
    #[inline]
    pub fn chunk_copy_from_local_erifull(&mut self, dim:usize, d1:Range<usize>, d2:Range<usize>, d3:Range<usize>, d4:Range<usize>, buf: Vec<T>) {
        //let mat_local = unsafe{ERIFull::from_vec_unchecked([d1.len(),d2.len(),d3.len(),d4.len()], buf)};
        // prepare the slices from self for copy
        let mut to_slices = self.get_slices_mut(dim, d1.clone(), d2.clone(), d3.clone(), d4.clone());

        // now prepare the slices form buf to copy
        let mut from_slices: Vec<&[T]> = Vec::new();
        let mut final_slice_len = 0_usize;
        let local_ind1 = d1.len();
        let local_ind2 = local_ind1*d2.len();
        let local_ind3 = local_ind2*d3.len();

        //let mut debug_total_len = 0_usize;

        let mut start = 0_usize;
        let mut final_len = 0_usize;
        let max_final_len = d1.len();
        for l in d4.enumerate() {
            let start_l = l.0*local_ind3;
            for k in d3.clone().enumerate() {
                if k.1 <= l.1 {
                    let start_kl = start_l + k.0*local_ind2;
                    for j in d2.clone().enumerate() {
                        if j.1 >=d1.start {
                            start = start_kl + j.0*local_ind1;
                            final_len = if j.1>=d1.end {max_final_len} else {j.1-d1.start+1};
                            from_slices.push(&buf[start..start+final_len]);
                            //debug_total_len += final_len;
                        }
                    }
                }
            }
        }
        //println!("debug: the length of from_slices: {}", debug_total_len);
        from_slices.into_iter().flatten().zip(to_slices).for_each(|value| {*value.1=value.0.clone()});
    }
}

impl ERIFold4<f64> {
    #[inline]
    /// Establish especially for the tensor generation of ERIFold4, for which
    /// the eri subtensors are generated by LibCINT
    pub fn chunk_copy_from_a_full_vector(&mut self, range: [Range<usize>;4], buf: Vec<f64>) {
        //========================================================
        // algorithm 1: locate and copy each slices one by one
        //========================================================
        let mut len = [0usize;4];
        len.iter_mut().zip(range.iter()).for_each(|(i,j)| *i=j.len());
        let mat_local = unsafe{ERIFull::from_vec_unchecked(len, buf)};
        if range[0].start<range[1].start {
            for (ll,l) in range[3].clone().enumerate() {
                for (kk,k) in range[2].clone().enumerate() {
                    if k>l {continue};
                    let klpair = (l+1)*l/2+k; // nevigate to the (k,l) position in the upper-formated tensor
                    let mut mat_full_kl = self.get_reducing_matrix_mut(klpair);
                    let mat_local_kl = mat_local.get_reducing_matrix(&[kk,ll]);
                    let mut local_start = 0usize;
                    for (jj,j) in range[1].clone().enumerate() {
                        let mut mat_full_klj = mat_full_kl
                            .get2d_slice_mut_uncheck([range[0].start,j],len[0]).unwrap();
                        //let mat_local_klj = mat_local_kl.get2d_slice([0,jj],len[0]).unwrap();
                        let mat_local_klj = mat_local_kl
                            .get1d_slice(local_start,len[0]).unwrap();
                        mat_full_klj.iter_mut().zip(mat_local_klj.iter()).for_each(|(t,f)| *t = f.clone());
                        local_start += mat_local.size[0];
                    };
                    //let mut tmp_slices: Vec<&mut [f64]> = Vec::new();
                    //let mut dd = mat_full_kl.data.split_at_mut(0).1;
                    //range[1].clone().fold((dd,0_usize),|(ee,offset),y| {

                    //    (dd,0_usize)
                    //});

                }
            }
        } else if range[0].start==range[1].start {
            for (ll,l) in range[3].clone().enumerate() {
                for (kk,k) in range[2].clone().enumerate() {
                    if k>l {continue};
                    let klpair = (l+1)*l/2+k; // nevigate to the (k,l) position in the upper-formated tensor
                    let mut mat_full_kl = self.get_reducing_matrix_mut(klpair);
                    let mat_local_kl = mat_local.get_reducing_matrix(&[kk,ll]);
                    let mut local_start = 0usize;
                    for (jj,j) in range[1].clone().enumerate() {
                        let mut mat_full_klj = mat_full_kl.get2d_slice_mut_uncheck([range[0].start,j],jj+1).unwrap();
                        //let mat_local_klj = mat_local_kl.get2d_slice([0,jj],jj).unwrap();
                        let mat_local_klj = mat_local_kl.get1d_slice(local_start,jj+1).unwrap();
                        //unsafe{
                        //    dcopy(len[0] as i32, mat_local_klj, 1, mat_full_klj, 1);
                        //}
                        mat_full_klj.iter_mut().zip(mat_local_klj.iter()).for_each(|(t,f)| *t = f.clone());
                        local_start += len[0];
                    }
                }
            }
        }
        //=======================================================================
        // algorithm 2: filter out the discontinued slices and copy them once
        //=======================================================================
        //let tmp_len = self.size[0] as f64;
        //let new_size = ((1.0+8.0*tmp_len).sqrt()*0.5-0.5) as usize;
        //let mut to_slices = self.get_slices_mut(new_size, range[0].clone(), range[1].clone(), range[2].clone(), range[3].clone());

    }
}