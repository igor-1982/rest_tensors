use std::ffi::{c_double, c_int};
use std::iter::Flatten;
use std::slice::{ChunksExactMut,ChunksExact};
use std::vec::IntoIter;
use std::{fmt::Display, collections::binary_heap::Iter, iter::Filter, convert};
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, Index, Range, MulAssign, DivAssign};
use typenum::{U2, Pow};
use rayon::{prelude::*,slice};
use itertools::iproduct;


use crate::external_libs::ri_ao2mo_f;
use crate::matrix_blas_lapack::{_dgemm_nn,_dgemm_tn, _dgemm_tn_v02};
use crate::{MatrixFullSliceMut, MatrixFullSlice, MatrixFull, BasicMatrix};
use crate::{index::{TensorIndex, TensorIndexUncheck}, Tensors4D, TensorOpt, TensorOptMut, TensorSlice, TensorSliceMut, TensorOptUncheck, TensorSliceUncheck, TensorSliceMutUncheck, TensorOptMutUncheck};

#[derive(Clone,Debug,PartialEq)]
pub struct RIFull<T> {
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
    pub fn empty() -> RIFull<T> {
        RIFull { size: [0,0,0], indicing: [0,0,0], data: Vec::new() }
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

    //pub fn get_reducing_matrix_mut(&mut self, i_reduced: usize) -> Option<MatrixFullSliceMut<T>> {
    //    let p_length = if let Some(value) = self.indicing.get(2) {value} else {return None};
    //    let p_start = p_length * i_reduced; 
    //    Some(MatrixFullSliceMut {
    //        size: &self.size[0..2],
    //        indicing: &self.indicing[0..2],
    //        data : &mut self.data[p_start..p_start+p_length]}
    //    )
    //}

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
        self.get_slices_mut_v01(x,y,z)
    }
    #[inline]
    pub fn get_slices_mut_v01(& mut self, x: Range<usize>, y: Range<usize>, z: Range<usize>) -> Flatten<IntoIter<&mut [T]>> {
        //let mut tmp_slices: Vec<&mut [T]> = vec![];
        let length = y.len()*z.len();
        let mut tmp_slices: Vec<&mut [T]> = Vec::with_capacity(length);
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
    pub fn get_slices_mut_v02(& mut self, x: Range<usize>, y: Range<usize>, z: Range<usize>) -> Flatten<IntoIter<&mut [T]>> {
        let length = y.len()*z.len();
        let mut tmp_slices: Vec<&mut [T]> = Vec::with_capacity(length);
        unsafe{tmp_slices.set_len(length)}
        let mut dd = self.data.split_at_mut(0).1;
        let len_slices_x = x.len();
        let len_y = self.indicing[1];
        let len_z = self.indicing[2];
        iproduct!(z,y).zip(tmp_slices.iter_mut()).fold((dd,0_usize),|(ee, offset), ((z,y),to_slice)| {
            let start = x.start + y*len_y + z*len_z;
            let gg = ee.split_at_mut(start-offset).1.split_at_mut(len_slices_x);
            *to_slice = gg.0;
            (gg.1,start+len_slices_x)
        });
        tmp_slices.into_iter().flatten()
    }
    #[inline]
    pub fn iter_slices_x(&self, y: usize, z: usize) -> std::slice::Iter<T> {
        let start = z*self.indicing[2] + y*self.indicing[1];
        let end = start + self.indicing[1];
        self.data[start..end].iter()
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
    #[inline]
    pub fn check_shape(&self, other:&RIFull<T>) -> bool {
        self.size.iter().zip(other.size.iter()).fold(true, |check,size| {
            check && size.0==size.1
        })
    }
}

impl RIFull<f64> {
    #[inline]
    pub fn self_scaled_add(&mut self, bm: &RIFull<f64>, b: f64) {
        /// A = A + b*B
        if self.check_shape(bm) {
            self.data.iter_mut()
                .zip(bm.data.iter())
                .for_each(|(c,p)| {*c +=p*b});
        } else {
            panic!("Error: Shape inconsistency happens when plus two matrices");
        }
    }
    #[inline]
    pub fn ao2mo(&self, eigenvector: &MatrixFull<f64>) -> anyhow::Result<RIFull<f64>> {
        self.ao2mo_v02(eigenvector)
    }
    #[inline]
    pub fn ao2mo_v01(&self, eigenvector: &MatrixFull<f64>) -> anyhow::Result<RIFull<f64>> {
        /// AO(num_basis, num_basis, num_auxbas) -> MO(num_auxbas, num_state, num_state)
        let num_basis = eigenvector.size.get(0).unwrap().clone();
        let num_state = eigenvector.size.get(1).unwrap().clone();
        let num_auxbas = self.size.get(2).unwrap().clone();
        let mut rimo = RIFull::new([num_auxbas,num_basis,num_state],0.0);

        for i_auxbas in (0..num_auxbas) {
            let i_aux = &self.get_reducing_matrix(i_auxbas).unwrap();
            let tmp_aux = _dgemm_nn(i_aux, &eigenvector.to_matrixfullslice());
            //_dgemm_tn_v02(&eigenvector.to_matrixfullslice(), 
            //              &tmp_aux.to_matrixfullslice(),
            //              rimo.get_slices_mut_old(i_auxbas..i_auxbas+1, 0..num_basis, 0..num_state)
            //            )
            let tmp_aux2 = _dgemm_tn(&eigenvector.to_matrixfullslice(), &tmp_aux.to_matrixfullslice());
            rimo.get_slices_mut_v01(i_auxbas..i_auxbas+1, 0..num_basis, 0..num_state)
                .zip(tmp_aux2.data.iter()).for_each(|(to, from)| {*to = *from});
        }
        Ok(rimo)
    }
    #[inline]
    pub fn ao2mo_v02(&self, eigenvector: &MatrixFull<f64>) -> anyhow::Result<RIFull<f64>> {
        /// AO(num_basis, num_basis, num_auxbas) -> MO(num_auxbas, num_state, num_state)
        let num_basis = eigenvector.size.get(0).unwrap().clone();
        let num_states = eigenvector.size.get(1).unwrap().clone();
        let num_auxbas = self.size.get(2).unwrap().clone();
        let mut ri3mo = RIFull::new([num_auxbas,num_states,num_states],0.0);
        //let mut buf = vec![0.0,num_basis*num_states*num_auxbas];
        //let (c_buf, buf_len, buf_cap) = (buf.as_mut_ptr() as *mut f64, buf.len(), buf.capacity());

        ri_ao2mo_f(&eigenvector.data_ref().unwrap(),
                   &self.data[..], 
                   &mut ri3mo.data[..],
                   num_states, num_basis, num_auxbas
                );
        //unsafe{
        //    let eigenvector_ptr = eigenvector.data.as_ptr();
        //    let ri3fn_ptr = self.data.as_ptr();
        //    let ri3mo_ptr = ri3mo.data.as_mut_ptr();
        //    ri_ao2mo_f_(eigenvector_ptr, 
        //        ri3fn_ptr, 
        //        ri3mo_ptr, 
        //        &(num_states as i32), 
        //        &(num_basis as i32), 
        //        &(num_auxbas as i32));
        //}

        Ok(ri3mo)
    }
}

