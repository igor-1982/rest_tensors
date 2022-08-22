use std::{fmt::Display, collections::binary_heap::Iter, iter::{Filter,Flatten, Map}, convert, slice::{ChunksExact,ChunksExactMut}};
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, Index, Range};
use libc::CLOSE_RANGE_CLOEXEC;
use typenum::{U2, Pow};
use rayon::{prelude::*, collections::btree_map::IterMut};
use std::vec::IntoIter;

use crate::{index::{TensorIndex, TensorIndexUncheck}, Tensors4D, TensorOpt, TensorOptMut, TensorSlice, TensorSliceMut, TensorOptUncheck, TensorSliceUncheck, TensorSliceMutUncheck, TensorOptMutUncheck, MatFormat};
//{Indexing,Tensors4D};


//#[derive(Clone, Copy,Debug, PartialEq)]
//pub enum MatFormat {
//    Full,
//    Upper,
//    Lower
//}
#[derive(Clone,Debug,PartialEq)]
pub struct MatrixFull<T:Clone+Display+Send+Sync> {
    /// Coloum-major 4-D ERI designed for quantum chemistry calculations specifically.
    //pub store_format : ERIFormat,
    //pub rank: usize,
    pub size : [usize;2],
    pub indicing: [usize;2],
    pub data : Vec<T>
}

//pub type MatrixFull<T> = Tensors4D<T,U2>;

impl <T: Copy + Clone + Display + Send + Sync> MatrixFull<T> {
    pub fn empty() -> MatrixFull<T> {
        unsafe{MatrixFull::from_vec_unchecked([0,0],Vec::new())}
    }
    pub fn new(size: [usize;2], new_default: T) -> MatrixFull<T> {
        let mut indicing = [0usize;2];
        let mut len = size.iter()
            .zip(indicing.iter_mut())
            .fold(1usize,|len,(di,ii)| {
            *ii = len;
            len * di
        });
        //len *= size[1];
        //println!("matrixfull size: {:?},{:?},{:?}",size,len,indicing);
        MatrixFull {
            size,
            indicing,
            data: vec![new_default.clone(); len]
        }
    }
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
    /// Generate a borrowed vector from MatrixFull<T>
    pub fn as_vec_ref(&self) -> &Vec<T> {
        &self.data
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
    pub fn transpose_old(&self) -> MatrixFull<T> {
        let mut trans_mat = self.clone();
        trans_mat.size[0] = self.size[1];
        trans_mat.size[1] = self.size[0];
        trans_mat.indicing = [0usize;2];
        let mut len = trans_mat.size.iter()
            .zip(trans_mat.indicing.iter_mut())
            .fold(1usize,|len,(di,ii)| {
                *ii = len;
                len * di
            });
        for i in (0..self.size[0]) {
            for j in (0..self.size[1]) {
                let tvalue = self.get2d([i,j]).unwrap();
                trans_mat.set2d([j,i],tvalue.clone());
            }
        }
        trans_mat
    }
    #[inline]
    pub fn transpose_and_drop_old(self) -> MatrixFull<T> {
        let mut trans_mat = self.clone();
        trans_mat.size[0] = self.size[1];
        trans_mat.size[1] = self.size[0];
        trans_mat.indicing = [0usize;2];
        let mut len = trans_mat.size.iter()
            .zip(trans_mat.indicing.iter_mut())
            .fold(1usize,|len,(di,ii)| {
                *ii = len;
                len * di
            });
        for i in (0..self.size[0]) {
            for j in (0..self.size[1]) {
                let tvalue = self.get2d([i,j]).unwrap();
                trans_mat.set2d([j,i],tvalue.clone());
            }
        }
        trans_mat
    }
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
            trans_mat.get_slices_mut(i..i+1,0..x_len).zip(c)
            .for_each(|(to,from)| {*to = *from})
        });
        trans_mat
    }
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
            trans_mat.get_slices_mut(i..i+1,0..x_len).zip(c)
            .for_each(|(to,from)| {*to = *from})
        });
        trans_mat
    }
    #[inline]
    pub fn check_shape(&self, other:&MatrixFull<T>) -> bool {
        self.size.iter().zip(other.size.iter()).fold(true, |check,size| {
            check && size.0==size.1
        })
    }
    #[inline]
    pub fn to_matrixupper(&self) -> MatrixUpper<T> {
        if self.size[0]!=self.size[1] {
            panic!("Error: Nonsymmetric matrix cannot be converted to the upper format");
        }
        //let mut um = MatrixUpper::from_vec_unchecked(self.size[0]*(self.size[0]+1)/2,self.data[0]);
        //(0..self.size[0]).into_iter().for_each(|i| {
        //    let from_slice = self.get2d_slice([0,i], i+1).unwrap();
        //    let mut to_slice = um.get1d_slice_mut(i*(i+1)/2, i+1).unwrap();
        //    to_slice.iter_mut().zip(from_slice.iter()).for_each(|(t,f)| {
        //        *t = f.clone()
        //    });
        //});
        unsafe{MatrixUpper::from_vec_unchecked(
            self.size[0]*(self.size[0]+1)/2,
            self.data.iter().enumerate()
                .filter(|id| id.0%self.size[0]<=id.0/self.size[0])
                .map(|ad| ad.1.clone())
                .collect::<Vec<_>>()
        )}
    }
    //#[inline]
    //pub fn iter_matrixupper(&self) -> std::iter::Map<std::iter::Filter<std::iter::Enumerate<std::slice::Iter<'_,T>>>> {
    //    self.data.iter().enumerate()
    //        .filter(|id| id.0%self.size[0]<=id.0/self.size[0])
    //        .map(|ad| ad.1)
    //}
    /// after extension, the shape of MatrixFull collapes to [self.data.len(),1];
    pub fn extend(&mut self, data: Vec<T>) {
        self.data.extend(data);
        let len = self.data.len();
        self.size = [len,1];
        self.indicing = [1,len];
    }
    pub fn reshape(&mut self, size:[usize;2]) {
        if size.iter().product::<usize>() !=self.data.len() {
            panic!("Cannot reshape a MatrixFull ({:?}) to a new shape with different data length: {:?}", &self.size,&size);
        } else {
            self.size = size;
            self.indicing = [0usize;2];
            let mut len = self.size.iter().zip(self.indicing.iter_mut()).fold(1usize,|len,(di,ii)| {
                *ii = len;
                len * di
            });
        }
    }

    #[inline]
    pub fn get_slices(&self, x: Range<usize>, y: Range<usize>) -> Flatten<IntoIter<&[T]>> {
        let mut tmp_slices = vec![&self.data[..]; y.len()];
        let len_slices_x = x.len();
        let len_y = self.indicing[1];
        tmp_slices.iter_mut().zip(y).for_each(|(t,y)| {
            let start = x.start + y*len_y;
            //println!("start: {}, end: {}, y:{}, z:{}", start, start+x.len(), y,z);
            *t = &self.data[start..start + len_slices_x];
        });
        tmp_slices.into_iter().flatten()
    }
    #[inline]
    pub fn iter_j(&self, j: usize) -> std::slice::Iter<T> {
        let len = self.indicing[1];
        self.data[j*len..(j+1)*len].iter()
    }
    #[inline]
    pub fn iter_slice_x(&self, y: usize) -> std::slice::Iter<T> {
        let start = self.indicing[1]*y;
        let end = self.indicing[1]*(y+1);
        self.data[start..end].iter()
    }
    #[inline]
    pub fn get_slice_x(&self, y: usize) -> & [T] {
        let start = self.indicing[1]*y;
        let end = self.indicing[1]*(y+1);
        & self.data[start..end]
    }
    #[inline]
    pub fn get_slices_mut(& mut self, x: Range<usize>, y: Range<usize>) -> Flatten<IntoIter<&mut [T]>> {
        let mut tmp_slices: Vec<&mut [T]> = vec![];
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
    pub fn iter_mut_j(&mut self, j: usize) -> Flatten<IntoIter<&mut [T]>> {
        self.get_slices_mut(0..self.size[0], j..j+1)
    }
    #[inline]
    pub fn par_iter_mut_j(&mut self, j: usize) -> rayon::slice::IterMut<T> {
        let start = self.indicing[1]*j;
        let end = start + self.indicing[1];
        self.data[start..end].par_iter_mut()
    }
    #[inline]
    pub fn get_slices_mut_j(&mut self, j: usize) -> &mut [T] {
        //self.get_slices_mut(0..self.size[0], j..j+1)
        let start = self.indicing[0]*j;
        let end = start + self.indicing[0];
        &mut self.data[start..end]
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
    pub fn par_iter_mut_columns(&mut self,range_column: Range<usize>) -> Option<rayon::slice::ChunksExactMut<T>>{
        if let Some(n_chunk) = self.size.get(0) {
            Some(self.data[n_chunk*range_column.start..n_chunk*range_column.end].par_chunks_exact_mut(*n_chunk))
        }  else {
            None
        }
    }
    #[inline]
    pub fn iter_columns(&self, range_column: Range<usize>) -> Option<ChunksExact<T>>{
        if let Some(n_chunk) = self.size.get(0) {
            Some(self.data[n_chunk*range_column.start..n_chunk*range_column.end]
                    .chunks_exact(*n_chunk))
        }  else {
            None
        }
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
    pub fn iter_columns_full(&self) -> ChunksExact<T>{
        self.data.chunks_exact(self.size[0])
    }
    #[inline]
    pub fn par_iter_columns_full(&self) -> rayon::slice::ChunksExact<T>{
        self.data.par_chunks_exact(self.size[0])
    }
    #[inline]
    pub fn iter_mut_columns_full(&mut self) -> ChunksExactMut<T>{
        self.data.chunks_exact_mut(self.size[0])
    }
    #[inline]
    pub fn par_iter_mut_columns_full(&mut self) -> rayon::slice::ChunksExactMut<T>{
        self.data.par_chunks_exact_mut(self.size[0])
    }
    //#[inline]
    //pub fn iter_rows(&self) -> Option<ChunksExact<T>>{
    //    //let tmp_v = vec![]
    //    if let Some(n_chunk) = self.size.get(1) {
    //        //Some(self.data.chunks_exact(*n_chunk))
    //    }  else {
    //        //None
    //    }
    //}
}

impl<T: Copy + Clone + Display + Send + Sync + Add + AddAssign> Add for MatrixFull<T> {
    type Output = MatrixFull<T>;

    fn add(self, other: MatrixFull<T>) -> MatrixFull<T> {
        let mut new_tensors: MatrixFull<T> = self.clone();
        new_tensors.data.iter_mut()
            .zip(other.data.iter())
            .for_each(|(t,f)| {
                *t += f.clone()
        });
        new_tensors
    }
}
impl<T: Clone + Display + Send + Sync + Sub + SubAssign> Sub for MatrixFull<T> {
    type Output = MatrixFull<T>;

    fn sub(self, other: MatrixFull<T>) -> MatrixFull<T> {
        let mut new_tensors: MatrixFull<T> = self.clone();
        new_tensors.data.iter_mut()
            .zip(other.data.iter())
            .for_each(|(t,f)| {
                *t -= f.clone()
        });
        new_tensors
    }
}
impl<T: Copy + Clone + Display + Send + Sync> Index<[usize;2]> for MatrixFull<T> {
    type Output = T;
    fn index(&self, position:[usize;2]) -> &Self::Output {
        self.get2d(position).unwrap()
    }
}

impl MatrixFull<f64> {
    pub fn get_diagonal_terms(&self) -> Option<Vec<&f64>> {
        //let tmp_len = self.size;
        let new_size = self.size[0];
        if new_size ==0 {
            return None
        } else if self.size[0] == self.size[1] {
            let mut tmp_v = vec![&self.data[0]; new_size];
            (0..new_size).for_each(|i| {
                if let Some(to_v) =tmp_v.get_mut(i) {
                    if let Some(fm_v) = self.data.get(i*new_size+i) {
                        *to_v = fm_v
                    }
                }
            });
            return Some(tmp_v)
        } else {
            return None
        }

    }

    pub fn add(&self, other: &MatrixFull<f64>) -> Option<MatrixFull<f64>> {
        if self.check_shape(other) {
            let mut new_tensors: MatrixFull<f64> = self.clone();
            //new_tensors.data.par_iter_mut()
            //    .zip(other.data.par_iter())
            //    .map(|(c,p)| {*c +=p});
            (0..new_tensors.data.len()).into_iter().for_each(|i| {
                new_tensors.data[i] += other.data[i].clone();
            });
            Some(new_tensors)
        } else {
            None
        }
    }
    pub fn scaled_add(&self, other: &MatrixFull<f64>,scaled_factor: f64) -> Option<MatrixFull<f64>> {
        if self.check_shape(other) {
            let mut new_tensors: MatrixFull<f64> = self.clone();
            //new_tensors.data.par_iter_mut()
            //    .zip(other.data.par_iter())
            //    .map(|(c,p)| {*c +=p*scaled_factor});
            (0..new_tensors.data.len()).into_iter().for_each(|i| {
                new_tensors.data[i] += scaled_factor*other.data[i].clone();
            });
            Some(new_tensors)
        } else {
            None
        }
    }
    #[inline]
    pub fn sub(&self, other: &MatrixFull<f64>) -> Option<MatrixFull<f64>> {
        if self.check_shape(other) {
            let mut new_tensors: MatrixFull<f64> = self.clone();
            //new_tensors.data.par_iter_mut()
            //    .zip(other.data.par_iter())
            //    .map(|(c,p)| {*c -=p});
            (0..new_tensors.data.len()).into_iter().for_each(|i| {
                new_tensors.data[i] -= other.data[i];
            });
            Some(new_tensors)
        } else {
            None
        }
    }
    #[inline]
    pub fn self_general_add(&mut self, bm: &MatrixFull<f64>,a: f64, b:f64) {
        /// A = a*A + b*B
        if self.check_shape(bm) {
            //let mut new_tensors: MatrixFull<f64> = self.clone();
            self.data.par_iter_mut()
                .zip(bm.data.par_iter())
                .for_each(|(c,p)| {*c =*c*a+p*b});
        } else {
            panic!("Error: Shape inconsistency happens when plus two matrices");
        }
    }
    #[inline]
    pub fn self_multiple(&mut self, a: f64) {
        /// A = a*A
        self.data.iter_mut()
            .for_each(|c| {*c =*c*a});
    }
    #[inline]
    pub fn par_self_multiple(&mut self, a: f64) {
        /// A = a*A
        self.data.par_iter_mut()
            .for_each(|c| {*c =*c*a});
    }
    #[inline]
    pub fn self_add(&mut self, bm: &MatrixFull<f64>) {
        /// A = A + B 
        if self.check_shape(bm) {
            self.data.iter_mut()
                .zip(bm.data.iter())
                .for_each(|(c,p)| {*c +=p});
        } else {
            panic!("Error: Shape inconsistency happens when plus two matrices");
        }
    }
    #[inline]
    pub fn par_self_add(&mut self, bm: &MatrixFull<f64>) {
        /// A = A + B 
        if self.check_shape(bm) {
            self.data.par_iter_mut()
                .zip(bm.data.par_iter())
                .for_each(|(c,p)| {*c +=p});
        } else {
            panic!("Error: Shape inconsistency happens when plus two matrices");
        }
    }
    #[inline]
    pub fn self_scaled_add(&mut self, bm: &MatrixFull<f64>,b: f64) {
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
    pub fn par_self_scaled_add(&mut self, bm: &MatrixFull<f64>,b: f64) {
        /// A = A + b*B
        if self.check_shape(bm) {
            self.data.par_iter_mut()
                .zip(bm.data.par_iter())
                .for_each(|(c,p)| {*c +=*p*b});
        } else {
            panic!("Error: Shape inconsistency happens when plus two matrices");
        }
    }
    pub fn self_sub(&mut self, bm: &MatrixFull<f64>) {
        /// A = A - B
        self.self_scaled_add(bm, -1.0)
    }
    pub fn par_self_sub(&mut self, bm: &MatrixFull<f64>) {
        /// A = A - B
        self.par_self_scaled_add(bm, -1.0)
    }

    pub fn lapack_dgemm(&mut self, a: &mut MatrixFull<f64>, b: &mut MatrixFull<f64>, opa: char, opb: char, alpha: f64, beta: f64) {
        self.to_matrixfullslicemut().lapack_dgemm(
            &mut a.to_matrixfullslicemut(),
            &mut b.to_matrixfullslicemut(), 
            opa, 
            opb, 
            alpha, 
            beta);
    }

    pub fn ddot(&mut self, b: &mut MatrixFull<f64>) -> Option<MatrixFull<f64>> {
        if let Some(tmp_mat) = self.to_matrixfullslicemut().ddot(&mut b.to_matrixfullslicemut()) {
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

#[derive(Debug,PartialEq)]
pub struct MatrixFullSliceMut<'a,T> {
    pub size : &'a [usize],
    pub indicing: &'a [usize],
    pub data : &'a mut [T]
}

impl <'a, T: Copy + Clone + Display + Send + Sync> MatrixFullSliceMut<'a, T> {
    //pub fn from_slice(sl: &[T]) -> MatrixFullSlice<'a, T> {
    //    MatrixFullSliceMut {
    //        size
    //    }
    //}
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
    pub fn get_slices_mut(& mut self, x: Range<usize>, y: Range<usize>) -> Flatten<IntoIter<&mut [T]>> {
        let mut tmp_slices: Vec<&mut [T]> = vec![];
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
}



#[derive(Clone,Debug,PartialEq)]
pub struct MatrixFullSlice<'a,T:Clone+Display> {
    pub size : &'a [usize],
    pub indicing: &'a [usize],
    pub data : &'a [T]
}

impl <'a, T: Copy + Clone + Display + Send + Sync> MatrixFullSlice<'a, T> {
    #[inline]
    pub fn iter_j(&self, j: usize) -> std::slice::Iter<T> {
        let start = self.size[0]*j;
        let end = start + self.size[0];
        self.data[start..end].iter()
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
    pub fn iter_slice_x(&self, y: usize) -> std::slice::Iter<T> {
        let start = self.indicing[1]*y;
        let end = self.indicing[1]*(y+1);
        self.data[start..end].iter()
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
            trans_mat.get_slices_mut(i..i+1,0..x_len).zip(c)
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
            trans_mat.get_slices_mut(i..i+1,0..x_len).zip(c)
            .for_each(|(to,from)| {*to = *from})
        });
        trans_mat
    }
}


#[derive(Clone,Debug,PartialEq)]
pub struct MatrixUpper<T:Clone+Display> {
    pub size :  usize,
    pub data : Vec<T>
}

impl <T: Copy + Clone + Display + Send + Sync> MatrixUpper<T> {
    pub fn empty() -> MatrixUpper<T> {
        unsafe{MatrixUpper::from_vec_unchecked(0,Vec::new())}
    }
    pub fn new(size: usize, new_default: T) -> MatrixUpper<T> {
        MatrixUpper {
            size,
            data: vec![new_default.clone(); size]
        }
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
    #[inline]
    pub fn to_slice_mut(&mut self) -> &mut [T] {
        &mut self.data[..]
    }
    #[inline]
    pub fn to_matrixupperslicemut(&mut self) -> MatrixUpperSliceMut<T> {
        MatrixUpperSliceMut {
            size: &self.size,
            data: &mut self.data[..],
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
pub struct MatrixUpperSliceMut<'a,T:Clone+Display> {
    pub size : &'a usize,
    //pub indicing: &'a usize,
    pub data : &'a mut[T]
}


#[derive(Clone,Debug,PartialEq)]
pub struct MatrixUpperSlice<'a,T:Clone+Display> {
    pub size : &'a usize,
    //pub indicing: &'a usize,
    pub data : &'a [T]
}

//impl<'a, T:Clone+Display> MatrixUpperSliceMut<'a,T> {
//    pub fn get_slices_mut(&mut self, x:Range<usize>, y:Range<usize>) -> Flatten<IntoIter<&mut [T]>> {
//        let mut tmp_slices: Vec<&mut [T]> = vec![];
//        let dd = self.data.split_at_mut(0).1;
//        y.fold((dd,0_usize),|(ee,offset),y| {
//        });
//    }
//}

