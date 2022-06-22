#![allow(unused)]
extern crate blas;
extern crate blas_src;
extern crate lapack;
extern crate lapack_src;

use std::fmt::Display;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use crate::{RecFn,SAFE_MINIMUM};
use crate::tensors::{MatFormat,Tensors};
use crate::index::Indexing;

use lapack::{dsyev,dspevx,dspgvx};
use blas::dgemm;
#[derive(Debug)]
pub struct TensorsSliceMut<'a, T: Clone> {
    pub store_format : &'a MatFormat,
    pub rank: usize,
    pub size : &'a [usize],
    pub indicing: &'a [usize],
    pub data : &'a mut [T],
}

impl <'a, T: Clone> Indexing for TensorsSliceMut<'a, T> {
    #[inline]
    fn indexing(&self, positions: &[usize]) -> usize 
    {
        let mut p_start: usize = 0;
        match self.store_format {
            MatFormat::Full => {
                let mut i:usize = 0;
                for i in 0..self.rank {
                    p_start += self.indicing[i]*positions[i];
                }
            },
            MatFormat::Lower => {
                let mut i: usize = 0;
                p_start = if positions[0]>=positions[1] {
                    (2*self.size[0]-positions[1]-1)*positions[1]/2 + positions[0]
                } else {
                    (2*self.size[0]-positions[0]-1)*positions[0]/2 + positions[1]
                };
                for i in 2..self.rank-2 {
                    p_start += self.indicing[i]*positions[i];
                }
            },
            MatFormat::Upper => {
                let mut i: usize = 0;
                p_start = if positions[0]<=positions[1] {
                    (positions[1]+1)*positions[1]/2 + positions[0]
                } else {
                    (positions[0]+1)*positions[0]/2 + positions[1]
                };
                for i in 2..self.rank-2 {
                    p_start += self.indicing[i]*positions[i];
                }
            }
        }
        p_start
    }
    #[inline]
    fn indexing_mat(&self, positions: &[usize]) -> usize 
    {
        let mut p_start: usize = 0;
        match self.store_format {
            MatFormat::Full => {
                let mut i:usize = 0;
                for i in 0..self.rank {
                    p_start += self.indicing[i]*positions[i];
                }
            },
            MatFormat::Lower => {
                let mut i: usize = 0;
                p_start = (2*self.size[0]-positions[1]-1)*positions[1]/2 + positions[0];
            },
            MatFormat::Upper => {
                let mut i: usize = 0;
                p_start = (positions[1]+1)*positions[1]/2 + positions[0];
            }
        }
        p_start
    }
    #[inline]
    fn reverse_indexing(&self, positions: usize) -> Vec<usize> 
        where T: Clone
    {
        let mut pos: Vec<usize> = vec![0;self.size.len()];
        let mut rest = positions;
        match self.store_format {
            MatFormat::Full => {
                (0..self.size.len()).rev().for_each(|i| {
                    pos[i] = rest/self.indicing[i];
                    rest = rest % self.indicing[i];
                })
            },
            MatFormat::Lower => {
                // according the definition of lower tensor, the first and second ranks have the same lenght
                (2..self.size.len()).rev().for_each(|i|{
                    pos[i] = rest/self.indicing[i];
                    rest = rest % self.indicing[i];
                });
                let index_pattern = RecFn(Box::new(|func: &RecFn<i32>, n: (i32,i32)| -> (i32,i32) {
                    match n.0 {
                        i if i<0 => {
                            (0,n.1+1)
                        },
                        0=>(0,n.1),
                        _ => {func.call(func,(n.0-n.1, n.1-1))}
                    }
                }));
                let (i,j) = index_pattern.call(&index_pattern,(rest as i32,self.size[0] as i32));
                let i = rest - (self.size[0]+j as usize -1)*(self.size[0]-j as usize)/2;
                let j = self.size[0] - j as usize;
                pos[0] = i;
                pos[1] = j as usize;
            },
            MatFormat::Upper => {
                // according the definition of lower tensor, the first and second ranks have the same lenght
                (2..self.size.len()).rev().for_each(|i|{
                    pos[i] = rest/self.indicing[i];
                    let rest = rest % self.indicing[i];
                });
                let index_pattern = RecFn(Box::new(|func: &RecFn<i32>, n: (i32,i32)| -> (i32,i32) {
                    match n.0-n.1 {
                        i if i<0 => {
                            (0,n.1-1)
                        },
                        0 => (0,n.1),
                        _ => {func.call(func,(n.0-n.1, n.1+1))}
                    }
                }));
                let (i,j) = index_pattern.call(&index_pattern,(rest as i32,0));
                let tmp_val = (j*(j+1)/2);
                let i = rest - tmp_val as usize;
                pos[0] = i;
                pos[1] = j as usize;
            },
        }
        pos
    }
}

impl <'a, T> TensorsSliceMut<'a, T> 
    where T: Clone + Display
    {
    pub fn formated_output(&self, n_len: usize, mat_form: String) {
        let mat_format = if mat_form.to_lowercase()==String::from("full") {MatFormat::Full
        } else if mat_form.to_lowercase()==String::from("upper") {MatFormat::Upper
        } else if mat_form.to_lowercase()==String::from("lower") {MatFormat::Lower
        } else {
            panic!("Error in determing the layout format of the matrix: {}", mat_form)
        };
        if self.rank!=2 {panic!("At present, the formated output is only available for the 2-D matrix")};
        let n_row = self.size[0];
        let n_column = self.size[1];
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
    #[inline]
    pub fn get(&self, position: &[usize]) -> Option<&T>{
        let index: usize = self.indexing(position);
        self.data.get(index)
        //if position.len()!=self.rank {
        //    println!("Error: it is a {}-D tensor, which cannot be indexed by a position of {:?}", self.rank,position);
        //    None
        //} else {
        //    let index: usize = self.indexing(position);
        //    Some(self.data[index].clone())
        //}
    }
    #[inline]
    pub fn set(&mut self, position: &[usize], new_data: T){
        let index: usize = self.indexing(position);
        if let Some(tmp_value) = self.data.get_mut(index) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
        //if position.len()!=self.rank {
        //    panic!("Error: it is a {}-D tensor, which cannot be indexed by a position of {:?}", self.rank,position);
        //} else {
        //    let index: usize = self.indexing(position);
        //    self.data[index] = new_data;
        //}
    }
    #[inline]
    pub fn set_mat(&mut self, position: &[usize], new_data: T){
        /// In order to guarantee the performance, the position should be in the correct order for different stored formats,
        /// which will not be checked in the indexing_mat()
        let index: usize = self.indexing_mat(position);
        if let Some(tmp_value) = self.data.get_mut(index) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    pub fn get_mat_slice(&mut self, position: &[usize],len: usize) -> Option<&mut [T]>{
        /// In order to guarantee the performance, the position should be in the correct order for different stored formats,
        /// which will not be checked in the indexing_mat()
        let index: usize = self.indexing_mat(position);
        //self.data.get(index)
        Some(&mut self.data[index..index+len])
    }
}
impl <'a> TensorsSliceMut<'a, f64> {
    pub fn dot(&mut self, b: &mut TensorsSliceMut<f64>) -> Option<Tensors<f64>> {
        /// for self a => a*b
        match self.store_format {
            MatFormat::Full => {
                if self.size[1]!=b.size[0] || self.rank!=2 || b.rank!=2 {
                    None
                } else {
                    let (m, n, k) = (self.size[0],b.size[1],self.size[1]);
                    let mut c: Vec<f64> = vec![0.0; (m*n)];
                    unsafe {
                        dgemm(b'N',b'N',m as i32,n as i32,k as i32,1.0,self.data,m as i32,b.data,k as i32,1.0,&mut c,m as i32);
                    }
                    Some(Tensors::from_vec('F',vec![m,n], c))
                }
            },
            MatFormat::Lower => {
                println!("Error: At present, only the matrix stored in full format can be operated");
                None
            },
            MatFormat::Upper => {
                println!("Error: At present, only the matrix stored in full format can be operated");
                None
            }
        }
    }
    pub fn multiple(&mut self, scaled_factor: f64) {
        /// for self a => a*scaled_factor
        (0..self.data.len()).into_iter().for_each(|i| {
            self.data[i] *= scaled_factor;
        });
    }
    pub fn diagonalize(&mut self) -> Option<(Tensors<f64>,Vec<f64>,i32)> {
        /// eigenvalues and eigenvectors of  self a
        match self.store_format {
            MatFormat::Full => {
                if self.size[1]!=self.size[0] || self.rank!=2 {
                    None
                } else {
                    let n = self.size[0] as i32;
                    let mut a: Vec<f64> = self.data.to_vec().clone();
                    let mut w: Vec<f64> = vec![0.0;self.size[0]];
                    let mut work: Vec<f64> = vec![0.0;4*self.size[0]];
                    let lwork = 4*self.size[0] as i32;
                    let mut info = 0;
                    unsafe {
                        dsyev(b'V',b'L',n,&mut a, n, &mut w, &mut work, lwork, &mut info);
                    }
                    if info!=0 {
                        panic!("Error in diagonalizing the matrix");
                    }
                    let eigenvectors = Tensors::from_vec('F',vec![self.size[0],self.size[0]], a);
                    //let eigenvalues = Tensors::from_vec(String::from("full"),vec![self.size[0]], w);
                    Some((eigenvectors, w,n))
                }
            },
            MatFormat::Lower => {
                if self.size[1]!=self.size[0] || self.rank!=2 {
                    None
                } else {
                    let n = self.size[0] as i32;
                    let mut a: Vec<f64> = self.data.to_vec().clone();
                    let mut w: Vec<f64> = vec![0.0;self.size[0]];
                    let mut z: Vec<f64> = vec![0.0;self.size[0]*self.size[0]];
                    let mut work: Vec<f64> = vec![0.0;8*self.size[0]];
                    let mut iwork: Vec<i32> = vec![0;5*self.size[0]];
                    let mut ifail: Vec<i32> = vec![0;self.size[0]];
                    let mut n_found:i32 = 0;
                    let lwork = 4*self.size[0] as i32;
                    let mut info = 0;
                    unsafe {
                        dspevx(b'V',b'A',b'L',n,&mut a, 0.0_f64, 0.0_f64,0,0,
                               SAFE_MINIMUM,&mut n_found, &mut w, &mut z, n, &mut work, &mut iwork, &mut ifail,&mut info);
                    }
                    let eigenvectors = Tensors::from_vec('F',vec![self.size[0],self.size[0]], z);
                    //let eigenvalues = Tensors::from_vec(String::from("full"),vec![self.size[0]], w);
                    Some((eigenvectors, w, n_found))
                }
            },
            MatFormat::Upper => {
                if self.size[1]!=self.size[0] || self.rank!=2 {
                    None
                } else {
                    let n = self.size[0] as i32;
                    let mut a: Vec<f64> = self.data.to_vec().clone();
                    let mut w: Vec<f64> = vec![0.0;self.size[0]];
                    let mut z: Vec<f64> = vec![0.0;self.size[0]*self.size[0]];
                    let mut work: Vec<f64> = vec![0.0;8*self.size[0]];
                    let mut iwork: Vec<i32> = vec![0;5*self.size[0]];
                    let mut ifail: Vec<i32> = vec![0;self.size[0]];
                    let mut n_found:i32 = 0;
                    let lwork = 4*self.size[0] as i32;
                    let mut info = 0;
                    unsafe {
                        dspevx(b'V',b'A',b'U',n,&mut a, 0.0_f64, 0.0_f64,0,0,
                               SAFE_MINIMUM,&mut n_found, &mut w, &mut z, n, &mut work, &mut iwork, &mut ifail,&mut info);
                    }
                    let eigenvectors = Tensors::from_vec('F',vec![self.size[0],self.size[0]], z);
                    //let eigenvalues = Tensors::from_vec(String::from("full"),vec![self.size[0]], w);
                    Some((eigenvectors, w, n_found))
                }
            }
        }
    }
    pub fn lapack_solver(&mut self,ovlp:TensorsSliceMut<f64>,num_orb:usize) -> Option<(Tensors<f64>,Vec<f64>)> {
        ///solve A*x=(lambda)*B*x
        let mut itype: i32 = 1;
        let mut n = self.size[0] as i32;
        let mut a = self.data.to_vec().clone();
        let mut b = ovlp.data.to_vec().clone();
        //a[4] = -8.653240731677;
        //println!("nbas:{},num_orb: {},a: {:?}",n,num_orb,&a);
        //println!("b: {:?}",&b);
        let mut m = 0;
        let mut w: Vec<f64> = vec![0.0;self.size[0]];
        let mut z: Vec<f64> = vec![0.0;self.size[0]*self.size[0]];
        let mut work: Vec<f64> = vec![0.0;8*self.size[0]];
        let mut iwork:Vec<i32> = vec![0;5*self.size[0]];
        let mut ifail:Vec<i32> = vec![0;self.size[0]];
        let mut info: i32  = 0;
        unsafe{
            dspgvx(&[itype],
                b'V',
                b'I',
                b'U',
                n,
                &mut a,
                &mut b,
                0.0,
                0.0,
                1,
                num_orb as i32,
                SAFE_MINIMUM,
                &mut m,
                &mut w,
                &mut z,
                n,
                &mut work,
                &mut iwork,
                &mut ifail,
                &mut info);
        }
        //println!("{:?}",&w);
        if info < 0 {
            panic!("Error:: Generalized eigenvalue problem solver dspgvx()\n The -{}th argument in dspgvx() has an illegal value. Check", info);
        } else if info > n {
            panic!("Error:: Generalized eigenvalue problem solver dspgvx()\n The leading minor of order {} of ovlp is not positive definite", info-n);
        } else if info > 0 {
            panic!("Error:: Generalized eigenvalue problem solver dspgvx()\n {} vectors failed to converge", info);
        }
        if m!=num_orb as i32 {
            panic!("Error:: The number of outcoming eigenvectors {} is unequal to the orbital number {}", m, num_orb);
        }
        let eigenvectors = Tensors::from_vec('F',vec![n as usize,m as usize],z);
        //let eigenvalues = Tensors::from_vec("full".to_string(),vec![n as usize],w);
        Some((eigenvectors, w))
    }
}

pub struct TensorsSlice<'a, T: Clone> {
    pub store_format : &'a MatFormat,
    pub rank: usize,
    pub size : &'a [usize],
    pub indicing: &'a [usize],
    pub data : &'a [T],
}

impl <'a, T> TensorsSlice<'a, T> 
    where T: Clone + Display
    {
    #[inline]
    pub fn get(&self, position: &[usize]) -> Option<&T>{
        let index: usize = self.indexing(position);
        self.data.get(index)
    }
    #[inline]
    pub fn get_mat(&self, position: &[usize]) -> Option<&T>{
        /// In order to guarantee the performance, the position should be in the correct order for different stored formats,
        /// which will not be checked in the indexing_mat()
        let index: usize = self.indexing_mat(position);
        self.data.get(index)
    }
    #[inline]
    pub fn get_mat_slice(&self, position: &[usize],len: usize) -> Option<&[T]>{
        /// In order to guarantee the performance, the position should be in the correct order for different stored formats,
        /// which will not be checked in the indexing_mat()
        let index: usize = self.indexing_mat(position);
        //self.data.get(index)
        Some(&self.data[index..index+len])
    }
    #[inline]
    pub fn get_mat_slice_hp(&self, start: usize,len: usize) -> &[T]{
        //let index: usize = self.indexing_mat(position);
        //self.data.get(index)
        &self.data[start..start+len]
    }
}
impl <'a, T: Clone> Indexing for TensorsSlice<'a, T> {
    #[inline]
    fn indexing(&self, positions: &[usize]) -> usize 
    // The general indexing mode, which, however, is very time consuming
        where T: Clone
    {
        let mut p_start: usize = 0;
        match self.store_format {
            MatFormat::Full => {
                for i in 0..self.rank {
                    p_start += self.indicing[i]*positions[i];
                }
            },
            MatFormat::Lower => {
                p_start = if positions[0]>=positions[1] {
                    (2*self.size[0]-positions[1]-1)*positions[1]/2 + positions[0]
                } else {
                    (2*self.size[0]-positions[0]-1)*positions[0]/2 + positions[1]
                };
                for i in 2..self.rank-2 {
                    p_start += self.indicing[i]*positions[i];
                }
            },
            MatFormat::Upper => {
                p_start = if positions[0]<=positions[1] {
                    (positions[1]+1)*positions[1]/2 + positions[0]
                } else {
                    (positions[0]+1)*positions[0]/2 + positions[1]
                };
                for i in 2..self.rank-2 {
                    p_start += self.indicing[i]*positions[i];
                }
            }
        }
        p_start
    }
    #[inline]
    fn indexing_mat(&self, positions: &[usize]) -> usize 
        where T: Clone
    {
        //let mut p_start: usize = 0;
        match self.store_format {
            MatFormat::Upper => {
                (positions[1]+1)*positions[1]/2 + positions[0]
            },
            MatFormat::Full => {
                self.indicing[0]*positions[0] + self.indicing[1]*positions[1]
            },
            MatFormat::Lower => {
                (2*self.size[0]-positions[1]-1)*positions[1]/2 + positions[0]
            }
        }
        //p_start
    }
    #[inline]
    fn reverse_indexing(&self, positions: usize) -> Vec<usize> 
        where T: Clone
    // The general reversed indexing mode, which, however, is very time consuming
    {
        let mut pos: Vec<usize> = vec![0;self.size.len()];
        let mut rest = positions;
        match self.store_format {
            MatFormat::Full => {
                (0..self.size.len()).rev().for_each(|i| {
                    pos[i] = rest/self.indicing[i];
                    rest = rest % self.indicing[i];
                })
            },
            MatFormat::Lower => {
                // according the definition of lower tensor, the first and second ranks have the same lenght
                (2..self.size.len()).rev().for_each(|i|{
                    pos[i] = rest/self.indicing[i];
                    rest = rest % self.indicing[i];
                });
                let index_pattern = RecFn(Box::new(|func: &RecFn<i32>, n: (i32,i32)| -> (i32,i32) {
                    match n.0 {
                        i if i<0 => {
                            (0,n.1+1)
                        },
                        0=>(0,n.1),
                        _ => {func.call(func,(n.0-n.1, n.1-1))}
                    }
                }));
                let (i,j) = index_pattern.call(&index_pattern,(rest as i32,self.size[0] as i32));
                let i = rest - (self.size[0]+j as usize -1)*(self.size[0]-j as usize)/2;
                let j = self.size[0] - j as usize;
                pos[0] = i;
                pos[1] = j as usize;
            },
            MatFormat::Upper => {
                // according the definition of lower tensor, the first and second ranks have the same lenght
                (2..self.size.len()).rev().for_each(|i|{
                    pos[i] = rest/self.indicing[i];
                    let rest = rest % self.indicing[i];
                });
                let index_pattern = RecFn(Box::new(|func: &RecFn<i32>, n: (i32,i32)| -> (i32,i32) {
                    match n.0-n.1 {
                        i if i<0 => {
                            (0,n.1-1)
                        },
                        0 => (0,n.1),
                        _ => {func.call(func,(n.0-n.1, n.1+1))}
                    }
                }));
                let (i,j) = index_pattern.call(&index_pattern,(rest as i32,0));
                let tmp_val = (j*(j+1)/2);
                let i = rest - tmp_val as usize;
                pos[0] = i;
                pos[1] = j as usize;
            },
        }
        pos
    }
}