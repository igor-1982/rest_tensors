#![allow(unused)]
extern crate blas;
extern crate blas_src;
extern crate lapack;
extern crate lapack_src;

use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use lapack::{dsyev,dspevx,dspgvx};
use blas::dgemm;
mod tensors_slice;
mod eri;
mod index;
mod tensors;
mod tensor_basic_operation;
mod matrix;
mod matrix_blas_lapack;
pub mod ri;
//use typenum::{U1,U2,U3,U4};
use crate::tensors_slice::{TensorsSliceMut,TensorsSlice};
//use itertools::iproduct;


pub use crate::tensor_basic_operation::*;
pub use crate::tensors::*;
pub use crate::eri::*;
pub use crate::matrix::*;
pub use crate::ri::*;

#[derive(Clone,Debug,PartialEq)]
pub struct Tensors4D<T:Clone+Display,D> {
    /// Coloum-major Tensors with the rank of 4 at most,
    /// designed for quantum chemistry calculations specifically.
    pub data : Vec<T>,
    pub size : [usize;4],
    pub indicing: [usize;4],
    pub rank : D,
    //pub store_format : MatFormat,
    //pub size : Vec<usize>,
    //pub indicing: [usize;4],
}


const SAFE_MINIMUM:f64 = 10E-12;

//recursive wrapper
struct RecFn<T>(Box<dyn Fn(&RecFn<T>,(T,T)) -> (T,T)>);
impl<T> RecFn<T> {
    fn call(&self, f: &RecFn<T>, n: (T,T)) -> (T,T) {
        (self.0(f,n))
    }
}



#[cfg(test)]
mod tests {
    use itertools::{iproduct, Itertools};
    use libc::access;

    use crate::{index::Indexing, tensors::Tensors, MatrixFull, RIFull, MatrixUpper, print_vec};
    //#[test]
    //fn test_matrix_index() {
    //    let size_a:Vec<usize>=vec![3,3];
    //    let mut tmp_a = vec![
    //        3.0,1.0,1.0,
    //        1.0,3.0,1.0,
    //        1.0,1.0,3.0];
    //    let mut my_a = Tensors::from_vec('F', size_a, tmp_a);
    //    println!("{}",my_a[(0usize,0usize)]);
    //}
    #[test]
    fn test_operator_overloading() {
        let a = MatrixFull::from_vec([3,3],vec![0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]).unwrap();
        let b = MatrixFull::from_vec([3,3],vec![8.0,3.0,4.0,2.0,6.0,3.0,9.0,16.0,6.0]).unwrap();
        println!("a:{:?}, b:{:?}", a,b);
        let c = a+b;
        println!("c:{:?}", c);
        println!("c:[{:8.4},{:8.4},{:8.4}]", c[[0,0]],c[[0,1]],c[[0,2]]);
        println!("c:[{:8.4},{:8.4},{:8.4}]", c[[1,0]],c[[1,1]],c[[1,2]]);
        println!("c:[{:8.4},{:8.4},{:8.4}]", c[[2,0]],c[[2,1]],c[[2,2]]);
        let a = MatrixFull::new([3,3],1);
        let b = MatrixFull::new([2,2],2);
        let c = b+a;
        println!("c:{:?}", c);
        let a = MatrixFull::from_vec([2,2],vec![3,2,1,3]).unwrap();
        let b = MatrixFull::from_vec([2,2],vec![2,3,3,1]).unwrap();
        let c = b-a;
        println!("c:{:?}", &c);
        println!("c:[{},{}]", c[[0,0]],c[[1,0]]);
        println!("c:[{},{}]", c[[0,1]],c[[1,1]]);
    }
    #[test]
    fn test_upper_matrix() {
        let size:Vec<usize>=vec![11,11];
        let mut tmp_v = vec![0.0;(size[1]+1)*size[1]/2];
        (0..tmp_v.len()).into_iter().for_each(|i| {
            tmp_v[i] = i as f64;
        });
        //let size = vec![11,11];
        let mut my_mat = Tensors::from_vec('U', size, tmp_v);
        let mut tmp_v = if let Some(fvalue) = my_mat.get_mut(&[2,6]) {
            fvalue
        } else {
            panic!("Error in getting a matrix element");
        };
        *tmp_v = 100.0;
        println!("index: {}",&my_mat.indexing(&[2,6]));
        println!("pos:   {:?}",&my_mat.reverse_indexing(0));
        println!("value: {:?}",&my_mat.get(&[2,6]).unwrap());
        //my_mat.formated_output(5, String::from("upper"));
        //assert_eq!(my_mat.get(&[2,6]).unwrap(), 100.0);
    }
    #[test]
    fn test_upper_tensor() {
        let size:Vec<usize>=vec![11,11,2];
        let mut tmp_v = vec![0.0;(size[0]+1)*size[1]/2*size[2]];
        (0..tmp_v.len()).into_iter().for_each(|i| {
            tmp_v[i] = i as f64;
        });
        let mut my_mat = Tensors::from_vec('U', size, tmp_v);
        let mut tmp_v = if let Some(fvalue) = my_mat.get_mut(&[2,6,1]) {
            fvalue
        } else {
            panic!("Error in getting a matrix element");
        };
        *tmp_v = 100.0;
        println!("{:?}",&my_mat);
        //my_mat.formated_output(5, String::from("full"));
        let mut tmp_rd_mat = my_mat.get_reducing_tensor(1).unwrap();
        tmp_rd_mat.formated_output(5, String::from("upper"));
        tmp_rd_mat.set(&[2,6],50.0);
        tmp_rd_mat.formated_output(5, String::from("upper"));
        assert_eq!(*my_mat.get(&[2,6,1]).unwrap(), 50.0);
    }
    #[test]
    fn test_dgemm() {
        let size_a:Vec<usize>=vec![2,3];
        let mut tmp_a = vec![
            1.0,4.0,
            2.0,5.0,
            3.0,6.0];
        let size_b:Vec<usize>=vec![3,4];
        let mut tmp_b = vec![
            1.0,5.0, 9.0,
            2.0,6.0,10.0,
            3.0,7.0,11.0,
            4.0,8.0,11.0];
        let mut my_a = Tensors::from_vec('F', size_a, tmp_a);
        let mut my_b = Tensors::from_vec('F', size_b, tmp_b);
        let mut my_c = my_a.dot(&mut my_b).unwrap();
        //my_c.formated_oupput
        my_c.formated_output(5,String::from("full"));
        println!("{:?}", my_c);
        //assert_eq!(my_c.get(&[2,6,1]).unwrap(), 50.0);
    }
    #[test]
    fn test_diagonalize() {
        let size_a:Vec<usize>=vec![3,3];
        let mut tmp_a = vec![
            3.0,1.0,1.0,
            1.0,3.0,1.0,
            1.0,1.0,3.0];
        let mut my_a = Tensors::from_vec('F', size_a, tmp_a);
        let (mut eigenvectors,mut eigenvalues,mut n) = my_a.diagonalize().unwrap();
        my_a.formated_output(5,String::from("full"));
        println!("eigenvalues: {:?}",eigenvalues);
        eigenvectors.formated_output(5,String::from("full"));
        let mut my_b = my_a.dot(&mut eigenvectors).unwrap();
        my_b.formated_output(5,String::from("full"));
        let mut eigenvectors_trans = eigenvectors.transpose().unwrap();
        let mut my_c = eigenvectors_trans.dot(&mut my_b).unwrap();
        my_c.formated_output(5,String::from("full"));

        let mut tmp_a = vec![
            3.0,1.0,3.0,1.0,1.0,3.0
            ];
        let size_a:Vec<usize>=vec![3,3];
        let mut my_a = Tensors::from_vec('F', size_a, tmp_a);
        let (mut full_eigenvectors,mut eigenvalues,mut n) = my_a.diagonalize().unwrap();
        println!("{:?}",eigenvalues);
        full_eigenvectors.formated_output(5,String::from("full"));
        //let mut full_eigenvectors = eigenvectors.copy(String::from("full")).unwrap();
        let mut full_my_a = my_a.duplicate('F').unwrap();
        full_my_a.formated_output(5,String::from("full"));
        //full_eigenvectors.formated_output(5,String::from("full"));
        let mut my_b = full_my_a.dot(&mut full_eigenvectors).unwrap();
        let mut full_eigenvectors_trans = full_eigenvectors.transpose().unwrap();
        let mut my_c = full_eigenvectors_trans.dot(&mut my_b).unwrap();
        my_c.formated_output(5,String::from("full"));
    }
    #[test]
    fn test_slice_concat() {
        let mut orig_a = vec![1,2,3,4,5,6,7];
        let mut orig_b = vec![10,12,15,27,31,3,1];
        let mut a = &mut orig_a[2..5];
        let mut b = &mut orig_b[2..5];
        let mut c = vec![a,b].into_iter().flatten();
        c.for_each(|i| {*i = *i+2});

        println!("{:?}", orig_a);
        println!("{:?}", orig_b);

        let dd = 2..6;
        println!("{},{},{}",dd.start, dd.end, dd.len());
        println!("{},{},{}",dd.start, dd.end, dd.len());
        dd.for_each(|i| {println!("{}",i)});
        
        //c.enumerate().for_each(|i| {
        //    println!{"index: {}, value: {}", i.0,i.1};
        //})
    }
    #[test]
    fn test_ri() {
        let orig_a = vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18];
        let mut tmp_ri = RIFull::from_vec([3,3,2],orig_a).unwrap();
        println!("{:?}",tmp_ri);
        println!("{}",tmp_ri[[0,0,0]]);
        println!("{}",tmp_ri[[0,0,1]]);
        println!("{}",tmp_ri[[2,2,1]]);
        tmp_ri[[0,0,0]] = 100;
        println!("{}",tmp_ri[[0,0,0]]);
        println!("{:?}",tmp_ri);

        //now test get_slices
        let dd = tmp_ri.get_slices(0..2, 1..3, 0..2);
        dd.enumerate().for_each(|i| {
            println!("{},{}",i.0,i.1)}
        );
        let dd = tmp_ri.get_slices_mut(0..2, 1..3, 0..2);
        dd.enumerate().for_each(|i| {
            *i.1 += 2}
        );
        println!("{:?}",&tmp_ri.data);

        iproduct!(2..5,1..4,0..3).for_each(|f| {
            println!("x:{}, y:{},z:{}", f.2,f.1,f.0);
        });

        let orig_b = vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
        let mut tmp_aux = MatrixFull::from_vec([4,4],orig_b).unwrap();
        let dd = tmp_aux.get_slices_mut(0..1, 2..4);
        dd.enumerate().for_each(|i| {println!("{},{}",i.0,i.1)});


        //(0..5).into_iter().for_each(|i| {
        //    (0..5).into_iter().for_each(|j| {
        //        
        //    })
        //})

        //let orig_b = vec![1,2,3,4,5,6];
        //let mut tmp_mat = MatrixUpper::from_vec(6,orig_b).unwrap();
        //println!("{:?}",tmp_mat[1]);
        //let orig_c = vec![1,2,3,4,5,6];
        //println!("{:?}", &orig_c[1..3]);


    }
    #[test]
    fn matfull_inverse_and_power() {
        let orig_a = vec![1.0, 0.2350377623170771, 0.00000000000000014780661935396685, 0.0000000000000001230564920088275, 0.0, 0.05732075877050055, 0.05732075877577619, 0.05732075876606951, 0.2350377623170771, 1.0000000000000002, 0.0000000000000006843048497264658, -0.0000000000000006063573786851014, 0.0, 0.4899272978714807, 0.4899272978956163, 0.48992729785120903, 0.00000000000000014780661935396685, 0.0000000000000006843048497264658, 1.0000000000000004, -0.00000000000000000000000000000030065232611750355, 0.0, 0.43694556071760393, -0.14564693387985528, -0.14564891678026162, 0.0000000000000001230564920088275, -0.0000000000000006063573786851014, -0.00000000000000000000000000000030065232611750355, 1.0000000000000004, 0.0, -0.00000000000000019929707359479999, -0.3447708719127397, 0.367656510461097, 0.0, 0.0, 0.0, 0.0, 1.0000000000000004, 0.0, -0.22548046384235368, -0.1858400020910537, 0.05732075877050055, 0.4899272978714807, 0.43694556071760393, -0.00000000000000019929707359479999, 0.0, 1.0000000000000002, 0.20144562931480953, 0.20144477338087088, 0.05732075877577619, 0.4899272978956163, -0.14564693387985528, -0.3447708719127397, -0.22548046384235368, 0.20144562931480953, 1.0000000000000002, 0.20144477335123845, 0.05732075876606951, 0.48992729785120903, -0.14564891678026162, 0.367656510461097, -0.1858400020910537, 0.20144477338087088, 0.20144477335123845, 1.0000000000000002];

        let mut tmp_mat = MatrixFull::from_vec([8,8],orig_a.clone()).unwrap();
        println!("tmp_mat:");
        print_vec(&tmp_mat.data, tmp_mat.size[0]);

        let mut inv_tmp_mat = tmp_mat.lapack_inverse().unwrap();
        println!("inv_tmp_mat:");
        print_vec(&inv_tmp_mat.data, inv_tmp_mat.size[0]);

        //let mut tmp_mat_2 = MatrixFull::new([8,8],0.0);
        //tmp_mat_2.lapack_dgemm(&mut tmp_mat, &mut inv_tmp_mat, 'N', 'N', 1.0, 0.0);
        //println!("tmp_mat * inv_tmp_mat:");
        //print_vec(&tmp_mat_2.data, tmp_mat_2.size[0]);

        let mut inv_tmp_mat_2 = tmp_mat.lapack_power(-0.5, 10.0E-6).unwrap();
        println!("inv_tmp_mat:");
        print_vec(&inv_tmp_mat_2.data, inv_tmp_mat_2.size[0]);

    }
}


fn print_vec(buf: &Vec<f64>, len_per_line: usize) {
        buf.chunks(len_per_line).for_each(|value| {
            let mut tmp_str = String::new();
            value.into_iter().enumerate().for_each(|x| {
                if x.0 == 0 {
                    tmp_str = format!("{:16.8}",x.1);
                } else {
                    tmp_str = format!("{},{:16.8}",tmp_str,x.1);
                }
            });
            println!("{}",tmp_str);
        });
    }