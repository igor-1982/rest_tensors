use std::{iter::Flatten, vec::IntoIter, ops::Range};

use blas::dgemm;
use lapack::{dsyev, dspgvx, dspevx,dgetrf,dgetri};
use nalgebra::matrix;
use rayon::prelude::*;

use crate::{MatrixFullSliceMut, MatrixFull, SAFE_MINIMUM, MatrixUpperSliceMut, TensorSlice, TensorSliceMut, MatrixFullSlice, MatrixFullSliceMut2, BasicMatrix};


pub fn general_check_shape<'a, T,Q,P>(matr_a: &'a T, matr_b: &'a Q, opa: char, opb: char) -> bool 
where T: BasicMatrix<'a, P>,
      Q: BasicMatrix<'a, P>
{
    match (opa, opb) {
        ('N','N') => {
            matr_a.size().iter().zip(matr_b.size().iter()).fold(true, |flag, (a,b)| flag && *a==*b)
        },
        ('T','N') => {
            matr_a.size().iter().rev().zip(matr_b.size().iter()).fold(true, |flag, (a,b)| flag && *a==*b)
        },
        ('N','T') => {
            matr_a.size().iter().rev().zip(matr_b.size().iter()).fold(true, |flag, (a,b)| flag && *a==*b)
        },
        ('T','T') => {
            matr_a.size().iter().zip(matr_b.size().iter()).fold(true, |flag, (a,b)| flag && *a==*b)
        },
        _ => {false}

    }
}

/// # an efficient dgemm manipulation for the matrices equipped with the BasicMatrix trait:
///  matr_c[(range_row_c, range_column_c)] =
///      alpha * opa(matr_a[(range_row_a, range_column_a)])*opb(matr_b[(range_row_b, range_column_b)]) +
///      beta * matr_c[(range_row_c, range_column_c)]
/// ```
///    use rest_tensors::matrix::MatrixFull;
///    use rest_tensors::matrix::matrix_blas_lapack::_dgemm;
///    let matr_a = MatrixFull::from_vec([3,3], (1..10).map(|x| x as f64).collect::<Vec<f64>>()).unwrap();
///    //             | 1.0 | 4.0 | 7.0 |
///    // matr_a =    | 2.0 | 5.0 | 8.0 |
///    //             | 3.0 | 6.0 | 9.0 |
///    let matr_b = MatrixFull::from_vec([3,3], (6..15).map(|x| x as f64).collect::<Vec<f64>>()).unwrap();
///    //             | 6.0 | 9.0 |12.0 |
///    // matr_b =    | 7.0 |10.0 |13.0 |
///    //             | 8.0 |11.0 |14.0 |
///    let mut matr_c = MatrixFull::new([3,3], 2.0);
/// 
///    _dgemm(&matr_a, (1..3, 1..3), 'N',
///           &matr_b, (0..2, 0..2), 'N',
///           &mut matr_c, (1..3, 0..2), 
///          1.0, 1.0);
///    //             |  2.0 |  2.0 | 2.0 |
///    // matr_c =    | 88.0 |127.0 | 2.0 |
///    //             |101.0 |144.0 | 2.0 |
///    assert_eq!(matr_c.get_submatrix(1..3, 0..2).data(),vec![88.0,101.0,127.0,146.0]);
/// 
/// 
///    let mut matr_c = MatrixFull::new([3,3],2.0);
///    _dgemm(&matr_a,     (1..3, 1..2), 'T',
///           &matr_b,     (0..2, 0..2), 'N',
///           &mut matr_c, (1..2, 0..2), 
///          1.0, 1.0);
///    //             |  2.0 |  2.0 | 2.0 |
///    // matr_c =    | 74.0 |107.0 | 2.0 |
///    //             |  2.0 |  2.0 | 2.0 |
///    assert_eq!(matr_c.get_submatrix(1..2, 0..2).data(),vec![74.0,107.0]);
/// 
///    let mut matr_c = MatrixFull::new([3,3],2.0);
///    _dgemm(&matr_a,     (1..3, 0..2), 'N',
///           &matr_b,     (0..1, 0..2), 'T',
///           &mut matr_c, (0..2, 0..1), 
///          1.0, 1.0);
///    //             | 59.0 |  2.0 | 2.0 |
///    // matr_c =    | 74.0 |  2.0 | 2.0 |
///    //             |  2.0 |  2.0 | 2.0 |
///    assert_eq!(matr_c.get_submatrix(0..2, 0..1).data(),vec![59.0,74.0])
/// ```

pub fn _dgemm<'a, T,Q,P>(
    matr_a: &T, sub_a_dim: (Range<usize>, Range<usize>), opa: char, 
    matr_b: &Q, sub_b_dim: (Range<usize>, Range<usize>), opb: char, 
    matr_c: &mut P, sub_c_dim: (Range<usize>, Range<usize>),
    alpha: f64, beta: f64,
) 
where T: BasicMatrix<'a, f64>,
      Q: BasicMatrix<'a, f64>, 
      P: BasicMatrix<'a, f64>
{
    let check_shape = match (&opa, &opb) {
        ('N','N') => {
            sub_a_dim.1.len() == sub_b_dim.0.len() && sub_a_dim.0.len() == sub_c_dim.0.len() && sub_b_dim.1.len() == sub_c_dim.1.len()
        },
        ('T','N') => {
            sub_a_dim.0.len() == sub_b_dim.0.len() && sub_a_dim.1.len() == sub_c_dim.0.len() && sub_b_dim.1.len() == sub_c_dim.1.len()
        },
        ('N','T') => {
            sub_a_dim.1.len() == sub_b_dim.1.len() && sub_a_dim.0.len() == sub_c_dim.0.len() && sub_b_dim.0.len() == sub_c_dim.1.len()
        },
        ('T','T') => {
            sub_a_dim.0.len() == sub_b_dim.1.len() && sub_a_dim.1.len() == sub_c_dim.0.len() && sub_b_dim.0.len() == sub_c_dim.1.len()
        },
        _ => {false}
    };
    // check the shapes of the input matrices for the dgemm operation
    if ! check_shape {panic!("ERROR:: Matr_A[{:},{:},{:}] * Matr_B[{:},{:},{:}] -> Matr_C[{:},{:}]",
         sub_a_dim.0.len(), sub_a_dim.1.len(), &opa,
         sub_b_dim.0.len(), sub_b_dim.1.len(), &opb,
         sub_c_dim.0.len(), sub_c_dim.1.len()
        )
    }
    let is_contiguous = matr_a.is_contiguous() && matr_b.is_contiguous() && matr_c.is_contiguous();
    if is_contiguous {
        let matr_c_size = [matr_c.size()[0],matr_c.size()[1]];
        crate::external_libs::general_dgemm_f(
            matr_a.data_ref().unwrap(), matr_a.size(), sub_a_dim.0, sub_a_dim.1, opa, 
            matr_b.data_ref().unwrap(), matr_b.size(), sub_b_dim.0, sub_b_dim.1, opb, 
            matr_c.data_ref_mut().unwrap(), &matr_c_size, sub_c_dim.0, sub_c_dim.1, 
            alpha, beta)
    } else {
        panic!("the matrixs into the dgemm function should be all stored in a contiguous memory block");
    }

}

impl <'a> MatrixFullSlice<'a, f64> {
    #[inline]
    pub fn ddot(&self, b: &MatrixFullSlice<f64>) -> Option<MatrixFull<f64>> {
        /// for self a => a*b
        let flag = self.size[1]==b.size[0];
        if flag {
            let (m, n, k) = (self.size[0],b.size[1],self.size[1]);
            let mut c: Vec<f64> = vec![0.0; (m*n)];
            unsafe {
                dgemm(b'N',b'N',m as i32,n as i32,k as i32,1.0,self.data,m as i32,b.data,k as i32,1.0,&mut c,m as i32);
            }
            Some(unsafe{MatrixFull::from_vec_unchecked([m,n], c)})
        } else {
            None
        }
    }
}

impl <'a> MatrixFullSliceMut<'a, f64> {
    //#[inline]
    //pub fn check_shape(&self, other: &MatrixFullSliceMut<f64>, opa: char, opb: char) -> bool 
    //{
    //    crate::matrix::matrix_blas_lapack::general_check_shape(self, other, opa, opb)
    //}
    pub fn lapack_dgemm(&mut self, a: &MatrixFullSlice<f64>, b: & MatrixFullSlice<f64>, opa: char, opb: char, alpha: f64, beta: f64) {

        /// for self c = alpha*opa(a)*opb(b) + beta*c
        /// 
        let m = if opa=='N' {a.size[0]} else {a.size[1]};
        let k = if opa=='N' {a.size[1]} else {a.size[0]};
        let n = if opb=='N' {b.size[1]} else {b.size[0]};
        let l = if opa=='N' {b.size[0]} else {a.size[1]};
        let lda = if opa=='N' {m.max(1)} else {k.max(1)};
        let ldb = if opb=='N' {k.max(1)} else {n.max(1)};
        let ldc = m.max(1);

        // check the consistence of the shape of three matrices: 
        //     op(a): m x k; op(b): k x n -> c: m x n
        //let flag = k==l && m == self.size[0] && n == self.size[1];
        let flag = true; 
        if flag {
            unsafe {
                dgemm(opa as u8,
                      opb as u8,
                      m as i32,
                      n as i32,
                      k as i32,
                      alpha,
                      a.data,
                      lda as i32,
                      b.data,
                      ldb as i32,
                      beta,
                      &mut self.data,
                      ldc as i32);
            }
        } else {
            panic!("Error: Inconsistency happens to perform dgemm w.r.t. op(a)*op(b) -> c");
        }
    }
    pub fn multiple(&mut self, scaled_factor: f64) {
        /// for self a => a*scaled_factor
        self.data.iter_mut().for_each(|i| {
            *i *= scaled_factor;
        });
    }
    pub fn lapack_dsyev(&mut self) -> Option<(MatrixFull<f64>,Vec<f64>,i32)> {
        /// eigenvalues and eigenvectors of self a
        if self.size[0]==self.size[1] {
            let ndim = self.size[0];
            let n= ndim as i32;
            let mut a: Vec<f64> = self.data.to_vec().clone();
            let mut w: Vec<f64> = vec![0.0;ndim];
            let mut work: Vec<f64> = vec![0.0;4*ndim];
            let lwork = 4*n;
            let mut info = 0;
            unsafe {
                dsyev(b'V',b'L',n,&mut a, n, &mut w, &mut work, lwork, &mut info);
            }
            if info!=0 {
                panic!("Error in diagonalizing the matrix");
            }
            let eigenvectors = MatrixFull::from_vec([ndim,ndim], a).unwrap();
            //let eigenvalues = Tensors::from_vec(String::from("full"),vec![self.size[0]], w);
            Some((eigenvectors, w,n))
        } else {
            None
        }
    }
    pub fn lapack_dgetrf(&mut self) -> Option<MatrixFull<f64>> {
        if self.size[0]==self.size[1] {
            let ndim = self.size[0];
            let n= ndim as i32;
            let mut a: Vec<f64> = self.data.to_vec().clone();
            let mut w: Vec<f64> = vec![0.0;ndim];
            let mut work: Vec<f64> = vec![0.0;4*ndim];
            let mut ipiv: Vec<i32> = vec![0;ndim];
            let lwork = 4*n;
            let mut info1 = 0;
            unsafe {
                dgetrf(n,n,&mut a,n, &mut ipiv, &mut info1);
            }
            if info1!=0 {
                panic!("Error happens when LU factorizing the matrix. dgetrf info: {}", info1);
            }
            let inv_mat = MatrixFull::from_vec([ndim,ndim], a).unwrap();
            Some(inv_mat)
        } else {
            println!("Error: The matrix for inversion should be NxN");
            None
        }
    }

    pub fn lapack_inverse(&mut self) -> Option<MatrixFull<f64>> {
        if self.size[0]==self.size[1] {
            let ndim = self.size[0];
            let n= ndim as i32;
            let mut a: Vec<f64> = self.data.to_vec().clone();
            let mut w: Vec<f64> = vec![0.0;ndim];
            let mut work: Vec<f64> = vec![0.0;4*ndim];
            let mut ipiv: Vec<i32> = vec![0;ndim];
            let lwork = 4*n;
            let mut info1 = 0;
            let mut info2 = 0;
            unsafe {
                dgetrf(n,n,&mut a,n, &mut ipiv, &mut info1);
                dgetri(n,&mut a,n, &mut ipiv, &mut work, lwork, &mut info2);
            }
            if info1!=0 || info2!=0 {
                panic!("Error happens when inversing the matrix. dgetrf info: {}; dgetri info: {}", info1, info2);
            }
            let inv_mat = MatrixFull::from_vec([ndim,ndim], a).unwrap();
            Some(inv_mat)
        } else {
            println!("Error: The matrix for inversion should be NxN");
            None
        }
    }

    pub fn lapack_power(&mut self,p:f64, threshold: f64) -> Option<MatrixFull<f64>> {
        let mut om = MatrixFull::new([self.size[0],self.size[1]],0.0);
        om.data.iter_mut().zip(self.data.iter()).for_each(|value| {*value.0=*value.1});
        if self.size[0]==self.size[1] {
            // because lapack sorts eigenvalues from small to large
            om.self_multiple(-1.0);
            // diagonalize the matrix
            let (mut eigenvector, mut eigenvalues, mut n) = om.to_matrixfullslicemut().lapack_dsyev().unwrap();
            // now we get the eigenvectors with the eigenvalues from large to small
            let (mut n_nonsigular, mut tmpv) = (0usize,0.0);
            eigenvalues.iter_mut().enumerate().for_each(|(i,value)| {
                *value = *value*(-1.0);
                //if *value >= threshold && *value <= tmpv {
                if *value >= threshold {n_nonsigular +=1};
            });

            if ! n as usize == self.size[0] {
                panic!("Found unphysical eigenvalues");
            }
            //println!("n_nonsigular: {}", n_nonsigular);

            //(0..n_nonsigular).into_iter().for_each(|i| {
            //    let ev_sqrt = eigenvalues[i].sqrt();
            //    let mut tmp_slice = eigenvector.get2d_slice_mut([0,i],self.size[0]).unwrap();
            //    tmp_slice.iter_mut().for_each(|v| {*v = *v*ev_sqrt.powf(p)});
            //    //println!("{}: {:?}", ev_sqrt,&tmp_slice);
            //});

            &eigenvector.data.chunks_exact_mut(self.size[0]).enumerate()
                .take_while(|(i,value)| i<&n_nonsigular)
                .for_each(|(i,value)| {
                if let Some(ev) = eigenvalues.get_mut(i) {
                    let ev_sqrt = ev.sqrt();
                    value.iter_mut().for_each(|v| {*v = *v*ev_sqrt.powf(p)});
                }
            });


            let mut eigenvector_b = eigenvector.clone();


            om.to_matrixfullslicemut().lapack_dgemm(
                &mut eigenvector.to_matrixfullslice(), 
                &mut eigenvector_b.to_matrixfullslice(), 
                'N', 'T', 1.0, 0.0);

            Some(om)
        } else {
            println!("Error: The matrix for power operations should be NxN");
            None
        }

    }
}


impl <'a> MatrixUpperSliceMut<'a, f64> {
    pub fn multiple(&mut self, scaled_factor: f64) {
        /// for self a => a*scaled_factor
        self.data.iter_mut().for_each(|i| {
            *i *= scaled_factor;
        });
    }
    pub fn lapack_dspevx(&mut self) -> Option<(MatrixFull<f64>,Vec<f64>,i32)> {
        /// eigenvalues and eigenvectors of self a
        let n = ((8.0*self.size.to_owned() as f32+1.0).sqrt()/2.0) as i32;
        let lwork = 4*n;
        let ndim = n as usize;
        let mut a: Vec<f64> = self.data.to_vec().clone();
        let mut w: Vec<f64> = vec![0.0;ndim];
        let mut z: Vec<f64> = vec![0.0;ndim*ndim];
        let mut work: Vec<f64> = vec![0.0;8*ndim*ndim];
        let mut iwork: Vec<i32> = vec![0;5*ndim];
        let mut ifail: Vec<i32> = vec![0;ndim];
        let mut n_found:i32 = 0;
        let mut info = 0;
        unsafe {
            dspevx(b'V',b'A',b'U',n,&mut a, 0.0_f64, 0.0_f64,0,0,
                   SAFE_MINIMUM,&mut n_found, &mut w, &mut z, n, &mut work, &mut iwork, &mut ifail,&mut info);
        }
        let eigenvectors = MatrixFull::from_vec([ndim,ndim], z).unwrap();
        //let eigenvalues = Tensors::from_vec(String::from("full"),vec![self.size[0]], w);
        Some((eigenvectors, w, n_found))
    }
    pub fn lapack_dspgvx(&mut self,ovlp:MatrixUpperSliceMut<f64>,num_orb:usize) -> Option<(MatrixFull<f64>,Vec<f64>)> {
        ///solve A*x=(lambda)*B*x
        /// A is "self"; B is ovlp
        let mut itype: i32 = 1;
        let n = ((8.0*self.size.to_owned() as f32+1.0).sqrt()/2.0) as i32;
        let ndim = n as usize;
        let mut a = self.data.to_vec().clone();
        let mut b = ovlp.data.to_vec().clone();
        let mut m = 0;
        let mut w: Vec<f64> = vec![0.0;ndim];
        let mut z: Vec<f64> = vec![0.0;ndim*ndim];
        let mut work: Vec<f64> = vec![0.0;8*ndim];
        let mut iwork:Vec<i32> = vec![0;5*ndim];
        let mut ifail:Vec<i32> = vec![0;ndim];
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
        let eigenvectors = MatrixFull::from_vec([ndim,num_orb],z).unwrap();
        //let eigenvalues = Tensors::from_vec("full".to_string(),vec![n as usize],w);
        Some((eigenvectors, w))
    }
}

// for c = a*b
#[inline]
pub fn _dgemm_nn(mat_a: &MatrixFullSlice<f64>, mat_b: &MatrixFullSlice<f64>) -> MatrixFull<f64> {
    let (ax,ay) = (mat_a.size[0], mat_a.size[1]);
    let (bx,by) = (mat_b.size[0], mat_b.size[1]);
    if ay!=bx {panic!("For the input matrices: mat_a[ax,ay], mat_b[bx,by], ay!=bx. dgemm false")};
    if (ax==0||by==0) {return MatrixFull::new([ax,by],0.0)};
    let mut mat_c = MatrixFull::new([ax,by],0.0);
    //let mat_aa = mat_a.transpose();
    mat_c.par_iter_columns_full_mut().zip(mat_b.par_iter_columns_full()).for_each(|(mat_c,mat_b)| {
        mat_b.iter().zip(mat_a.iter_columns_full()).for_each(|(mat_b,mat_a)| {
            mat_c.iter_mut().zip(mat_a.iter()).for_each(|(mat_c,mat_a)| {
                *mat_c += mat_a*mat_b;
            })
        });
        //mat_c.iter_mut().zip(mat_a.iter_columns_full()).for_each(|(mat_c,mat_a)| {
        //    *mat_c = mat_a.iter().zip(mat_b.iter()).fold(0.0,|acc,(a,b)| acc + a*b)
        //});
    });
    mat_c
}
// for c = a**T*b
#[inline]
pub fn _dgemm_tn(mat_a: &MatrixFullSlice<f64>, mat_b: &MatrixFullSlice<f64>) -> MatrixFull<f64> {
    let (ax,ay) = (mat_a.size[0], mat_a.size[1]);
    let (bx,by) = (mat_b.size[0], mat_b.size[1]);
    if ax!=bx {panic!("For the input matrices: mat_a[ax,ay], mat_b[bx,by], ay!=bx. dgemm false")};
    if (ay==0||by==0) {return MatrixFull::new([ay,by],0.0)};
    let mut mat_c = MatrixFull::new([ay,by],0.0);
    //let mat_aa = mat_a.transpose();
    mat_c.par_iter_columns_full_mut().zip(mat_b.par_iter_columns_full()).for_each(|(mat_c,mat_b)| {
        mat_c.iter_mut().zip(mat_a.iter_columns_full()).for_each(|(mat_c,mat_a)| {
            *mat_c = mat_a.iter().zip(mat_b.iter()).fold(0.0,|acc,(a,b)| acc + a*b)
        });
    });
    mat_c
}
pub fn _dgemm_tn_v02(mat_a: &MatrixFullSlice<f64>, mat_b: &MatrixFullSlice<f64>, to_slice: Flatten<IntoIter<&mut [f64]>>) {
    let (ax,ay) = (mat_a.size[0], mat_a.size[1]);
    let (bx,by) = (mat_b.size[0], mat_b.size[1]);
    if ax!=bx {panic!("For the input matrices: mat_a[ax,ay], mat_b[bx,by], ay!=bx. dgemm false")};
    //if (ay==0||by==0) {return MatrixFull::new([ay,by],0.0)};
    //let mut mat_c = MatrixFull::new([ay,by],0.0);
    //let mat_aa = mat_a.transpose();
    let mut mat_c = MatrixFullSliceMut2{
        size: &[ay,by],
        indicing: &[1,ay],
        data: to_slice.collect::<Vec<&mut f64>>()
    };
    mat_c.iter_mut_columns_full().zip(mat_b.iter_columns_full()).for_each(|(mat_c,mat_b)| {
        mat_c.iter_mut().zip(mat_a.iter_columns_full()).for_each(|(mat_c,mat_a)| {
            **mat_c = mat_a.iter().zip(mat_b.iter()).fold(0.0,|acc,(a,b)| acc + a*b)
        });
    });
}

#[inline]
// einsum: ij, j -> ij
pub fn _einsum_01(mat_a: &MatrixFullSlice<f64>, vec_b: &[f64]) -> MatrixFull<f64>{
    let i_len = mat_a.size[0];
    let j_len = vec_b.len();
    if (i_len == 0 || j_len ==0) {return MatrixFull::new([i_len,j_len],0.0)};
    let mut om = MatrixFull::new([i_len,j_len],0.0);

    om.par_iter_columns_full_mut().zip(mat_a.par_iter_columns(0..j_len).unwrap())
    .map(|(om_j,mat_a_j)| {(om_j,mat_a_j)})
    .zip(vec_b.par_iter())
    .for_each(|((om_j,mat_a_j),vec_b_j)| {
        om_j.iter_mut().zip(mat_a_j.iter()).for_each(|(om_ij,mat_a_ij)| {
            *om_ij = *mat_a_ij*vec_b_j
        });
    });
    om 
}
#[inline]
// einsum ip, ip -> p
pub fn _einsum_02(mat_a: &MatrixFullSlice<f64>, mat_b: &MatrixFullSlice<f64>) -> Vec<f64> {
    let a_y = mat_a.size.get(1).unwrap();
    let b_y = mat_b.size.get(1).unwrap();
    let a_x = mat_a.size.get(0).unwrap();
    let b_x = mat_b.size.get(0).unwrap();
    if (*a_x == 0 || *b_x ==0) {return vec![0.0;*a_y.min(b_y)]};
    let mut out_vec = vec![0.0;*a_y.min(b_y)];

    mat_a.par_iter_columns_full().zip(mat_b.par_iter_columns_full())
    .map(|(mat_a_p, mat_b_p)| (mat_a_p,mat_b_p))
    .zip(out_vec.par_iter_mut())
    .for_each(|((mat_a_p,mat_b_p),out_vec_p)| {
        *out_vec_p = mat_a_p.iter().zip(mat_b_p.iter())
            .fold(0.0, |acc, (mat_a_ip, mat_b_ip)| 
            {acc + mat_a_ip*mat_b_ip});
    });
    out_vec
}

#[test]
fn test_einsum_02() {
    let mut mat_a = MatrixFull::from_vec([2,2],vec![3.0,4.0,2.0,6.0]).unwrap();
    let mut mat_b = mat_a.clone();
    let mut mat_c = _einsum_02(&mat_a.to_matrixfullslice(), &mat_b.to_matrixfullslice());
    println!("{:?}", mat_c);

}
