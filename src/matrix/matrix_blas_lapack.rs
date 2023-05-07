use std::{iter::Flatten, vec::IntoIter, ops::Range};

use blas::{dgemm,dtrmm, dsymm};
use lapack::{dsyev, dspgvx, dspevx,dgetrf,dgetri,dlamch, dsyevx, dpotrf, dtrtri, dpotri};
use nalgebra::matrix;
use rayon::prelude::*;

use crate::{MatrixFullSliceMut, MatrixFull, SAFE_MINIMUM, MatrixUpperSliceMut, TensorSlice, TensorSliceMut, MatrixFullSlice, MatrixFullSliceMut2, BasicMatrix};

use super::ParMathMatrix;


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
///    //             |101.0 |146.0 | 2.0 |
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
        ('T','T') =>{
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
    let check_within = sub_a_dim.0.end <= matr_a.size()[0] && sub_a_dim.1.end <= matr_a.size()[1]
                          && sub_b_dim.0.end <= matr_b.size()[0] && sub_b_dim.1.end <= matr_b.size()[1]
                          && sub_c_dim.0.end <= matr_c.size()[0] && sub_c_dim.1.end <= matr_c.size()[1];
    // check if the sub matrix block is within the matrix
    if ! check_within {
        panic!("ERROR:: Matr_A.size: {:?}, SubMatr_A: ({:?},{:?}). Matr_B.size: {:?}, SubMatr_B: ({:?},{:?}); Matr_C.size: {:?}, SubMatr_C: ({:?},{:?});  ",
        matr_a.size(), sub_a_dim.0, sub_a_dim.1,
        matr_b.size(), sub_b_dim.0, sub_b_dim.1,
        matr_c.size(), sub_c_dim.0, sub_c_dim.1,
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

/// # computes all eigenvalues and , optionally, eigenvectors of a real symmetric matrix A  
///    jobz: char, 'N' or 'V'  
///  = 'N': compute eigenvalues only;  
///  = 'V': compute eigenvalues and eigenvectors.  
/// 
/// Example:
/// ```
///    use rest_tensors::{MatrixFull, MatrixUpper};
///    use rest_tensors::matrix::matrix_blas_lapack::_dsyev;
///    // generate a matrix with only the upper triangle of A is stored in `MatrixUpper`.
///    let matr_a = MatrixUpper::from_vec(6, (1..7).map(|x| x as f64).collect::<Vec<f64>>()).unwrap();
///    //             | 1.0 | 2.0 | 4.0 |
///    // matr_a =    | 0.0 | 3.0 | 5.0 |
///    //             | 0.0 | 0.0 | 6.0 |
///    // transfer it to a MatrixFull format and store in matr_b
///    let matr_b = matr_a.to_matrixfull().unwrap();
/// 
///    // jobz = 'V': computes all eigenvalues and eigenvectors
///    let(eigenvectors, eigenvalues, ndim) = _dsyev(&matr_b, 'V');
///    // the eigenvectors reference data
///    let eigvec_benchmark = vec![
///         -0.6827362941552275, -0.38559063640162244, 0.6206375864887483, 
///          0.6202872696512856, -0.7547821848190948, 0.21341873532627673, 
///         -0.3861539275363389, -0.5306823104260265, -0.7544941548144386];
///    let diff = &eigenvectors.clone().unwrap().data.iter().zip(eigvec_benchmark.iter()).fold(0.0,|acc,(a,b)| acc + (a-b).powf(2.0));
///    assert!(diff < &10E-7, "eigenvectors: {:?}, {:?}", &eigenvectors.unwrap().data, &eigvec_benchmark);
///    // the eigenvalues reference data
///    let eigval_benchmark = vec![-1.5066326307865059, -0.05739624271478554, 11.564028873501286];
///    let diff = eigenvalues.iter().zip(eigval_benchmark.iter()).fold(0.0,|acc,(a,b)| acc + (a-b).powf(2.0));
///    assert!(diff < 10E-7, "eigenvalues: {:?}, {:?}", &eigenvalues, &eigval_benchmark);
/// 
///    // jobz = 'N': computes all eigenvalues only
///    let(eigenvectors, eigenvalues, ndim) = _dsyev(&matr_b, 'N');
///    assert_eq!(eigenvectors, None);
///    // the eigenvalues reference data
///    let eigval_benchmark = vec![-1.5066326307865059, -0.05739624271478554, 11.564028873501286];
///    let diff = eigenvalues.iter().zip(eigval_benchmark.iter()).fold(0.0,|acc,(a,b)| acc + (a-b).powf(2.0));
///    assert!(diff < 10E-7, "eigenvalues: {:?}, {:?}", eigenvalues, eigval_benchmark);
/// ```
pub fn _dsyev<'a, T>(matr_a: &T, jobz: char) -> (Option<MatrixFull<f64>>,Vec<f64>,i32) 
where T: BasicMatrix<'a, f64>
{
    let ndim = matr_a.size()[0];
    let size = [ndim, matr_a.size()[1]];
    if size[0]==size[1] {
        let n= ndim as i32;
        let mut a = matr_a.data_ref().unwrap().iter().map(|x| *x).collect::<Vec<f64>>();
        let mut w: Vec<f64> = vec![0.0;ndim];
        let mut work: Vec<f64> = vec![0.0;4*ndim];
        let lwork = 4*n;
        let mut info = 0;
        unsafe {
            dsyev(jobz.clone() as u8,b'L',n,&mut a, n, &mut w, &mut work, lwork, &mut info);
        }
        if info<0 {
            panic!("Error in _dsyev: the {}-th argument had an illegal value", -info);
        } else if info>0 {
            panic!("Error in _dsyev: the algorithm failed to converge; {}-off-diagonal elements of an intermediate tridiagonal form did not converge to zero", -info);

        }
        let eigenvectors = if jobz.eq(&'V') {
            Some(MatrixFull::from_vec([ndim,ndim], a).unwrap())
        } else {None};
        //let eigenvalues = Tensors::from_vec(String::from("full"),vec![self.size[0]], w);
        (eigenvectors, w,n)
    } else {
        panic!("Error in _dsyev: the algorithm is only vaild for real symmetric matrices")
    }
}
/// # DSYMM performs one of the matrix-matrix operations  
///    C := alpha*A*B + beta*C,  
/// or  
///    C := alpha*B*A + beta*C,  
/// where alpha and beta are scalars, A is a symmetric matrix and B and C are m by n matrices
pub fn _dsymm<'a, T, Q, P>  (
    matr_a: &T, matr_b: &Q, matr_c: &mut P,
    side: char, uplo: char,
    alpha: f64, beta: f64
)
where T: BasicMatrix<'a, f64>,
      Q: BasicMatrix<'a, f64>, 
      P: BasicMatrix<'a, f64>
{
    //let side0 = side.to_lowercase();
    //let uplo0 = uplo.to_lowercase();
    let m = matr_c.size()[0] as i32;
    let n = matr_c.size()[1] as i32;

    let lda = if side.to_string().to_lowercase().eq("l") {m} else {n};
    let ldb = m;
    let ldc = m;
    unsafe {
        dsymm(side as u8, uplo as u8, m, n, alpha, 
            matr_a.data_ref().unwrap(), lda, 
            matr_b.data_ref().unwrap(), ldb, beta, 
            matr_c.data_ref_mut().unwrap(), ldc)
    }

}

/// # computes the Cholesky factorization of a real symmetric positive definite matrix A  
/// uplo: char, `U` or `L`  
/// = `U`: Upper triangle of A is stored  
/// = `V`: Lower triangle of A is stored  
/// 
/// **Note**: [`_dpotrf`] destroys the input matrix `matr_a`, which, on exist is the factor 'U' or 'L' from
/// the Cholesky factorizatoin A = U**T*U or A = L*L**T  
/// 
/// Example
/// ```
///    use rest_tensors::{MatrixFull, MatrixUpper};
///    use rest_tensors::matrix::matrix_blas_lapack::_dpotrf;
///    // generate a matrix with only the upper triangle of A is stored in `MatrixUpper`.
///    let matr_0 = MatrixUpper::from_vec(6, vec![4.0,12.0,37.0,-16.0,-43.0,98.0]).unwrap();
///    // transfer it to a MatrixFull format and store in matr_a
///    let mut matr_a = matr_0.to_matrixfull().unwrap();
///    //println!("{:?}", &matr_a);
///    //            |  4.0 | 12.0 |-16.0 |
///    // matr_a =   | 12.0 | 37.0 |-43.0 |
///    //            |-16.0 |-43.0 | 98.0 |
///    _dpotrf(&mut matr_a, 'U');
///    let U = matr_a.iter_matrixupper().unwrap().map(|x| *x).collect::<Vec<f64>>();
///    assert_eq!(U, vec![2.0,6.0,1.0,-8.0,5.0,3.0]);
/// ```
pub fn _dpotrf<'a, T>(matr_a: &mut T, uplo: char)  
where T: BasicMatrix<'a, f64>
{
    let size = [matr_a.size()[0], matr_a.size()[1]];
    if size[0]==size[1] {
        let n = size[0] as i32;
        let mut info = 0;
        unsafe{dpotrf(uplo as u8, n, &mut matr_a.data_ref_mut().unwrap(),n, &mut info)};
        if info<0 {
            panic!("ERROR in DPOTRF: The {}-th argument has an illegal value", info);
        } else if info >0 {
            panic!("ERROR in DPOTRF: The leading minor of order {} is not positive definite, and the factorization could not be completed", info);
        }
    } else {
        panic!("ERROR: cannot make Cholesky factorization of a matrix with different row and column lengths");
    }
}
/// #  Compute a matrix inversion
/// 
///    A -> A^{-1}
/// 
/// Example
/// ```
///    use rest_tensors::{MatrixFull, MatrixUpper, BasicMatrix};
///    use rest_tensors::matrix::matrix_blas_lapack::{_dinverse, _dgemm};
///    // generate a matrix with only the upper triangle of A is stored in `MatrixUpper`.
///    let matr_0 = MatrixUpper::from_vec(6, vec![4.0,12.0,37.0,-16.0,-43.0,98.0]).unwrap();
///    // transfer it to a MatrixFull format and store in matr_a
///    let mut matr_a = matr_0.to_matrixfull().unwrap();
///    //println!("{:?}", &matr_a);
///    //            |  4.0 | 12.0 |-16.0 |
///    // matr_a =   | 12.0 | 37.0 |-43.0 |
///    //            |-16.0 |-43.0 | 98.0 |
///    let matr_a_inv = _dinverse(&mut matr_a).unwrap();
/// 
///    // matr_b = matr_a * matr_a_inv = unit_matrix
///    let mut matr_b = MatrixFull::new([3,3],0.0);
///    _dgemm(
///        &matr_a, (0..3,0..3),'N',
///        &matr_a_inv, (0..3,0..3),'N',
///        &mut matr_b, (0..3,0..3),
///        1.0, 0.0
///    );
///    let unit_matr = MatrixFull::from_vec([3,3], vec![
///       1.0, 0.0, 0.0,
///       0.0, 1.0, 0.0,
///       0.0, 0.0, 1.0]).unwrap();
///    let diff = unit_matr.data_ref().unwrap().iter().zip(matr_b.data_ref().unwrap().iter()).fold(0.0,|acc, (x,y)| {acc + (x-y).powf(2.0)});
///    assert!(diff < 10E-7, "matr_b is not a unit matrix: {:?}", &matr_b);
/// ```
pub fn _dinverse<'a, T>(matr_a: &T) -> Option<MatrixFull<f64>> 
where T: BasicMatrix<'a, f64>
{
    let size = [matr_a.size()[0], matr_a.size()[1]];
    if size[0]==size[1] {
        let ndim = size[0];
        let n= ndim as i32;
        let mut a = matr_a.data_ref().unwrap().iter().map(|x| *x).collect::<Vec<f64>>();
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
            println!("Error happens when inversing the matrix. dgetrf info: {}; dgetri info: {}", info1, info2);
            return(None)
        }
        let inv_mat = MatrixFull::from_vec([ndim,ndim], a).unwrap();
        Some(inv_mat)
    } else {
        println!("Error: The matrix for inversion should be NxN");
        None
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

    pub fn lapack_dsyevx(&mut self) -> Option<(MatrixFull<f64>,Vec<f64>,i32)> {

        // Set absolute tolerance for eigenvalue precision
        // 2*SAFE_MINIMUM is the recommended for the most precise choice.
        // WARNING: SAFE_MINIMUM = dmach('S' as usize), which might give zero if overoptimized by the compiler.
        // In this case, try a lowermost value of 1E-12 instead

        let abs_tol = unsafe{dlamch(b'S')}*2.0_f64;
        let threshold = -1E10;

        let job_z = b'V';
        let my_range = b'V';
        let uplo = b'U';
        let i_lower = 1;
        let i_upper = self.size[0] as i32;
        let v_lower = threshold;
        let v_upper = 1E5;

        /// eigenvalues and eigenvectors of self a
        if self.size[0]==self.size[1] {
            let ndim = self.size[0];
            let n= ndim as i32;
            let mut a: Vec<f64> = self.data.to_vec().clone();
            let mut w: Vec<f64> = vec![0.0;ndim];
            let mut work: Vec<f64> = vec![0.0;8*ndim];
            let mut iwork: Vec<i32> = vec![0; 5*ndim];
            let mut lwork = 8*ndim as i32;
            let mut ifail:Vec<i32> = vec![0;ndim];
            let mut info = 0;
            let mut n_found = 0_i32;
            unsafe {
                //dsyevx(b'V',b'L',n,&mut a, n, &mut w, &mut work, lwork, &mut info);
                dsyevx(job_z, my_range, uplo, n, &mut self.data, n, v_lower, v_upper, i_lower, i_upper,
                abs_tol, &mut n_found, & mut w, &mut a, n, &mut work, lwork, &mut iwork, &mut ifail, &mut info);
            }
            if info<0 {
                println!("ERROR :: Eigenvalue solver dsyevx():");
                panic!  ("         The {}th argument in dspevx() has an illegal value. Check", -info);
            } else if info>0 {
                println!("ERROR :: Eigenvalue solver dsyevx():");
                panic!  ("         {} eigenvectors failed to converged. ifail: {:?}", info, ifail);
            }

            if n_found < n {
                println!("Matrix is singular: please use {} out of a possible {} specified spectrums", n_found, n);
            }
            let eigenvectors = MatrixFull::from_vec([ndim,ndim], a).unwrap();
            //let eigenvalues = Tensors::from_vec(String::from("full"),vec![self.size[0]], w);
            Some((eigenvectors, w,n_found))
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

    pub fn lapack_dpotrf(&mut self, uplo: u8) {
        if self.size[0]==self.size[1] {
            let n = self.size[0] as i32;
            let mut info = 0;
            unsafe{dpotrf(uplo, n, &mut self.data,n, &mut info)};
            if info!=0 {
                panic!("ERROR: DPOTRF failed with {}", info);
            }
        } else {
            panic!("ERROR: cannot make Cholesky factorization of a matrix with different row and column lengths");
        }
    }

    pub fn lapack_dtrtri(&mut self, uplo: u8) {
        if self.size[0]==self.size[1] {
            let n = self.size[0] as i32;
            let mut info = 0;
            let diag = b'N';
            unsafe{dtrtri(uplo, diag, n, &mut self.data,n, &mut info)};
            if info!=0 {
                panic!("ERROR: DTRITI failed with {}", info);
            }
        } else {
            panic!("ERROR: cannot make Cholesky factorization of a matrix with different row and column lengths");
        }

    }

    pub fn cholesky_decompose_inverse(&mut self,uplo:char) -> Option<MatrixFull<f64>> {
        let size = [self.size[0], self.size[1]];
        let n = size[0] as i32;

        if size[0] == size[1] {
            let ndim = size[0];
            let mut info = 0;
            let mut data = self.data.iter().map(|x| *x).collect::<Vec<f64>>();

            //perform Cholesky decomposition on the matrix A -> A = L*L'
            unsafe{dpotrf(uplo.clone() as u8, n, &mut data,n, &mut info)};
            if info!=0 {
                panic!("ERROR: DPOTRF failed with {}", info);
            }
            // compute the inverse of L 
            unsafe{dpotri(uplo.clone() as u8, n, &mut data,n, &mut info)};
            if info!=0 {
                panic!("ERROR: DPOTRI failed with {}", info);
            }

            let mut l_matr = MatrixFull::from_vec(size.clone(), data).unwrap();

            if uplo.eq(&'L') {
                for j in 0..ndim {
                    for i in j+1..ndim {
                        l_matr[(j,i)] = l_matr[(i,j)];
                    }
                }
            } else if uplo.eq(&'U') {
                for j in 0..ndim {
                    for i in j+1..ndim {
                        l_matr[(i,j)] = l_matr[(j,i)];
                    }
                }
            }

            unsafe{dpotrf(uplo.clone() as u8, n, &mut l_matr.data, n, &mut info)};

            if uplo.eq(&'L') {
                for j in 0..ndim {
                    for i in j+1..ndim {
                        l_matr[(j,i)] = 0.0_f64;
                    }
                }
            } else if uplo.eq(&'U') {
                for j in 0..ndim {
                    for i in j+1..ndim {
                        l_matr[(i,j)] = 0.0_f64;
                    }
                }
            }
            //let mut out_matr = l_matr.clone();

            //unsafe{dtrmm(b'L', b'U', b'T', b'N', n, n, 1.0, 
            //    &l_matr.data, n, &mut out_matr.data, n)};

            //unsafe{dtrmm(b'R', b'U', b'N', b'N', n, n, 1.0, 
            //    &l_matr.data, n, &mut a_matr.data, n)};


            //out_matr.to_matrixfullslicemut().lapack_dgemm(&cf.to_matrixfullslice(), &cf.to_matrixfullslice(), 'N', 'N', 1.0, 0.0);

            Some(l_matr)

        } else {
            None
        }
        

    }

    pub fn lapack_power(&mut self,p:f64, threshold: f64) -> Option<MatrixFull<f64>> {
        let mut om = MatrixFull::new([self.size[0],self.size[1]],0.0);
        om.data.par_iter_mut().zip(self.data.par_iter()).for_each(|value| {*value.0=*value.1});
        if self.size[0]==self.size[1] {
            // because lapack sorts eigenvalues from small to large
            om.par_self_multiple(-1.0);
            // diagonalize the matrix
            let (mut eigenvector, mut eigenvalues, mut n) = om.to_matrixfullslicemut().lapack_dsyev().unwrap();
            // now we get the eigenvectors with the eigenvalues from large to small
            let (mut n_nonsigular, mut tmpv) = (0usize,0.0);
            eigenvalues.iter_mut().for_each(|value| {
                *value = *value*(-1.0);
                //if *value >= threshold && *value <= tmpv {
                if *value >= threshold {
                    n_nonsigular +=1;
                };
            });

            //println!("{:?}", &eigenvalues);
            //println!("n_nonsigular: {}", n_nonsigular);

            if n as usize != self.size[0] {
                panic!("Found unphysical eigenvalues");
            } else if n_nonsigular != self.size[0] {
                println!("n_nonsigular: {}", n_nonsigular);
            }

            //(0..n_nonsigular).into_iter().for_each(|i| {
            //    let ev_sqrt = eigenvalues[i].sqrt();
            //    let mut tmp_slice = eigenvector.get2d_slice_mut([0,i],self.size[0]).unwrap();
            //    tmp_slice.iter_mut().for_each(|v| {*v = *v*ev_sqrt.powf(p)});
            //    //println!("{}: {:?}", ev_sqrt,&tmp_slice);
            //});

            &eigenvector.data.par_chunks_exact_mut(self.size[0]).enumerate()
                //.filter(|(i,value)| i<&n_nonsigular)
                .for_each(|(i,value)| {
                    if i<n_nonsigular {
                        if let Some(ev) = eigenvalues.get(i) {
                            let ev_sqrt = ev.sqrt();
                            value.iter_mut().for_each(|v| {*v = *v*ev_sqrt.powf(p)});
                        }
                    } else {
                        value.iter_mut().for_each(|v| {*v = 0.0f64});
                    }
            });

            //let eigenvector_b = eigenvector.clone();

            om.to_matrixfullslicemut().lapack_dgemm(
                &eigenvector.to_matrixfullslice(), 
                &eigenvector.to_matrixfullslice(), 
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

/// An matrix operation for the Vxc preparation: Contraction of a slice with a full matrix
/// `(mat_b[num_basis, num_grid], slice_c[num_grid]) -> mat_a[num_basis, num_grid]`
/// einsum(ij,i->ij)
pub fn contract_vxc_0_serial<'a, T,Q>(mat_a: &mut T, mat_b: &Q, slice_c: &[f64], scaling_factor: Option<f64>) 
where T: BasicMatrix<'a, f64>,
      Q: BasicMatrix<'a, f64>
{
    let mat_a_size = [mat_a.size()[0],mat_a.size()[1]];
    match scaling_factor {
        None =>  {
            mat_a.data_ref_mut().unwrap().chunks_exact_mut(mat_a_size[0])
                .zip(mat_b.data_ref().unwrap().chunks_exact(mat_a_size[0]))
                .zip(slice_c.iter())
                .for_each(|((mat_a,mat_b),slice_c)| {
                    mat_a.iter_mut().zip(mat_b.iter()).for_each(|(mat_a, mat_b)| {
                        *mat_a += mat_b*slice_c
                });
            });
        },
        Some(s) => {
            mat_a.data_ref_mut().unwrap().chunks_exact_mut(mat_a_size[0])
                .zip(mat_b.data_ref().unwrap().chunks_exact(mat_a_size[0]))
                .zip(slice_c.iter())
                .for_each(|((mat_a,mat_b),slice_c)| {
                    mat_a.iter_mut().zip(mat_b.iter()).for_each(|(mat_a, mat_b)| {
                        *mat_a += mat_b*slice_c*s
                });
            });

        }
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
// for c = a*b
#[inline]
pub fn _dgemm_nn_serial(mat_a: &MatrixFullSlice<f64>, mat_b: &MatrixFullSlice<f64>) -> MatrixFull<f64> {
    let (ax,ay) = (mat_a.size[0], mat_a.size[1]);
    let (bx,by) = (mat_b.size[0], mat_b.size[1]);
    if ay!=bx {panic!("For the input matrices: mat_a[ax,ay], mat_b[bx,by], ay!=bx. dgemm false")};
    if (ax==0||by==0) {return MatrixFull::new([ax,by],0.0)};
    let mut mat_c = MatrixFull::new([ax,by],0.0);
    //let mat_aa = mat_a.transpose();
    mat_c.iter_columns_full_mut().zip(mat_b.iter_columns_full()).for_each(|(mat_c,mat_b)| {
        mat_b.iter().zip(mat_a.iter_columns_full()).for_each(|(mat_b,mat_a)| {
            mat_c.iter_mut().zip(mat_a.iter()).for_each(|(mat_c,mat_a)| {
                *mat_c += mat_a*mat_b;
            })
        });
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
// for c = a**T*b
#[inline]
pub fn _dgemm_tn_serial(mat_a: &MatrixFullSlice<f64>, mat_b: &MatrixFullSlice<f64>) -> MatrixFull<f64> {
    let (ax,ay) = (mat_a.size[0], mat_a.size[1]);
    let (bx,by) = (mat_b.size[0], mat_b.size[1]);
    if ax!=bx {panic!("For the input matrices: mat_a[ax,ay], mat_b[bx,by], ay!=bx. dgemm false")};
    if (ay==0||by==0) {return MatrixFull::new([ay,by],0.0)};
    let mut mat_c = MatrixFull::new([ay,by],0.0);
    //let mat_aa = mat_a.transpose();
    mat_c.iter_columns_full_mut().zip(mat_b.iter_columns_full()).for_each(|(mat_c,mat_b)| {
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
pub fn _einsum_01_rayon(mat_a: &MatrixFullSlice<f64>, vec_b: &[f64]) -> MatrixFull<f64>{
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
pub fn _einsum_02_rayon(mat_a: &MatrixFullSlice<f64>, mat_b: &MatrixFullSlice<f64>) -> Vec<f64> {
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

#[inline]
// einsum: ij, j -> ij
pub fn _einsum_01_serial(mat_a: &MatrixFullSlice<f64>, vec_b: &[f64]) -> MatrixFull<f64>{
    let i_len = mat_a.size[0];
    let j_len = vec_b.len();
    if (i_len == 0 || j_len ==0) {return MatrixFull::new([i_len,j_len],0.0)};
    let mut om = MatrixFull::new([i_len,j_len],0.0);

    om.iter_columns_full_mut().zip(mat_a.iter_columns(0..j_len).unwrap())
    .map(|(om_j,mat_a_j)| {(om_j,mat_a_j)})
    .zip(vec_b.iter())
    .for_each(|((om_j,mat_a_j),vec_b_j)| {
        om_j.iter_mut().zip(mat_a_j.iter()).for_each(|(om_ij,mat_a_ij)| {
            *om_ij = *mat_a_ij*vec_b_j
        });
    });
    om 
}
#[inline]
// einsum ip, ip -> p
pub fn _einsum_02_serial(mat_a: &MatrixFullSlice<f64>, mat_b: &MatrixFullSlice<f64>) -> Vec<f64> {
    let a_y = mat_a.size.get(1).unwrap();
    let b_y = mat_b.size.get(1).unwrap();
    let a_x = mat_a.size.get(0).unwrap();
    let b_x = mat_b.size.get(0).unwrap();
    if (*a_x == 0 || *b_x ==0) {return vec![0.0;*a_y.min(b_y)]};
    let mut out_vec = vec![0.0;*a_y.min(b_y)];

    mat_a.iter_columns_full().zip(mat_b.iter_columns_full())
    .map(|(mat_a_p, mat_b_p)| (mat_a_p,mat_b_p))
    .zip(out_vec.iter_mut())
    .for_each(|((mat_a_p,mat_b_p),out_vec_p)| {
        *out_vec_p = mat_a_p.iter().zip(mat_b_p.iter())
            .fold(0.0, |acc, (mat_a_ip, mat_b_ip)| 
            {acc + mat_a_ip*mat_b_ip});
    });
    out_vec
}
//i, j -> ij
//vec_a: column vec of i rows, vec_b: row vec of j columns
//produces matrix of i,j
pub fn _einsum_03(vec_a: &[f64], vec_b: &[f64]) -> MatrixFull<f64> {

    let i_len = vec_a.len();
    let j_len = vec_b.len();
    if (i_len == 0 || j_len ==0) {return MatrixFull::new([i_len,j_len],0.0)};
    let mut om = MatrixFull::new([i_len,j_len],0.0);

    om.iter_columns_full_mut().zip(vec_b.iter())
    .map(|(om_j, vec_b)| {(om_j, vec_b)})
    .for_each(|(om_j, vec_b)| {
        om_j.iter_mut().zip(vec_a.iter()).for_each(|(om_ij, vec_a)| {
            *om_ij = *vec_a*vec_b
        });
    });

    om
}

pub fn _einsum_03_forvec(vec_a: &Vec<f64>, vec_b: &Vec<f64>) -> MatrixFull<f64> {

    let i_len = vec_a.len();
    let j_len = vec_b.len();
    if (i_len == 0 || j_len ==0) {return MatrixFull::new([i_len,j_len],0.0)};
    let mut om = MatrixFull::new([i_len,j_len],0.0);

    om.iter_columns_full_mut().zip(vec_b.iter())
    .map(|(om_j, vec_b)| {(om_j, vec_b)})
    .for_each(|(om_j, vec_b)| {
        om_j.iter_mut().zip(vec_a.iter()).for_each(|(om_ij, vec_a)| {
            *om_ij = *vec_a*vec_b
        })
    });

    om
}

#[test]
fn test_einsum_02() {
    let mut mat_a = MatrixFull::from_vec([2,2],vec![3.0,4.0,2.0,6.0]).unwrap();
    let mut mat_b = mat_a.clone();
    let mut mat_c = _einsum_02_rayon(&mat_a.to_matrixfullslice(), &mat_b.to_matrixfullslice());
    println!("{:?}", mat_c);

}

#[test]
fn test_sqrt_inverse() {
    let mut mat_a = MatrixFull::from_vec([3,3], vec![
         4.0,  12.0, -16.0,
        12.0,  37.0, -43.0,
       -16.0, -43.0,  98.0
    ]).unwrap();

    let sqrt_inverse_mat_a = mat_a.to_matrixfullslicemut().lapack_power(-0.5, 1e-10).unwrap();

    sqrt_inverse_mat_a.formated_output(3, "full");

    let mut inverse_01 = MatrixFull::new([3,3],0.0);

    inverse_01.to_matrixfullslicemut().lapack_dgemm(
        &sqrt_inverse_mat_a.to_matrixfullslice(),
        &sqrt_inverse_mat_a.to_matrixfullslice(),'N','N',1.0,0.0);

    inverse_01.formated_output(3,"full");

    let cholesky_inverse = mat_a.to_matrixfullslicemut().cholesky_decompose_inverse('L').unwrap();

    cholesky_inverse.formated_output(3, "full");

    let mut inverse_02 = MatrixFull::new([3,3],0.0);

    inverse_02.to_matrixfullslicemut().lapack_dgemm(
        &cholesky_inverse.to_matrixfullslice(),
        &cholesky_inverse.to_matrixfullslice(),'N','T',1.0,0.0);

    inverse_02.formated_output(3,"full");

    let cholesky_inverse = mat_a.to_matrixfullslicemut().cholesky_decompose_inverse('U').unwrap();

    cholesky_inverse.formated_output(3, "full");

    let mut inverse_03 = MatrixFull::new([3,3],0.0);

    inverse_03.to_matrixfullslicemut().lapack_dgemm(
        &cholesky_inverse.to_matrixfullslice(),
        &cholesky_inverse.to_matrixfullslice(),'T','N',1.0,0.0);

    inverse_02.formated_output(3,"full");


}

#[test]
fn test_symm_dgemm() {
    let mut mat_a = MatrixFull::from_vec([3,3], vec![
         4.0,  12.0, -16.0,
        12.0,  37.0, -43.0,
       -16.0, -43.0,  98.0
    ]).unwrap();
    let mut mat_b = MatrixFull::from_vec([3,3], vec![
         4.0,  10.0, -10.0,
        10.0,  37.0, -23.0,
       -10.0, -23.0,  98.0
    ]).unwrap();
    let mut mat_c = MatrixFull::new([3,3],0.0);
    _dgemm(&mat_a, (0..3,0..3), 'N', 
        &mat_b, (0..3,0..3), 'N', 
        &mut mat_c, (0..3,0..3), 1.0, 0.0);
    mat_c.formated_output(3, "full");
    println!("{},{},{}",(0..20).start, (0..20).end, (0..20).len());
}
