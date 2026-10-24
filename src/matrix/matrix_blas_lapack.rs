use std::{iter::Flatten, vec::IntoIter, ops::Range, ptr::addr_of_mut};
use std::os::raw::c_int;

use blas::{dgemm,dtrmm, dsymm, dsyrk, dgemv};
use itertools::Itertools;
use lapack::{dgees, dgeev, dgeqp3, dgesv, dgesvd, dgetrf, dgetri, dlamch, dormqr, dpotrf, dpotri, dspevx, dspgvx, dsyev, dsyevd, dsyevx, dtrtrs, dtrtri, Select2F64};
use nalgebra::{matrix, Matrix};
use rayon::prelude::*;

use crate::{basic_check_shape, BasicMatUp, BasicMatrix, MatrixFull, MatrixFullSlice, MatrixFullSliceMut, MatrixFullSliceMut2, MatrixUpperSliceMut, TensorSlice, TensorSliceMut, SAFE_MINIMUM};

use super::ParMathMatrix;


#[cfg(not(feature="intel-mkl"))]
extern "C" {
    pub fn openblas_get_num_threads() -> ::std::os::raw::c_int;
    pub fn openblas_set_num_threads(n: ::std::os::raw::c_int);
    pub fn goto_get_num_threads() -> ::std::os::raw::c_int;
    pub fn goto_set_num_threads(n: ::std::os::raw::c_int);
    pub fn omp_get_max_threads() -> ::std::os::raw::c_int;
    pub fn omp_set_num_threads(n: ::std::os::raw::c_int);
}
#[cfg(feature="intel-mkl")]
extern "C" {
    pub fn mkl_set_num_threads(n: *const c_int);
    pub fn mkl_set_num_threads_local(n: c_int) -> c_int;
    pub fn mkl_get_max_threads() -> c_int;
}



pub fn omp_get_num_threads_wrapper() -> usize {
    let mut num_threads = 1;
    #[cfg(not(feature="intel-mkl"))] {
        num_threads = unsafe{omp_get_max_threads()} as usize;
    }
    #[cfg(feature="intel-mkl")] {
        num_threads = unsafe{mkl_get_max_threads()} as usize;
    }
    num_threads
}

/// Set number of threads to control effective thread usage of BLAS.
///
/// This function is designed to be thread-local and thread-safe. It is expected to be
/// called before using BLAS function. In usual cases,
/// - When inside rayon spawned parallel threads, use `n=1` to avoid thread overscription; 
/// - When in main thread, use `n=num_threads` to call BLAS in multithread. The `num_thread`
///   is usually specified by user input card, or obtained by rayon global thread numbers.
///
/// # Note
///
/// Only OpenBLAS is verified.
pub fn omp_set_num_threads_wrapper(n: usize) {
    #[cfg(not(feature="intel-mkl"))]
    unsafe {
        // OpenMP-built OpenBLAS (default): `omp_set_num_threads` writes the calling
        // thread's ICV only, which is exactly right inside rayon workers (n=1) and
        // on the main thread (n=default). Do NOT use openblas_set_num_threads here:
        // on an OpenMP build it is a process-global setting, and pinning it to 1
        // from a rayon worker serializes every later BLAS call (measured: the RI-K
        // build went 15s -> 321s per SCF iteration on ts2).
        #[cfg(not(feature = "openblas-pthread"))]
        omp_set_num_threads(n as ::std::os::raw::c_int);
        // pthread-built OpenBLAS: the OpenMP API is a no-op, use the atomic global.
        #[cfg(feature = "openblas-pthread")]
        openblas_set_num_threads(n as ::std::os::raw::c_int);
    }
    #[cfg(feature="intel-mkl")]
    unsafe {
        // mkl_set_num_threads() is a GLOBAL (process-wide) setting and takes a
        // global lock; calling it from every rayon worker races with the restore
        // calls and can force MKL to reconfigure its thread pool. The thread-local
        // variant matches omp_set_num_threads() semantics: it overrides the thread
        // count for the calling thread only. n=0 would clear the override.
        mkl_set_num_threads_local(n as i32);
    }
}

/// Set number of threads in global/main thread.
///
/// This function is designed to be called when building global thread pool.
/// It is not designed to be called inside parallel (spawned threads from rayon).
pub fn omp_set_num_threads_global_wrapper(n: usize) {
    #[cfg(not(feature="intel-mkl"))]
    unsafe {
        // Which library is loaded decides which of the two calls has any effect, and that is
        // not known when this crate is compiled: a payload may ship either the OpenMP-built or
        // the pthread-built OpenBLAS. Both calls are safe here, because the function is only
        // called on the main thread before any parallel region is entered:
        //   * OpenMP-built OpenBLAS reacts to `omp_set_num_threads` (thread-local ICV, and this
        //     call also sets the value inherited by the rayon workers);
        //   * pthread-built OpenBLAS (`libopenblasp*`, as in the current SIF) ignores the OpenMP
        //     API completely and only honours `openblas_set_num_threads`.
        // Emitting only the OpenMP call silently leaves a pthread build at its auto-detected
        // (= number of cores) thread count, which then multiplies with the worker pool. Measured
        // on 2x56 cores, 268 atoms / def2-SVP / B3LYP / IEFPCM: 224 OS threads at 55-86% CPU
        // utilisation and a 471 s RI-K build per SCF iteration.
        omp_set_num_threads(n as ::std::os::raw::c_int);
        openblas_set_num_threads(n as ::std::os::raw::c_int);
    }
    #[cfg(feature="intel-mkl")]
    unsafe {
        let tmp_n = n as i32;
        mkl_set_num_threads(&tmp_n as *const c_int);
    }
}

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


/// Level 2 BLAS
pub fn _dgemv<'a, T>(matr_a: &T, vec_x: &[f64], vec_y: &mut [f64], trans: char, alpha: f64, beta: f64, incx: i32, incy: i32) 
where T: BasicMatrix<'a, f64> 
{

    let m = matr_a.size()[0];
    let n = matr_a.size()[1];
    let lda = m.max(1);

    let check_shape_x = if trans.to_string().to_lowercase().eq("n") {
        vec_x.len() == 1 + (n-1)*incx.abs() as usize
    } else {
        vec_x.len() == 1 + (m-1)*incx.abs() as usize
    }; 

    let check_shape_y = if trans.to_string().to_lowercase().eq("n") {
        vec_y.len() == 1 + (m-1)*incy.abs() as usize
    } else {
        vec_y.len() == 1 + (n-1)*incy.abs() as usize
    }; 

    // check the shapes of the input matrices for the dgemm operation
    if ! (check_shape_x & check_shape_y) {panic!("ERROR:: Matr_A[{:},{:},{:}] * Vec_X[{:}] -> Vec_Y[{:}]",
         m, n, &trans,
         vec_x.len(), 
         vec_y.len() 
        )
    }

    unsafe{
        dgemv(trans as u8, m as i32, n as i32, alpha, matr_a.data_ref().unwrap(), lda as i32, vec_x, incx as i32, beta, vec_y, incy as i32);
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
            matr_a.data_ref().unwrap(), &matr_a.size(), sub_a_dim.0, sub_a_dim.1, opa, 
            matr_b.data_ref().unwrap(), &matr_b.size(), sub_b_dim.0, sub_b_dim.1, opb, 
            matr_c.data_ref_mut().unwrap(), &matr_c_size, sub_c_dim.0, sub_c_dim.1, 
            alpha, beta)
    } else {
        panic!("the matrixs into the dgemm function should be all stored in a contiguous memory block");
    }

}

pub fn _dgemm_full<'a, T,Q,P>(
    matr_a: &T, opa: char, 
    matr_b: &Q, opb: char, 
    matr_c: &mut P, alpha: f64, beta: f64,
) 
where T: BasicMatrix<'a, f64>,
      Q: BasicMatrix<'a, f64>, 
      P: BasicMatrix<'a, f64>
{
    let size_a = matr_a.size();
    let sub_a_dim = (0..size_a[0], 0..size_a[1]);
    let size_b = matr_b.size();
    let sub_b_dim = (0..size_b[0], 0..size_b[1]);
    let size_c = matr_c.size();
    let sub_c_dim = (0..size_c[0], 0..size_c[1]);
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
    let m = if opa=='N' {size_a[0]} else {size_a[1]};
    let k = if opa=='N' {size_a[1]} else {size_a[0]};
    let n = if opb=='N' {size_b[1]} else {size_b[0]};
    let l = if opa=='N' {size_b[0]} else {size_a[1]};
    let lda = if opa=='N' {m.max(1)} else {k.max(1)};
    let ldb = if opb=='N' {k.max(1)} else {n.max(1)};
    let ldc = m.max(1);
    // check the shapes of the input matrices for the dgemm operation
    if ! check_shape {panic!("ERROR:: Matr_A[{:},{:},{:}] * Matr_B[{:},{:},{:}] -> Matr_C[{:},{:}]",
         sub_a_dim.0.len(), sub_a_dim.1.len(), &opa,
         sub_b_dim.0.len(), sub_b_dim.1.len(), &opb,
         sub_c_dim.0.len(), sub_c_dim.1.len()
        )
    }

    let is_contiguous = matr_a.is_contiguous() && matr_b.is_contiguous() && matr_c.is_contiguous();
    if is_contiguous {
        //let matr_c_size = [matr_c.size()[0],matr_c.size()[1]];
        unsafe{
        dgemm(opa as u8,
              opb as u8,
              m as i32,
              n as i32,
              k as i32,
              alpha,
              matr_a.data_ref().unwrap(),
              lda as i32,
              matr_b.data_ref().unwrap(),
              ldb as i32,
              beta,
              matr_c.data_ref_mut().unwrap(),
              ldc as i32);
        }
        //crate::external_libs::general_dgemm_f(
        //    matr_a.data_ref().unwrap(), matr_a.size(), sub_a_dim.0, sub_a_dim.1, opa, 
        //    matr_b.data_ref().unwrap(), matr_b.size(), sub_b_dim.0, sub_b_dim.1, opb, 
        //    matr_c.data_ref_mut().unwrap(), &matr_c_size, sub_c_dim.0, sub_c_dim.1, 
        //    alpha, beta)
    } else {
        panic!("the matrixs into the dgemm function should be all stored in a contiguous memory block");
    }

}


#[inline]
pub fn _dgemm_scaled<'a, T,Q> (
    matr_a: &T, opa: char, 
    matr_b: &Q, opb: char, 
    alpha: f64
) -> MatrixFull<f64>
where T: BasicMatrix<'a, f64>,
      Q: BasicMatrix<'a, f64> {
    let axy = matr_a.size();
    let bxy = matr_b.size();
    if (axy[0]==0||bxy[1]==0) {return MatrixFull::new([axy[0],bxy[1]],0.0_f64)};
    let mut c = match (&opa, &opb) {
        ('N','N') => MatrixFull::new([axy[0],bxy[1]], 0.0_f64),
        ('T','N') => MatrixFull::new([axy[1],bxy[1]], 0.0_f64),
        ('N','T') => MatrixFull::new([axy[0],bxy[0]], 0.0_f64),
        ('T','T') => MatrixFull::new([axy[1],bxy[0]], 0.0_f64),
        _ => MatrixFull::new([0,0], 0.0_f64),
    };
    
    
    MatrixFull::new([axy[0],bxy[1]], 0.0_f64);
    _dgemm_full(matr_a, opa, matr_b, opb, &mut c, alpha, 0.0);
    c
}

/// Solve the triangular system `op(A) · X = B` in place on the right-hand side matrix
/// `matr_b` (`op(A)` is `A` for `transa = 'N'`, or `A^T` for `'T'`/`'C'`), where
/// `matr_a` is a square triangular matrix of which only the specified triangle
/// (`uplo = 'U'`/`'L'`) is read. This is the LAPACK `dtrtrs` driver, i.e. the
/// solve-only counterpart of `_dgemm` for triangular factors.
///
/// Returns `true` on success and `false` if the triangular matrix is singular.
///
/// Example: with the upper Cholesky factor `U` of `J = U^T U`, solving
/// `U^T · X = B` in place (`_dtrtrs(&u, &mut b, 'U', 'T', 'N')`) yields
/// `X = L^{-1} · B` with `L = U^T`.
pub fn _dtrtrs<'a, T, Q>(
    matr_a: &T,
    matr_b: &mut Q,
    uplo: char,
    transa: char,
    diag: char,
) -> bool
where
    T: BasicMatrix<'a, f64>,
    Q: BasicMatrix<'a, f64>,
{
    let n = matr_a.size()[0];
    let nrhs = matr_b.size()[1];
    let lda = n as i32;
    let ldb = matr_b.size()[0] as i32;
    let mut info = 0_i32;
    unsafe {
        dtrtrs(
            uplo as u8,
            transa as u8,
            diag as u8,
            n as i32,
            nrhs as i32,
            matr_a.data_ref().unwrap(),
            lda,
            matr_b.data_ref_mut().unwrap(),
            ldb,
            &mut info,
        );
    }
    info == 0
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
pub fn _dsyev<'a, T>(matr_a: &T, jobz: char) -> (Option<MatrixFull<f64>>, Vec<f64>, i32)
where
    T: BasicMatrix<'a, f64>,
{
    //let ndim = matr_a.size()[0];
    //let size = [ndim, matr_a.size()[1]];
    let size = matr_a.size();
    let ndim = size[0];
    if size[0] == size[1] {
        let n = ndim as i32;
        let mut a = matr_a
            .data_ref()
            .unwrap()
            .iter()
            .map(|x| *x)
            .collect::<Vec<f64>>();
        let mut w: Vec<f64> = vec![0.0; ndim];
        let mut work: Vec<f64> = vec![0.0; 4 * ndim];
        let lwork = 4 * n;
        let mut info = 0;
        unsafe {
            dsyev(
                jobz.clone() as u8,
                b'L',
                n,
                &mut a,
                n,
                &mut w,
                &mut work,
                lwork,
                &mut info,
            );
        }
        if info < 0 {
            panic!(
                "Error in _dsyev: the {}-th argument had an illegal value",
                -info
            );
        } else if info > 0 {
            panic!("Error in _dsyev: the algorithm failed to converge; {}-off-diagonal elements of an intermediate tridiagonal form did not converge to zero", -info);
        }
        let eigenvectors = if jobz.eq(&'V') {
            Some(MatrixFull::from_vec([ndim, ndim], a).unwrap())
        } else {
            None
        };
        //let eigenvalues = Tensors::from_vec(String::from("full"),vec![self.size[0]], w);
        (eigenvectors, w, n)
    } else {
        panic!("Error in _dsyev: the algorithm is only vaild for real symmetric matrices")
    }
}

/// Consuming, zero-copy version of [`_dsyev`]: the input matrix buffer is reused
/// for the eigenvectors (LAPACK dsyev with jobz='V' overwrites the matrix with
/// the eigenvectors), avoiding the n² copy that `_dsyev` performs internally.
/// Saves one n×n buffer per call — significant when many calls run concurrently.
pub fn _dsyev_inplace(
    matr_a: MatrixFull<f64>,
    jobz: char,
) -> (Option<MatrixFull<f64>>, Vec<f64>, i32) {
    let size = matr_a.size();
    let ndim = size[0];
    if size[0] == size[1] {
        let n = ndim as i32;
        let mut a = matr_a.data; // take the buffer without copying
        let mut w: Vec<f64> = vec![0.0; ndim];
        let mut work: Vec<f64> = vec![0.0; 4 * ndim];
        let lwork = 4 * n;
        let mut info = 0;
        unsafe {
            dsyev(
                jobz as u8,
                b'L',
                n,
                &mut a,
                n,
                &mut w,
                &mut work,
                lwork,
                &mut info,
            );
        }
        if info < 0 {
            panic!(
                "Error in _dsyev_inplace: the {}-th argument had an illegal value",
                -info
            );
        } else if info > 0 {
            panic!("Error in _dsyev_inplace: the algorithm failed to converge; {}-off-diagonal elements of an intermediate tridiagonal form did not converge to zero", info);
        }
        let eigenvectors = if jobz.eq(&'V') {
            Some(MatrixFull::from_vec([ndim, ndim], a).unwrap())
        } else {
            None
        };
        (eigenvectors, w, n)
    } else {
        panic!("Error in _dsyev_inplace: the algorithm is only vaild for real symmetric matrices")
    }
}

pub fn _dsyevd<'a, T>(matr_a: &T, jobz: char) -> (Option<MatrixFull<f64>>, Vec<f64>, i32)
where
    T: BasicMatrix<'a, f64>,
{
    //let ndim = matr_a.size()[0];
    //let size = [ndim, matr_a.size()[1]];
    let size = matr_a.size();
    let ndim = size[0];
    if size[0] == size[1] {
        let n = ndim as i32;
        let mut a = matr_a
            .data_ref()
            .unwrap()
            .iter()
            .map(|x| *x)
            .collect::<Vec<f64>>();
        let mut w: Vec<f64> = vec![0.0; ndim];
        let lwork = if jobz.eq(&'N') {
            4 * n
        } else {
            7 * n + 2 * n * n
        };
        let mut work: Vec<f64> = vec![0.0; lwork as usize];
        let liwork = if jobz.eq(&'N') { 2 } else { 4 + 5 * n };
        let mut iwork: Vec<i32> = vec![0; liwork as usize];
        let mut info = 0;
        unsafe {
            dsyevd(
                jobz.clone() as u8,
                b'L',
                n,
                &mut a,
                n,
                &mut w,
                &mut work,
                lwork,
                &mut iwork,
                liwork,
                &mut info,
            );
        }
        if info < 0 {
            panic!(
                "Error in _dsyevd: the {}-th argument had an illegal value",
                -info
            );
        } else if info > 0 {
            panic!("Error in _dsyevd: the algorithm failed to converge; {}-off-diagonal elements of an intermediate tridiagonal form did not converge to zero", -info);
        }
        let eigenvectors = if jobz.eq(&'V') {
            Some(MatrixFull::from_vec([ndim, ndim], a).unwrap())
        } else {
            None
        };
        //let eigenvalues = Tensors::from_vec(String::from("full"),vec![self.size[0]], w);
        (eigenvectors, w, n)
    } else {
        panic!("Error in _dsyev: the algorithm is only vaild for real symmetric matrices")
    }
}
/// # DSYMM performs one of the matrix-matrix operations  
///    C := alpha*A*B + beta*C,  
/// or  
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
    //let m = matr_c.size()[0] as i32;
    //let n = matr_c.size()[1] as i32;
    let matr_c_size = matr_c.size();
    let m = matr_c_size[0] as i32;
    let n = matr_c_size[1] as i32;

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

/// # DSYRK performs one of the matrix-matrix operations  
///    C := alpha*A*A**T + beta*C,  
/// or  
///    C := alpha*A**T*A + beta*C,  
/// where alpha and beta are scalars, A is a symmetric matrix and B and C are m by n matrices
/// 
/// Two important keywords:
/// uplo =  'U' or 'u' Only the upper triangular part of C is to be referenced
///         'L' or 'l' Only the lower triangular part of C is to be referenced
/// trans = 'N' or 'n' C := alpha*A*A**T + beta*C  
///         'T' or 't' C := alpha*A**T*A + beta*C  
/// 
pub fn _dsyrk<'a, T, P> (
    matr_a: &T, matr_c: &mut P,
    uplo: char, trans: char, 
    alpha: f64, beta: f64
)
where T: BasicMatrix<'a, f64>,
      P: BasicMatrix<'a, f64>
{
    //let m = matr_c.size()[0] as i32;
    //let n = matr_c.size()[1] as i32;
    let matr_c_size = matr_c.size();
    let m = matr_c_size[0] as i32;
    let n = matr_c_size[1] as i32;
    if m!=n {panic!("matr_b should be symmetric")};

    let k = if trans.to_string().to_lowercase().eq("n") {matr_a.size()[1] as i32} else {matr_a.size()[0] as i32};
    let lda = if trans.to_string().to_lowercase().eq("n") {n.max(1)} else {k.max(1)}; 
    let ldc = n.max(1);


    unsafe {
        dsyrk(uplo as u8, trans as u8, n, k, alpha, matr_a.data_ref().unwrap(), lda, beta, matr_c.data_ref_mut().unwrap(),ldc)
    }

}


/// # [`_dgees`] computs for an N-by-N real nonsymmetric matrix A, the eigenvalues, the real Schur form T, and optionally, the matrix of Schur vectors Z. This gives the Schur factorization A = Z*T*(T**T)
/// 
/// Optionally, it also orders the eigenvalues on the diagonal of the
/// real Schur form so that selected eigenvalues are at the top left.
/// The leading columns of Z then form an orthonormal basis for the
/// invariant subspace corresponding to the selected eigenvalues.
///
/// A matrix is in real Schur form if it is upper quasi-triangular with
/// 1-by-1 and 2-by-2 blocks. 2-by-2 blocks will be standardized in the
/// form
///         [  a  b  ]
///         [  c  a  ]
///
/// where b*c < 0. The eigenvalues of such a block are a +- sqrt(bc).
/// 
/// Example
/// ```
///    use rest_tensors::{MatrixFull, MatrixUpper};
///    use rest_tensors::matrix::matrix_blas_lapack::_dgees;
///    // generate a matrix with only the upper triangle of A is stored in `MatrixUpper`.
///    let matr_0 = MatrixUpper::from_vec(6, vec![4.0,12.0,37.0,-16.0,-43.0,98.0]).unwrap();
///    // transfer it to a MatrixFull format and store in matr_a
///    let mut matr_a = matr_0.to_matrixfull().unwrap();
///    //println!("{:?}", &matr_a);
///    //            |  4.0 | 12.0 |-16.0 |
///    // matr_a =   | 12.0 | 37.0 |-43.0 |
///    //            |-16.0 |-43.0 | 98.0 |
///    let (matr_b, vs, wr, wi) = _dgees(&mut matr_a, 'V', 'N', None);
///    //let U = matr_a.iter_matrixupper().unwrap().map(|x| *x).collect::<Vec<f64>>();
///    //assert_eq!(U, vec![2.0,6.0,1.0,-8.0,5.0,3.0]);
///    matr_b.formated_output(3, "T");
///    vs.formated_output(3,"Eigenvector");
///    println!("{:?}", &wr);
///    println!("{:?}", &wi);
///    
/// ```
/// 
pub fn _dgees<'a, T>(matr: &T, jobvs: char, sort: char, select: Select2F64)  -> (MatrixFull<f64>, MatrixFull<f64>, Vec<f64>, Vec<f64>, usize)
where T: BasicMatrix<'a, f64> {
    let matr_size = matr.size();
    let mut matr_a = unsafe{MatrixFull::from_vec_unchecked(
        matr_size,
        matr.data_ref().unwrap().iter().map(|x| *x).collect())};
    let n0 = matr_a.size()[0];
    let n = n0 as i32;

    let lda = n;
    let ldvs = if jobvs.to_string().to_lowercase().eq("v") {n} else {1};
    let mut vs = MatrixFull::new([ldvs as usize,n0],0.0);
    let mut sdim = if sort.to_string().to_lowercase().eq("s") {n} else {0};;
    let mut wr: Vec<f64> = vec![0.0;n0];
    let mut wi: Vec<f64> = vec![0.0;n0];
    let mut work: Vec<f64> = vec![0.0;4*n0];
    let lwork = 4*n;
    let mut bwork: Vec<i32> = vec![0;n0];
    let mut info = 0;
    //let a = matr_a.data_ref_mut().unwrap();

    unsafe{dgees(
        jobvs as u8, 
        sort as u8, 
        select, 
        n, 
        matr_a.data_ref_mut().unwrap(), 
        lda, 
        &mut sdim, 
        &mut wr, 
        &mut wi, 
        vs.data_ref_mut().unwrap(), 
        ldvs, 
        &mut work, 
        lwork, 
        &mut bwork, 
        &mut info)}

    (matr_a, vs, wr, wi, sdim as usize)
    //MatrixFull::new([10,10],0.0)
}
pub fn _dgeev<'a, T>(
    matr: &T, 
    jobvl: char, 
    jobvr: char
) -> (MatrixFull<f64>, Vec<f64>, Vec<f64>, MatrixFull<f64>, MatrixFull<f64>, i32)
where 
    T: BasicMatrix<'a, f64> 
{
    let matr_size = matr.size();
    let mut matr_a = unsafe {
        MatrixFull::from_vec_unchecked(
            matr_size,
            matr.data_ref().unwrap().iter().copied().collect()
        )
    };
    
    let n0 = matr_a.size()[0];
    let n = n0 as i32;
    
    let lda = n;
    
    // 根据是否计算左/右特征向量确定矩阵维度
    let ldvl = if jobvl.to_ascii_uppercase() == 'V' { n } else { 1 };
    let ldvr = if jobvr.to_ascii_uppercase() == 'V' { n } else { 1 };
    
    // 初始化输出数组
    let mut wr: Vec<f64> = vec![0.0; n0];
    let mut wi: Vec<f64> = vec![0.0; n0];
    let mut vl = MatrixFull::new([ldvl as usize, n0], 0.0);
    let mut vr = MatrixFull::new([ldvr as usize, n0], 0.0);
    
    // 工作数组（可以优化大小，这里使用安全值）
    let lwork = 4 * n;
    let mut work: Vec<f64> = vec![0.0; lwork as usize];
    let mut info = 0;
    
    // 调用 dgeev
    unsafe {
        dgeev(
            jobvl as u8,
            jobvr as u8,
            n,
            matr_a.data_ref_mut().unwrap(),
            lda,
            &mut wr,
            &mut wi,
            vl.data_ref_mut().unwrap(),
            ldvl,
            vr.data_ref_mut().unwrap(),
            ldvr,
            &mut work,
            lwork,
            &mut info
        );
    }
    
    (matr_a, wr, wi, vl, vr, info)
}

/// # [`_dsvd`] computes the singular value decomposition (SVD) of a general real M-by-N matrix A
/// 
/// The SVD is written as A = U * Sigma * V^T
/// 
/// * `jobu`: char, 'A' or 'S' or 'N'
///   - `'A'`: all M columns of U are returned in array U
///   - `'S'`: the first min(M,N) columns of U are returned
///   - `'N'`: no columns of U are computed
/// * `jobvt`: char, 'A' or 'S' or 'N'
///   - `'A'`: all N rows of V^T are returned in array VT
///   - `'S'`: the first min(M,N) rows of V^T are returned
///   - `'N'`: no rows of V^T are computed
///
/// Returns (U, singular_values, VT) where:
/// - U: MatrixFull<f64> of size M x M (or M x min(M,N) for 'S')
/// - singular_values: Vec<f64> of length min(M,N), sorted in descending order
/// - VT: MatrixFull<f64> of size N x N (or min(M,N) x N for 'S')
///
/// Example
/// ```
///    use rest_tensors::{MatrixFull};
///    use rest_tensors::matrix::matrix_blas_lapack::_dsvd;
///    let matr_a = MatrixFull::from_vec([4,2], vec![
///        2.0, 4.0,
///        1.0, 3.0,
///        0.0, 0.0,
///        0.0, 0.0
///    ]).unwrap();
///    // A = | 2.0  1.0  0.0  0.0 |^T
///    //     | 4.0  3.0  0.0  0.0 |
///    let (u, mut s, vt) = _dsvd(&matr_a, 'A', 'A');
///    let s_bench = vec![5.464985704219043, 0.3659661906262578];
///    let diff_s = s.iter().zip(s_bench.iter()).fold(0.0, |acc,(a,b)| acc + (a-b).powf(2.0));
///    assert!(diff_s < 1e-7);
/// ```
pub fn _dsvd<'a, T>(matr: &T, jobu: char, jobvt: char) -> (MatrixFull<f64>, Vec<f64>, MatrixFull<f64>)
where
    T: BasicMatrix<'a, f64>
{
    let size = matr.size();
    let m = size[0];
    let n = size[1];
    let k = m.min(n);

    let mut a = matr.data_ref().unwrap().iter().copied().collect::<Vec<f64>>();
    let mut s: Vec<f64> = vec![0.0; k];

    let ldu = if jobu.to_ascii_uppercase() == 'A' { m.max(1) } else if jobu.to_ascii_uppercase() == 'S' { m.max(1) } else { 1usize };
    let ldvt = if jobvt.to_ascii_uppercase() == 'A' { n.max(1) } else if jobvt.to_ascii_uppercase() == 'S' { k.max(1) } else { 1usize };

    let u_cols = if jobu.to_ascii_uppercase() == 'A' { m } else if jobu.to_ascii_uppercase() == 'S' { k } else { 0 };
    let vt_rows = if jobvt.to_ascii_uppercase() == 'A' { n } else if jobvt.to_ascii_uppercase() == 'S' { k } else { 0 };

    let mut u = vec![0.0; ldu * u_cols];
    let mut vt = vec![0.0; ldvt * n];

    let lwork = (3 * k + m.max(n)).max(5 * k) as i32;
    let mut work: Vec<f64> = vec![0.0; lwork as usize];
    let mut info = 0;

    unsafe {
        dgesvd(
            jobu as u8,
            jobvt as u8,
            m as i32,
            n as i32,
            &mut a,
            m as i32,
            &mut s,
            &mut u,
            ldu as i32,
            &mut vt,
            ldvt as i32,
            &mut work,
            lwork,
            &mut info,
        );
    }

    if info < 0 {
        panic!("ERROR in _dsvd: the {}-th argument had an illegal value", -info);
    } else if info > 0 {
        panic!("ERROR in _dsvd: dgesvd did not converge, {} superdiagonals of an intermediate bidiagonal form did not converge to zero", info);
    }

    let u_mat = MatrixFull::from_vec([ldu, u_cols], u).unwrap();
    let vt_mat = MatrixFull::from_vec([ldvt, n], vt).unwrap();

    (u_mat, s, vt_mat)
}

/// # [`_dpotrf`] dpotrf computes the Cholesky factorization of a real symmetric positive definite matrix A  
/// uplo: char, `U` or `L`  
/// = `U`: Upper triangle of A is stored  
/// = `L`: Lower triangle of A is stored  
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
    //let size = [matr_a.size()[0], matr_a.size()[1]];
    let size = matr_a.size();
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

/// # Invert a square matrix **in place** (no O(n²) copy)
///
/// Performs LAPACK `dgetrf` (LU factorization) followed by `dgetri` (inversion)
/// directly on the matrix's data vector, eliminating the internal data copy that
/// [`_dinverse`] makes.
///
/// Only `ipiv` (n i32s) and `work` (4n f64s) are allocated as scratch.
///
/// Returns `true` on success, `false` on LAPACK failure (singular matrix, etc.).
///
/// **Warning**: the input matrix is overwritten with its inverse.
pub fn _dinverse_inplace(matr_a: &mut MatrixFull<f64>) -> bool {
    let size = matr_a.size;
    if size[0] != size[1] {
        println!("Error: The matrix for in-place inversion should be NxN, got {:?}",
                 size);
        return false;
    }
    let ndim = size[0];
    let n = ndim as i32;
    let mut ipiv: Vec<i32> = vec![0; ndim];
    let lwork = 4 * n;
    let mut work: Vec<f64> = vec![0.0; 4 * ndim];
    let mut info1 = 0;
    let mut info2 = 0;
    unsafe {
        dgetrf(n, n, &mut matr_a.data, n, &mut ipiv, &mut info1);
        if info1 != 0 {
            println!("Error in _dinverse_inplace: dgetrf info = {}", info1);
            return false;
        }
        dgetri(n, &mut matr_a.data, n, &mut ipiv,
               &mut work, lwork, &mut info2);
    }
    if info2 != 0 {
        println!("Error in _dinverse_inplace: dgetri info = {}", info2);
        return false;
    }
    true
}


/// # Solve a linear system Ax = b
///
/// Given an n×n matrix A and a vector b of length n, solve for x in Ax = b.
/// Uses the LAPACK subroutine dgesv, which performs LU factorization with partial pivoting.
///
/// Returns `Some(x)` on success, or `None` if A is singular.
///
/// Example
/// ```
///    use rest_tensors::MatrixFull;
///    use rest_tensors::matrix::matrix_blas_lapack::_dsolve;
///    // A = | 2.0 | 1.0 |
///    //     | 5.0 | 3.0 |
///    let matr_a = MatrixFull::from_vec([2,2], vec![2.0, 5.0, 1.0, 3.0]).unwrap();
///    // b = [4.0, 7.0]
///    let b = vec![4.0, 7.0];
///    let x = _dsolve(&matr_a, &b).unwrap();
///    // A*x should equal b: [2*1 + 1*2, 5*1 + 3*2] = [4, 7] ✓
///    assert!((x[0] - 1.0).abs() < 1e-10);
///    assert!((x[1] - 2.0).abs() < 1e-10);
/// ```
pub fn _dsolve<'a, T>(matr_a: &T, b: &[f64]) -> Option<Vec<f64>>
where T: BasicMatrix<'a, f64>
{
    let size = matr_a.size();
    if size[0] != size[1] {
        panic!("ERROR in _dsolve: matrix A must be square (n×n), got {:?}×{:?}", size[0], size[1]);
    }
    if b.len() != size[0] {
        panic!("ERROR in _dsolve: vector b length ({}) must equal matrix dimension ({})", b.len(), size[0]);
    }

    let ndim = size[0];
    let n = ndim as i32;
    let nrhs = 1_i32;

    // Copy A and b since dgesv overwrites them
    let mut a = matr_a.data_ref().unwrap().to_vec();
    let mut x = b.to_vec();
    let mut ipiv: Vec<i32> = vec![0; ndim];
    let mut info = 0;

    unsafe {
        dgesv(n, nrhs, &mut a, n, &mut ipiv, &mut x, n, &mut info);
    }

    if info < 0 {
        panic!("ERROR in _dsolve: the {}-th argument had an illegal value", -info);
    } else if info > 0 {
        println!("WARNING in _dsolve: matrix A is singular (U({},{}) = 0), factorization failed", info-1, info-1);
        return None;
    }

    Some(x)
}

/// # SVD-based Moore-Penrose pseudoinverse
///
/// Computes the pseudoinverse A^+ of a general real M-by-N matrix A using SVD.
///
/// The SVD factorizes A = U * Sigma * V^T. Singular values below
/// rcond * max(sigma) are treated as zero.
///
/// `rcond`: relative threshold for singular value truncation.
///   Use `None` for the default (f64::EPSILON).
///
/// Returns `Some(A^+)` where A^+ is an N-by-M matrix, or `None` on failure.
///
/// Example
/// ```
///    use rest_tensors::{MatrixFull};
///    use rest_tensors::matrix::matrix_blas_lapack::_pinv;
///    // A = | 1.0 | 2.0 |
///    //     | 3.0 | 4.0 |
///    //     | 5.0 | 6.0 |
///    let matr_a = MatrixFull::from_vec([3,2], vec![
///        1.0, 3.0, 5.0,
///        2.0, 4.0, 6.0
///    ]).unwrap();
///    let a_pinv = _pinv(&matr_a, None).unwrap();
///    // A^+ * A should approximately equal identity
///    let mut identity_approx = MatrixFull::new([2,2], 0.0);
///    crate::matrix::matrix_blas_lapack::_dgemm_full(
///        &a_pinv, 'N', &matr_a, 'N', &mut identity_approx, 1.0, 0.0);
///    let diff = (identity_approx[[0,0]] - 1.0).abs()
///             + (identity_approx[[0,1]] - 0.0).abs()
///             + (identity_approx[[1,0]] - 0.0).abs()
///             + (identity_approx[[1,1]] - 1.0).abs();
///    assert!(diff < 1e-10);
/// ```
pub fn _pinv<'a, T>(matr: &T, rcond: Option<f64>) -> Option<MatrixFull<f64>>
where T: BasicMatrix<'a, f64>
{
    let size = matr.size();
    let m = size[0];
    let n = size[1];

    if m == 0 || n == 0 {
        return Some(MatrixFull::new([n, m], 0.0));
    }

    let (u, s, vt) = _dsvd(matr, 'S', 'S');

    let rcond = rcond.unwrap_or(f64::EPSILON);
    let threshold = rcond * s[0];

    let s_inv: Vec<f64> = s.iter()
        .map(|&x| if x > threshold { 1.0 / x } else { 0.0 })
        .collect();

    let k = s.len();
    let mut u_mat = u;
    let vt_mat = vt;

    for j in 0..k {
        u_mat.iter_column_mut(j).for_each(|x| *x *= s_inv[j]);
    }

    let mut a_pinv = MatrixFull::new([n, m], 0.0);
    _dgemm_full(&vt_mat, 'T', &u_mat, 'T', &mut a_pinv, 1.0, 0.0);

    Some(a_pinv)
}

/// # QR-based Moore-Penrose pseudoinverse for square matrices
///
/// Computes A^+ using pivoted QR decomposition (dgeqp3) instead of SVD.
/// For full-rank matrices, this is equivalent to A^{-1}.
/// For rank-deficient matrices, diagonal entries of R below
/// rcond * |R[0,0]| are treated as zero.
///
/// The algorithm:
/// 1. Compute pivoted QR: A*P = Q*R
/// 2. Determine numerical rank r from |R[i,i]| threshold
/// 3. If r == n: A^+ = P * R^{-1} * Q^T
/// 4. If r < n:  A^+ = P * [R11^{-1}  -R11^{-1}*R12; 0 0] * Q^T
///
/// This is ~4-5x faster than the SVD-based pseudoinverse for large matrices.
///
/// `matr`: square matrix (M×M), destroyed in-place (overwritten by QR factors)
/// `x`: caller-provided n×n output buffer (e.g. `vec![0.0; n*n]`), overwritten with A^+
/// `rcond`: relative threshold for rank determination. Use `None` for f64::EPSILON.
///
/// Returns `Some(A^+)` (M×M), or `None` on failure.
///
/// Example
/// ```
///    use rest_tensors::{MatrixFull};
///    use rest_tensors::matrix::matrix_blas_lapack::_qrpinv;
///    let mut matr_a = MatrixFull::from_vec([3,3], vec![
///        1.0, 0.0, 2.0,
///        0.0, 3.0, 0.0,
///        2.0, 0.0, 1.0,
///    ]).unwrap();
///    let matr_orig = matr_a.clone();
///    let a_pinv = _qrpinv(&mut matr_a, vec![0.0f64; 9], None).unwrap();
///    let mut identity_approx = MatrixFull::new([3,3], 0.0);
///    crate::matrix::matrix_blas_lapack::_dgemm_full(
///        &a_pinv, 'N', &matr_orig, 'N', &mut identity_approx, 1.0, 0.0);
///    let diff = (identity_approx[[0,0]] - 1.0).abs()
///             + (identity_approx[[1,1]] - 1.0).abs()
///             + (identity_approx[[2,2]] - 1.0).abs();
///    assert!(diff < 1e-10);
/// ```
/// modified by deltah
/// used for ISDF
/// revise in-place instead of clone
pub fn _qrpinv(matr: &mut MatrixFull<f64>, mut x: Vec<f64>, rcond: Option<f64>) -> Option<MatrixFull<f64>>
{
    let size = matr.size();
    let n = size[0];
    if size[0] != size[1] {
        panic!("_qrpinv only works for square matrices, got {:?}×{:?}", n, size[1]);
    }
    if n == 0 {
        return Some(MatrixFull::new([0, 0], 0.0));
    }

    let rcond = rcond.unwrap_or(f64::EPSILON);

    // 1. Copy matrix and compute pivoted QR: A*P = Q*R
    // instead, use itself
    //let mut a = matr.data_ref().unwrap().to_vec();
    let mut jpvt = vec![0i32; n];
    let mut tau = vec![0.0f64; n];
    let mut info = 0i32;

    // workspace query
    let mut work = vec![0.0f64; 1];
    unsafe {
        dgeqp3(n as i32, n as i32, &mut matr.data, n as i32, &mut jpvt, &mut tau, &mut work, -1, &mut info);
    }
    if info != 0 {
        panic!("_qrpinv: dgeqp3 workspace query failed with info={}", info);
    }
    let lwork = work[0] as i32;
    work.resize(lwork.max(1) as usize, 0.0);
    unsafe {
        dgeqp3(n as i32, n as i32, &mut matr.data, n as i32, &mut jpvt, &mut tau, &mut work, lwork, &mut info);
    }
    if info != 0 {
        panic!("_qrpinv: dgeqp3 failed with info={}", info);
    }

    // 2. Determine numerical rank r
    let r = {
        let threshold = rcond * matr.data[0].abs();
        let mut count = 0usize;
        for i in 0..n {
            if matr.data[i + i * n].abs() > threshold {
                count += 1;
            } else {
                break;
            }
        }
        count
    };

    // 3. Build inverse permutation: inv_perm[jpvt[k]] = k
    let jpvt_0: Vec<usize> = jpvt.iter().map(|&x| (x - 1) as usize).collect();
    let mut inv_perm = vec![0usize; n];
    for (k, &j) in jpvt_0.iter().enumerate() {
        inv_perm[j] = k;
    }

    if r == 0 {
        // Rank is zero — return zero matrix
        return Some(MatrixFull::new([n, n], 0.0));
    }

    // from outside
    // let mut x = vec![0.0f64; n * n];

    if r == n {
        // === Full-rank case ===
        // R^{-1} in upper triangle of a, reflectors preserved in lower triangle
        unsafe {
            dtrtri(b'U' as u8, b'N' as u8, n as i32, &mut matr.data, n as i32, &mut info);
        }
        if info != 0 {
            return None;
        }

        // Build C = R^{-1}^T (lower triangle in column-major = transpose of upper)
        let mut c = vec![0.0f64; n * n];
        for j in 0..n {
            for i in 0..=j {
                // C[j,i] = a[i,j]  (transpose of upper triangular)
                c[j + i * n] = matr.data[i + j * n];
            }
        }
        // C = Q * C = Q * R^{-1}^T
        unsafe {
            dormqr(b'L' as u8, b'N' as u8, n as i32, n as i32, n as i32,
                &matr.data, n as i32, &tau, &mut c, n as i32, &mut work, -1, &mut info);
        }
        let lwork2 = work[0] as i32;
        work.resize(lwork2.max(1) as usize, 0.0);
        unsafe {
            dormqr(b'L' as u8, b'N' as u8, n as i32, n as i32, n as i32,
                &matr.data, n as i32, &tau, &mut c, n as i32, &mut work, lwork2, &mut info);
        }
        if info != 0 {
            return None;
        }

        // pinv[i,j] = C[j, inv_perm[i]]  (= (Q*R^{-1}T)^T[inv_perm[i], j] = (R^{-1}*Q^T)[inv_perm[i], j])
        for j in 0..n {
            for i in 0..n {
                x[i + j * n] = c[j + inv_perm[i] * n];
            }
        }
    } else {
    // === Rank-deficient case ===
    // Directly use x (n×n, already zeroed) to store R11 and R12, no extra allocation.

    // 1. Copy R11 (upper triangular part) into x[0..r, 0..r]
    for j in 0..r {
        for i in 0..=j {
            x[i + j * n] = matr.data[i + j * n];
        }
    }

    // 2. Copy R12 into x[0..r, r..n]
    for j in 0..(n - r) {
        for i in 0..r {
            x[i + (j + r) * n] = matr.data[i + (j + r) * n];
        }
    }

    // 3. Compute R11^{-1} in-place in the upper-left r×r block of x
    unsafe {
        dtrtri(
            b'U' as u8,
            b'N' as u8,
            r as i32,
            &mut x[..],   // pass whole x as mutable slice; lda = n ensures LAPACK only touches first r columns
            n as i32,
            &mut info,
        );
    }
    if info != 0 {
        return None;
    }

    // 4. Solve Z = R11^{-1} * R12, overwriting the R12 area in x.
    //    Split x into two non-overlapping mutable slices to avoid aliasing.
    let (a_area, b_area) = x.split_at_mut(r * n);
    // a_area: columns 0..r-1 (lda = n)  -> contains R11^{-1} in upper-left r×r
    // b_area: columns r..n-1 (lda = n)  -> contains R12 in upper r rows

    unsafe {
        dtrtrs(
            b'U' as u8,
            b'N' as u8,
            b'N' as u8,
            r as i32,
            (n - r) as i32,
            a_area,          // A = R11^{-1}, read-only (automatic reborrow from &mut [f64] to &[f64])
            n as i32,
            b_area,          // B = R12, mutable
            n as i32,
            &mut info,
        );
    }
    if info != 0 {
        return None;
    }

    // 5. Negate Z (still in b_area, upper r rows)
    for idx in 0..(r * (n - r)) {
        b_area[idx] = -b_area[idx];
    }

    // 6. Build C = X^T, where X = [R11^{-1}  -Z; 0 0], stored in x.
    let mut c = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            // c[j, i] = x[i, j]  (transpose)
            c[j + i * n] = x[i + j * n];
        }
    }

    // 7. Apply Q to C: C = Q * C = Q * X^T
    //    Query optimal work size first.
    unsafe {
        dormqr(
            b'L' as u8,
            b'N' as u8,
            n as i32,
            n as i32,
            n as i32,
            &matr.data[..],    // Q matrix
            n as i32,
            &tau[..],          // scalar factors
            &mut c[..],        // C
            n as i32,
            &mut work[..],     // workspace
            -1,
            &mut info,
        );
    }
    let lwork2 = work[0] as i32;
    work.resize(lwork2.max(1) as usize, 0.0);
    unsafe {
        dormqr(
            b'L' as u8,
            b'N' as u8,
            n as i32,
            n as i32,
            n as i32,
            &matr.data[..],
            n as i32,
            &tau[..],
            &mut c[..],
            n as i32,
            &mut work[..],
            lwork2,
            &mut info,
        );
    }
    if info != 0 {
        return None;
    }

    // 8. Apply permutation: pinv = P * C^T = P * X * Q^T
    //    Write back to x (the final result matrix)
    for j in 0..n {
        for i in 0..n {
            x[i + j * n] = c[j + inv_perm[i] * n];
        }
    }
    }
    Some(unsafe { MatrixFull::from_vec_unchecked([n, n], x) })
}

pub fn _power<'a, T>(matr_a: &T, p: f64, threshold: f64) -> Option<MatrixFull<f64>> 
where T: BasicMatrix<'a, f64>
{
    let size = matr_a.size();
    if size[0]==size[1] {

        // because lapack sorts eigenvalues from small to large, we multiply the elements with -1.0
        let mut om = MatrixFull::new([size[0],size[1]],0.0);
        om.data.iter_mut().zip(matr_a.data_ref().unwrap().iter()).for_each(|value| {*value.0=*value.1*-1.0f64});


        // diagonalize the matrix
        let (mut opt_eigenvector, mut eigenvalues, mut n) = _dsyevd(&om, 'V');
        let mut eigenvector = opt_eigenvector.unwrap();

        // now we get the eigenvectors with the eigenvalues from large to small
        let (mut n_nonsigular, mut tmpv) = (0usize,0.0);
        eigenvalues.iter_mut().for_each(|value| {
            *value = *value*(-1.0);
            //if *value >= threshold && *value <= tmpv {
            if *value >= threshold {
                n_nonsigular +=1;
            };
        });

        if n as usize != size[0] {
            panic!("Found unphysical eigenvalues");
        } else if n_nonsigular != size[0] {
            println!("debug: _power: n_nonsigular: {}", n_nonsigular);
        }

        //&eigenvector.data.par_chunks_exact_mut(size[0]).enumerate()
        eigenvector.iter_columns_full_mut().enumerate()
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

        
        _dgemm_full(&eigenvector, 'N', &eigenvector, 'T', &mut om, 1.0, 0.0);

        Some(om)
    } else {
        println!("Error: The matrix for power operations should be NxN");
        None
    }

}

pub fn _power_rayon_for_symmetric_matrix<'a, T>(matr_a: &T, p: f64, threshold: f64) -> Option<MatrixFull<f64>> 
where T: BasicMatrix<'a, f64>
{
    let size = matr_a.size();
    if size[0]==size[1] {

        // because lapack sorts eigenvalues from small to large, we multiply the elements with -1.0
        let mut om = MatrixFull::new([size[0],size[1]],0.0);
        om.data.par_iter_mut().zip(matr_a.data_ref().unwrap().par_iter()).for_each(|value| {*value.0=*value.1*-1.0f64});


        // diagonalize the matrix
        let (mut opt_eigenvector, mut eigenvalues, mut n) = _dsyevd(&om, 'V');
        let mut eigenvector = opt_eigenvector.unwrap();

        // now we get the eigenvectors with the eigenvalues from large to small
        let (mut n_nonsigular, mut tmpv) = (0usize,0.0);
        eigenvalues.iter_mut().for_each(|value| {
            *value = *value*(-1.0);
            //if *value >= threshold && *value <= tmpv {
            if *value >= threshold {
                n_nonsigular +=1;
            };
        });

        if n as usize != size[0] {
            panic!("Found unphysical eigenvalues");
        } else if n_nonsigular != size[0] {
            //println!("_power_rayon => n_nonsigular: {}, {}", n_nonsigular, size[0]);
            //println!("{:?}", &eigenvalues);
        }

        &eigenvector.data.par_chunks_exact_mut(size[0]).enumerate()
        //eigenvector.iter_columns_full_mut().enumerate()
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
        
        _dgemm_full(&eigenvector, 'N', &eigenvector, 'T', &mut om, 1.0, 0.0);

        Some(om)
    } else {
        println!("Error: The matrix for power operations should be NxN");
        None
    }

}
//i, j -> ij
//vec_a: column vec of i rows, vec_b: row vec of j columns

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

    pub fn lapack_dgetrf_full(&mut self) -> Option<(MatrixFull<f64>,Vec<i32>)> {
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
            Some((inv_mat, ipiv))
        } else {
            println!("Error: The matrix for inversion should be NxN");
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
                println!("ERROR: DPOTRF failed with {}", info);
                return None;
            }
            // compute the inverse of L 
            unsafe{dpotri(uplo.clone() as u8, n, &mut data,n, &mut info)};
            if info!=0 {
                println!("ERROR: DPOTRI failed with {}", info);
                return None
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
    let chunk_size = mat_a.size()[0];
    match scaling_factor {
        None =>  {
            mat_a.data_ref_mut().unwrap().chunks_exact_mut(chunk_size)
                .zip(mat_b.data_ref().unwrap().chunks_exact(chunk_size))
                .zip(slice_c.iter())
                .for_each(|((mat_a,mat_b),slice_c)| {
                    mat_a.iter_mut().zip(mat_b.iter()).for_each(|(mat_a, mat_b)| {
                        *mat_a += mat_b*slice_c
                });
            });
        },
        Some(s) => {
            mat_a.data_ref_mut().unwrap().chunks_exact_mut(chunk_size)
                .zip(mat_b.data_ref().unwrap().chunks_exact(chunk_size))
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

pub fn _sqrt_inverse_efficient() {

}
// A function to determine if an eigenvalue should be selected based on some criterion.
extern "C" fn select(ar: *const f64, ai: *const f64) -> i32 {
    unsafe{if ((*ar)*(*ar)).sqrt() < 1.0e-8 || ((*ai)*(*ai)).sqrt() > 1.0e-8 { 0 } else { 1 }} // Example criterion
    //unsafe{if *ar < 1.0e-8  { 0 } else { 1 }} // Example criterion
}

#[test]
fn test_dgees() {

    //let matr_0 = crate::MatrixUpper::from_vec(6, vec![4.0,12.0,37.0,-16.0,-43.0,98.0]).unwrap();
    //let matr_0 = crate::MatrixUpper::from_vec(6, vec![2.0,-1.0,2.0,0.0,-1.0,2.0]).unwrap();
    //let mut matr_a = matr_0.to_matrixfull().unwrap();
    let mut matr_a = MatrixFull::from_vec([3,3],vec![1.0,2.0,3.0,2.0,5.0,6.0,3.0,6.0,9.0]).unwrap();
    //let mut matr_a = MatrixFull::from_vec([3,3],vec![1.0,2.2,0.3,2.2,5.0,1.0,0.3,1.0,9.0]).unwrap();
    //let mut matr_a = MatrixFull::from_vec([3,3],vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
    println!("matr_a:");
    matr_a.formated_output(6, "full");

    let mut matr_b = MatrixFull::new([3,3],0.0);
    let mut matr_c = MatrixFull::new([3,3],0.0);
    let mut matr_d = MatrixFull::new([3,3],0.0);
    let mut matr_e = MatrixFull::new([3,3],0.0);


    // ================================= initial_guess with schur decomposition =========
    //let (mut matr_b, vs, wr, wi,sdim) = _dgees(&matr_a, 'V', 'S', Some(select));
    //matr_b.formated_output(3, "full");
    //vs.formated_output(3, "full");
    //println!("{:?}", &wr);
    //println!("{:?}", &wi);
    //println!("{:?}", sdim);
    ////let sdim = 3;

    //matr_b.iter_mut().for_each(|x| {
    //    if (*x).abs() < 1.0e-10 {
    //        *x = 0.0
    //    }
    //});

    //(sdim..matr_b.size()[0]).for_each(|i| {matr_b[(i,i)]=0.0});
    //matr_b.formated_output(3, "full");


    //_dgemm_full(&vs, 'N', &matr_b, 'N', &mut matr_d, 1.0, 0.0);
    //_dgemm_full(&matr_d, 'N', &vs, 'T', &mut matr_c, 1.0, 0.0);
    //matr_c.formated_output(3, "full");

    //let (mut matr_b, vs, wr, wi,sdim) = _dgees(&matr_c, 'V', 'S', Some(select));
    //matr_b.formated_output(3, "full");
    //vs.formated_output(3, "full");

    //let mut n_singular = 0_usize;
    //let mut converged_flag = false;
    ////let level_shift = vec![2.0,1.0,0.1,0.05];
    ////let level_shift = vec![1.0e-3,1.0e-5,1.0e-6, 1.0e-7, 1.0e-8, 0.0];
    //let level_shift = vec![1.0e-3, 1.0e-4, 0.0];
    //let sub_matr = MatrixFull::from_vec([sdim, sdim],matr_b.iter_submatrix(0..sdim, 0..sdim).map(|x| *x).collect_vec()).unwrap();
    //let mut sub_matr_c = MatrixFull::new([sdim,sdim],0.0);
    //let mut sub_matr_d = MatrixFull::new([sdim,sdim],0.0);

    //(sub_matr_c, n_singular, converged_flag) = _newton_schulz_inverse_square_root_v02(&sub_matr, 1.0e-8, 0.3, 20, &level_shift);

    //let mut matr_c = matr_b.clone();
    ////let mut matr_c = MatrixFull::new([3,3],0.0);

    //matr_c.iter_submatrix_mut(0..sdim, 0..sdim).zip(sub_matr_c.iter()).for_each(|(x,y)| {*x = *y});

    //matr_c.formated_output(3, "full");

    ////_dgemm_full(&matr_c, 'N', &matr_c, 'N', &mut matr_d, 1.0, 0.0);
    //println!("==== print final results ====");
    //println!("original matr_a:");
    //matr_a.formated_output(6, "full");

    //_dgemm_full(&vs, 'N', &matr_c, 'N', &mut matr_d, 1.0, 0.0);
    //_dgemm_full(&matr_d, 'N', &vs, 'T', &mut matr_b, 1.0, 0.0);
    //println!("\n matr_b = 1/sqrt(matr_a): inverse square root of matr_a by using newton_schulz algorithm:");
    //matr_b.formated_output(6, "full");

    //_dgemm_full(&matr_b, 'N', &matr_b, 'N', &mut matr_c, 1.0, 0.0);
    //println!("\n matr_c = matr_b * matr_b: inverse of matr_a:");
    //matr_c.formated_output(6, "full");

    //println!("\n matr_d = matr_c^(-1) = matr_a:");
    //matr_d = _power(&matr_c, -1.0, 1.0e-8).unwrap();
    //matr_d.formated_output(6, "full");

    //println!("\n matr_e = matr_a * matr_c = I");
    //_dgemm_full(&matr_a, 'N', &matr_c, 'N', &mut matr_e, 1.0, 0.0);
    //matr_e.formated_output(6, "full");
    // ================================= initial_guess with schur decomposition =========

    // ================================= plant initial_guess ========================
    let (mut matr_b, vs, wr, wi,sdim) = _dgees(&matr_a, 'V', 'S', Some(select));
    matr_b.formated_output(3, "full");
    vs.formated_output(3, "full");
    println!("{:?}", &wr);
    println!("{:?}", &wi);
    println!("{:?}", sdim);
    //let sdim = 3;

    matr_b.iter_mut().for_each(|x| {
        if (*x).abs() < 1.0e-10 {
            *x = 0.0
        }
    });

    (sdim..matr_b.size()[0]).for_each(|i| {matr_b[(i,i)]=0.0});
    matr_b.formated_output(3, "full");


    _dgemm_full(&vs, 'N', &matr_b, 'N', &mut matr_d, 1.0, 0.0);
    _dgemm_full(&matr_d, 'N', &vs, 'T', &mut matr_c, 1.0, 0.0);
    matr_c.formated_output(3, "full");
    let mut n_singular = 0;
    let mut converged_flag = false;
    let level_shift = vec![0.0];
    (matr_c, n_singular, converged_flag) = _newton_schulz_inverse_square_root_v02(&matr_c, 1.0e-8, 1.0, 100, &level_shift);
    _dgemm_full(&matr_c, 'N', &matr_c, 'N', &mut matr_d, 1.0, 0.0);
    println!("inverse of matr_a:");
    matr_d.formated_output(6, "full");
    _dgemm_full(&matr_d, 'N', &matr_a, 'N', &mut matr_c, 1.0, 0.0);
    matr_c.formated_output(6, "full");
    // ================================= plant initial_guess ========================



    //let (mut matr_b, vs, wr, wi) = _dgees(&mut matr_a, 'V', 'N', None);
    //println!("quasi triangular matrix T:");
    //matr_b.formated_output(6, "full");
    //println!("U and corresponding eigenvalues (real and image parts):");
    //vs.formated_output(6,"full");
    //println!("{:?}", &wr);
    //println!("{:?}", &wi);

    //_dgemm_full(&vs, 'N', &matr_b, 'N', &mut matr_c, 1.0, 0.0);
    //_dgemm_full(&matr_c, 'N', &vs, 'T', &mut matr_a, 1.0, 0.0);
    //println!("Repeat matr_a by U*T*U**T:");
    //matr_a.formated_output(6, "full");

    //// ================================= using _newton_schulz =========
    ////let mut sub_matr = MatrixFull::from_vec([2,2],matr_b.iter_submatrix(0..2, 0..2).map(|x| *x).collect_vec()).unwrap();
    //// perform the inverse of squart root 
    //let mut n_singular = 0_usize;
    //let mut converged_flag = false;
    //(matr_c, n_singular, converged_flag) = _newton_schulz_inverse_square_root(&matr_b, 1.0e-8, 0.3, 100);
    //matr_c.formated_output(6, "full");
    ////matr_c = matr_b.clone();
    ////matr_c.iter_submatrix_mut(0..2,0..2).zip(sub_matr.iter()).for_each(|(to, from)| {*to = *from});
    //_dgemm_full(&vs, 'N', &matr_c, 'N', &mut matr_b, 1.0, 0.0);
    //_dgemm_full(&matr_b, 'N', &vs, 'T', &mut matr_d, 1.0, 0.0);
    //_dgemm_full(&matr_d, 'N', &matr_d, 'N', &mut matr_c, 1.0, 0.0);
    //println!("inverse of matr_a:");
    //matr_c.formated_output(6, "full");
    //_dgemm_full(&matr_c, 'N', &matr_a, 'N', &mut matr_d, 1.0, 0.0);
    //matr_d.formated_output(6, "full");
    // ================================= using _newton_schulz =========


    // ================================= using _power =========
    //matr_c = _power(&matr_b, -0.5, 1.0e-8).unwrap();
    ////matr_c = matr_b.clone();
    ////matr_c.iter_submatrix_mut(0..2,0..2).zip(sub_matr.iter()).for_each(|(to, from)| {*to = *from});

    //_dgemm_full(&vs, 'N', &matr_c, 'N', &mut matr_b, 1.0, 0.0);
    //_dgemm_full(&matr_b, 'N', &vs, 'T', &mut matr_d, 1.0, 0.0);
    //println!("inverse of matr_a:");
    //matr_d.formated_output(6, "full");
    //_dgemm_full(&matr_d, 'N', &matr_d, 'N', &mut matr_c, 1.0, 0.0);
    //matr_c.formated_output(6, "full");
    // ================================= using _power =========


    //let mut sub_matr = MatrixFull::from_vec([2,2], matr_b.iter_submatrix(0..2, 0..2).map(|x| *x).collect()).unwrap();
    //sub_matr.formated_output(2, "full");

    //let mut sqrt_sub = _power_rayon(&sub_matr, 0.5, 1.0e-8).unwrap();
    //sqrt_sub.formated_output(2, "full");

    //sub_matr = _power(&sqrt_sub, 2.0, 1.0e-8).unwrap();
    //sub_matr.formated_output(2, "full");

    ////sqrt_sub[(1,0)] = 0.0;

    //matr_b.iter_submatrix_mut(0..2, 0..2).zip(sqrt_sub.data_ref().unwrap()).for_each(|(to, from)| {*to = *from});

    //_dgemm_full(&vs, 'N', &matr_b, 'N', &mut matr_c, 1.0, 0.0);

    //_dgemm_full(&matr_c, 'N', &vs, 'T', &mut matr_a, 1.0, 0.0);

    ////_dgemm_full(&matr_a, 'N', &matr_a, 'N', &mut matr_c, 1.0, 0.0);
    //matr_c = _power_rayon(&matr_a, 2.0, 1.0E-8).unwrap();

    //matr_c.formated_output(6, "full");

    // ===========================================================================================================
    ////let matr_0 = crate::MatrixUpper::from_vec(6, vec![1.0, 0.5, 2.0, 12.0,37.0,98.0]).unwrap();
    //let matr_0 = crate::MatrixUpper::from_vec(6, vec![4.0,12.0,37.0,-16.0,-43.0,98.0]).unwrap();
    ////let matr_0 = crate::MatrixUpper::from_vec(6, vec![2.0,-1.0,2.0,0.0,-1.0,2.0]).unwrap();
    //// transfer it to a MatrixFull format and store in matr_a
    //let mut matr_a = matr_0.to_matrixfull().unwrap();
    ////let mut matr_a = MatrixFull::from_vec([3,3],vec![2.0,1.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
    //let mut matr_b = MatrixFull::new([3,3],0.0);
    //let mut matr_c = MatrixFull::new([3,3],0.0);
    //let trace = matr_a.iter_diagonal();
    //matr_a.formated_output(6, "full");
    //let mut matr_b = _power(&matr_a, -0.5, 1.0e-10).unwrap();
    //matr_b.formated_output(6, "full");
    //_dgemm_full(&matr_b, 'N', &matr_b, 'N', &mut matr_c, 1.0, 0.0);
    //_dgemm_full(&matr_c, 'N', &matr_a, 'N', &mut matr_b, 1.0, 0.0);
    //matr_b.formated_output(6, "full");
    ////_dgemm_full(&matr_c, 'N', &matr_a, 'N', &mut matr_b, 1.0, 0.0);
    ////println!("inverse using _power:");
    ////matr_b.formated_output(6, "full");
    ////let mut matr_c = _dinverse(&matr_a).unwrap();
    ////matr_c.formated_output(6, "full");
    ////_dgemm_full(&matr_c, 'N', &matr_a, 'N', &mut matr_b, 1.0, 0.0);
    ////println!("inverse using _dinverse:");
    ////matr_b.formated_output(6, "full");



    //let (mut matr_b, n_sigular, converge_flag) = _newton_schulz_inverse_square_root(&matr_a, 1.0e-7, 0.6, 300);
    //_dgemm_full(&matr_b, 'N', &matr_b, 'N', &mut matr_c, 1.0, 0.0);
    //matr_c.formated_output(6, "full");
    //println!("n_singular: {}", n_sigular);
    //_dgemm_full(&matr_c, 'N', &matr_a, 'N', &mut matr_b, 1.0, 0.0);
    //matr_b.formated_output(6, "full");

    ////let matr_b = _newton_schulz_inverse_square_root(&matr_a, 1.0e-4, 100);

    ////_dgemm_full(&matr_b, 'N', &matr_b, 'N', &mut matr_c, 1.0, 0.0);
    ////matr_c.formated_output(6, "full");


    //////let mut matr_b = _power(&matr_c, 2.0, 1.0e-8).unwrap();
    ////let mut matr_b = MatrixFull::new([3,3],0.0);
    ////_dgemm_full(&matr_c, 'N', &matr_c, 'N', &mut matr_b, 1.0, 0.0);
    ////matr_b.formated_output(6, "full");
    // ===========================================================================================================
}

pub fn _newton_schulz_inverse_square_root_v02(original_matr: &MatrixFull<f64>, threshold: f64, mix_parameter: f64, num_iter: usize, level_shift: &Vec<f64>)  -> (MatrixFull<f64>, usize, bool)
{
    let num_elems = original_matr.data_ref().unwrap().len();
    let mut unit_matr = MatrixFull::new([original_matr.size()[0],original_matr.size()[1]],0.0);
    unit_matr.iter_diagonal_mut().unwrap().for_each(|x| *x = 3.0);

    let mut next_matr_0 = MatrixFull::new([original_matr.size()[0],original_matr.size()[1]],0.0);
    let mut next_matr_1 =  next_matr_0.clone();
    let mut prev_matr_0 =  next_matr_0.clone();

    // now prepare the initial guess 
    let mut initial_matr = MatrixFull::new([original_matr.size()[0],original_matr.size()[1]],0.0);
    let mut original_norm = original_matr.data_ref().unwrap().par_iter()
        .fold_with(0.0, |acc, x| { acc + x.powf(2.0)}).sum::<f64>();
    original_norm = original_norm.sqrt();


    //   1) renormalize the matrix with respect to its norm
    let matr_0 = MatrixFull::from_vec([original_matr.size()[0], original_matr.size()[1]], 
        original_matr.iter().map(|x| {*x/original_norm}).collect_vec()).unwrap();
    matr_0.formated_output(3, "full");
    let norm = matr_0.data_ref().unwrap().par_iter()
        .fold_with(0.0, |acc, x| { acc + x.powf(2.0)}).sum::<f64>().sqrt();
    println!("norm: {} before, {} after scale", &original_norm, &norm);

    let mut initial_guess = None;
    let mut converge_flag = false;
    //let level_shift = vec![1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6];
    for i in 0..level_shift.len() {
        let mut out_matr = matr_0.clone();
        let mut n_singular = 0;
        let mut lf_converge_flag = false;
        //let original_guess = initial_guess.clone();
        (out_matr, n_singular, lf_converge_flag) = _newton_schulz_inverse_square_root_lf(&matr_0, &initial_guess, threshold,mix_parameter, level_shift[i], num_iter);

        let mut fine_mix_parameter = mix_parameter;

        if !lf_converge_flag {
            let mut lf_converge_flag_2 = lf_converge_flag;
            let mut curr_step = 0;
            while ! lf_converge_flag_2 {
                fine_mix_parameter = fine_mix_parameter/2.0;
                curr_step += 1;
                //println!("fine_mix_parameter: {}", &fine_mix_parameter);
                (out_matr, n_singular, lf_converge_flag_2) = _newton_schulz_inverse_square_root_lf(&matr_0, &initial_guess, threshold,fine_mix_parameter, level_shift[i], num_iter);
                if curr_step > 5 {
                    //initial_guess = Some(out_matr);
                    //converge_flag = lf_converge_flag_2;
                    break
                }
            }
            initial_guess = Some(out_matr);
            converge_flag = lf_converge_flag_2;
            break
        } else {
            converge_flag = lf_converge_flag;
            initial_guess = Some(out_matr)
        }
    }

    next_matr_1 = initial_guess.unwrap();

    let n_singular = 0;

    let rescaled_norm = original_norm.powf(-0.5);
    next_matr_1.data.par_iter_mut().for_each(|x| *x *= rescaled_norm);

    (next_matr_1, n_singular, converge_flag)
}

pub fn _newton_schulz_inverse_square_root_lf(input_matr: &MatrixFull<f64>, initial_guess: &Option<MatrixFull<f64>>, threshold: f64, mix_parameter: f64, level_shift: f64, num_iter: usize)  -> (MatrixFull<f64>, usize, bool)
{

    println!("mix_parameter: {}", mix_parameter);
    let num_elems = input_matr.data_ref().unwrap().len();
    let mut unit_matr = MatrixFull::new([input_matr.size()[0],input_matr.size()[1]],0.0);
    unit_matr.iter_diagonal_mut().unwrap().for_each(|x| *x = 3.0);

    //  level-shift
    let mut matr = input_matr.clone();
    matr.iter_diagonal_mut().unwrap().for_each(|to| {
        if *to > threshold {
            *to += level_shift
        }  else {
            *to = 0.0
        }
    });

    let mut trace_1 = matr.iter_diagonal().unwrap()
        .fold(0.0, |acc, x| {acc + x })/input_matr.size()[0] as f64;
    let initial_a = 1.0/trace_1.sqrt();
    //println!("trace: {:16.8},initial_a: {:16.8}", &trace_1, &initial_a);

    let mut next_matr_0 = MatrixFull::new([input_matr.size()[0],input_matr.size()[1]],0.0);
    let mut next_matr_1 =  next_matr_0.clone();
    let mut prev_matr_0 =  next_matr_0.clone();

    // now prepare the initial guess 
    let mut initial_matr = if let Some(tmp_matr) = initial_guess {
        tmp_matr.clone()
    } else {
        let mut initial_matr = MatrixFull::new([input_matr.size()[0],input_matr.size()[1]],0.0);
        initial_matr.iter_diagonal_mut().unwrap().zip(matr.iter_diagonal().unwrap()).for_each(|(to, from) | {
            if (*from).abs() < threshold {
                *to = 0.0;
            } else {
                *to = initial_a
            }
        });
        initial_matr
    };

    // X_next = 0.5* X_curr * (3I - A*X_curr*X_curr)
    _dgemm_full(&matr, 'N', &initial_matr, 'N', &mut next_matr_0, 1.0, 0.0);
    _dgemm_full(&next_matr_0, 'N', &initial_matr, 'N', &mut next_matr_1, 1.0, 0.0);

    next_matr_0 = unit_matr.par_scaled_add(&next_matr_1, -1.0).unwrap();

    _dgemm_full(&initial_matr, 'N', &next_matr_0, 'N', &mut next_matr_1, 0.5, 0.0);

    let mut norm = initial_matr.data_ref().unwrap().par_iter().zip(next_matr_1.data_ref().unwrap().par_iter())
        .fold_with(0.0, |acc, (x,y)| { acc + (x-y).powf(2.0)}).sum::<f64>().sqrt()/num_elems as f64;
    

    let mut prev_norm = norm;
    prev_matr_0 = initial_matr.clone();
    initial_matr = next_matr_1.clone();


    let mut converge_flag = norm<=threshold;
    let mut final_iter = 0_usize;

    let mut count_unchanged_results = 0;
    for cur_iter in 0..num_iter {

        // X_next = 0.5* X_curr * (3I - A*X_curr*X_curr)

        _dgemm_full(&matr, 'N', &initial_matr, 'N', &mut next_matr_0, 1.0, 0.0);
        _dgemm_full(&next_matr_0, 'N', &initial_matr, 'N', &mut next_matr_1, 1.0, 0.0);
        next_matr_0 = unit_matr.par_scaled_add(&next_matr_1, -1.0).unwrap();
        _dgemm_full(&initial_matr, 'N', &next_matr_0, 'N', &mut next_matr_1, 0.5, 0.0);

        // compute the difference between X_next and X_curr: ||X_next - X_curr||
        let mut norm = initial_matr.data_ref().unwrap().par_iter().zip(next_matr_1.data_ref().unwrap().par_iter())
            .fold_with(0.0, |acc, (x,y)| { acc + (x-y).powf(2.0)}).sum::<f64>().sqrt()/num_elems as f64;

        let self_norm = next_matr_1.data_ref().unwrap().par_iter()
            .fold_with(0.0, |acc, x| { acc + x.powf(2.0)}).sum::<f64>().sqrt();


        converge_flag = norm <= threshold;

        let change_norm = norm - prev_norm;
        prev_norm = norm;

        let mut dev_symm_norm = 0.0;

        prev_matr_0 = initial_matr.clone();
        

        if converge_flag {
            println!("Iteration after {} steps: 1) diff_norm: {:16.8}; 2) self_norm: {:16.8}; 3) broken symmetry: {:16.8}", &cur_iter, &norm, &self_norm/num_elems as f64, &dev_symm_norm);
            break
        } else {
            if norm <= threshold * 10.0 {
                initial_matr.data_ref_mut().unwrap().par_iter_mut().zip(next_matr_1.data_ref().unwrap().par_iter()).for_each(|(to, from)| {
                    *to = mix_parameter*(*from) + (1.0-mix_parameter)*(*to);
                });
                initial_matr.data_ref_mut().unwrap().par_iter_mut().zip(prev_matr_0.data_ref().unwrap().par_iter()).for_each(|(to, from)| {
                    *to = mix_parameter*(*from) + (1.0-mix_parameter)*(*to);
                });
                count_unchanged_results = 0;

            } else if change_norm > threshold {
                count_unchanged_results = 0;
                initial_matr.data_ref_mut().unwrap().par_iter_mut().zip(next_matr_1.data_ref().unwrap().par_iter()).for_each(|(to, from)| {
                    *to = mix_parameter*(*from) + (1.0-mix_parameter)*(*to);
                });
                //initial_matr  = (initial_matr.transpose() + initial_matr)*0.5;
                
                let tmp_self_norm = initial_matr.data_ref().unwrap().par_iter()
                    .fold_with(0.0, |acc, x| { acc + x.powf(2.0)}).sum::<f64>().sqrt();
                if tmp_self_norm >= 100.0 {

                    initial_matr.data_ref_mut().unwrap().par_iter_mut().for_each(|x| *x /= tmp_self_norm);
                    //let wr = initial_matr.iter_diagonal().unwrap().map(|x| *x).collect_vec();
                    initial_matr.data_ref_mut().unwrap().par_iter_mut().for_each(|x| *x /= tmp_self_norm);
                    initial_matr.iter_diagonal_mut().unwrap().for_each(|x| *x *= tmp_self_norm);
                }
            }  else if change_norm.abs() <= threshold {
                if count_unchanged_results < 5 {
                    initial_matr.data_ref_mut().unwrap().par_iter_mut().zip(next_matr_1.data_ref().unwrap().par_iter()).for_each(|(to, from)| {
                        *to = mix_parameter*(*from) + (1.0-mix_parameter)*(*to);
                    });
                    count_unchanged_results += 1;
                } else {
                    initial_matr.iter_diagonal_mut().unwrap().for_each(|to | *to = initial_a);
                    count_unchanged_results = 0;
                }
            } else {
                initial_matr = next_matr_1.clone();
                count_unchanged_results = 0;
            }
        };

        println!("Iteration after {} steps: 1) diff_norm: {:16.8}; 2) self_norm: {:16.8}", &cur_iter, &norm, &self_norm/num_elems as f64);

        final_iter += 1;

        //initial_matr.formated_output(3, "full");


    }
    if converge_flag {
        println!("Iteration is converged after {} steps", final_iter);
    } else {
        println!("WARNNING: Iteration IS NOT converged after {} steps", final_iter);
    }

    let n_singular = 0;

    (next_matr_1, n_singular, converge_flag)
}


//fn _newton_schulz_inverse_square_root<'a, T>(matr: &MatrixFull<f64>, threshold: f64, num_iter: usize)  -> MatrixFull<f64>
//where T: BasicMatrix<'a, f64> 
pub fn _newton_schulz_inverse_square_root_v01(original_matr: &MatrixFull<f64>, threshold: f64, mix_parameter: f64, num_iter: usize)  -> (MatrixFull<f64>, usize, bool)
{

    let num_elems = original_matr.data_ref().unwrap().len();
    let mut unit_matr = MatrixFull::new([original_matr.size()[0],original_matr.size()[1]],0.0);
    unit_matr.iter_diagonal_mut().unwrap().for_each(|x| *x = 3.0);

    let mut next_matr_0 = MatrixFull::new([original_matr.size()[0],original_matr.size()[1]],0.0);
    let mut next_matr_1 =  next_matr_0.clone();
    let mut prev_matr_0 =  next_matr_0.clone();

    // now prepare the initial guess 
    let mut initial_matr = MatrixFull::new([original_matr.size()[0],original_matr.size()[1]],0.0);
    let mut original_norm = original_matr.data_ref().unwrap().par_iter()
        .fold_with(0.0, |acc, x| { acc + x.powf(2.0)}).sum::<f64>();
    original_norm = original_norm.sqrt();


    //   1) renomalize the matrix with respect to its norm
    let matr_0 = MatrixFull::from_vec([original_matr.size()[0], original_matr.size()[1]], 
        original_matr.iter().map(|x| {*x/original_norm}).collect_vec()).unwrap();
    matr_0.formated_output(3, "full");
    let norm = matr_0.data_ref().unwrap().par_iter()
        .fold_with(0.0, |acc, x| { acc + x.powf(2.0)}).sum::<f64>().sqrt();
    println!("norm: {} before, {} after scale", &original_norm, &norm);

    // processing
    //let mut p_matr = MatrixFull::new([original_matr.size()[0], original_matr.size()[1]], 0.0);
    //p_matr.iter_diagonal_mut().unwrap().zip(matr_0.iter_diagonal().unwrap()).for_each(|(to,from)| {
    //    if *from < 1.0e-8 {
    //        *to = 1.0;
    //    } else {
    //        *to = 1.0/(*from)
    //    }
    //});
    //p_matr.formated_output(3, "full");
    //let mut matr = MatrixFull::new([original_matr.size()[0], original_matr.size()[1]], 0.0);
    //_dgemm_full(&p_matr, 'N', &matr_0, 'N', &mut next_matr_0, 1.0, 0.0);
    //_dgemm_full(&next_matr_0, 'N', &p_matr, 'N', &mut matr, 1.0, 0.0);

    //  level-shift
    let mut matr = matr_0.clone();
    matr.iter_diagonal_mut().unwrap().for_each(|to| *to += 1.0e-3);
    

    let mut trace_1 = matr.iter_diagonal().unwrap()
        .fold(0.0, |acc, x| {acc + x })/matr.size()[0] as f64;
    let initial_a = 1.0/trace_1.sqrt();
    println!("trace: {:16.8},initial_a: {:16.8}", &trace_1, &initial_a);
    initial_matr.iter_diagonal_mut().unwrap().for_each(|to | *to = initial_a);
    let n_singular = 0;

    //let (mut matr_b, vs, wr, wi) = _dgees(&matr, 'N', 'N', None);
    //let mut n_singular = 0_usize;
    //wr.iter().for_each(|x| {if *x<=1.0e-8 {n_singular += 1}});
    //initial_matr.iter_diagonal_mut().unwrap().zip(wr.iter()).for_each(|(to, from)| {
    //    *to = if *from<=0.0 {
    //        initial_a
    //    } else if *from<=1.0e-3 {
    //        initial_a
    //    } else {
    //        (*from).powf(-0.5)
    //    }
    //});

    //initial_matr.formated_output(3, "full");

    // X_next = 0.5* X_curr * (3I - A*X_curr*X_curr)
    // _dgemm_full(&matr, 'N', &initial_matr, 'N', &mut next_matr_0, 1.0, 0.0);
    // _dgemm_full(&next_matr_0, 'N', &initial_matr, 'N', &mut next_matr_1, 1.0, 0.0);

    // next_matr_0 = unit_matr.par_scaled_add(&next_matr_1, -1.0).unwrap();

    // _dgemm_full(&initial_matr, 'N', &next_matr_0, 'N', &mut next_matr_1, 0.5, 0.0);

    // let mut norm = initial_matr.data_ref().unwrap().par_iter().zip(next_matr_1.data_ref().unwrap().par_iter())
    //     .fold_with(0.0, |acc, (x,y)| { acc + (x-y).powf(2.0)}).sum::<f64>().sqrt()/num_elems as f64;

    

    let mut prev_norm = norm;
    //prev_matr_0 = initial_matr.clone();
    //initial_matr = next_matr_1.clone();


    let mut converge_flag = norm<=threshold;
    let mut final_iter = 0_usize;
    for cur_iter in 0..num_iter {

        // X_next = 0.5* X_curr * (3I - A*X_curr*X_curr)

        _dgemm_full(&matr, 'N', &initial_matr, 'N', &mut next_matr_0, 1.0, 0.0);
        _dgemm_full(&next_matr_0, 'N', &initial_matr, 'N', &mut next_matr_1, 1.0, 0.0);
        next_matr_0 = unit_matr.par_scaled_add(&next_matr_1, -1.0).unwrap();
        _dgemm_full(&initial_matr, 'N', &next_matr_0, 'N', &mut next_matr_1, 0.5, 0.0);

        // compute the difference between X_next and X_curr: ||X_next - X_curr||
        let mut norm = initial_matr.data_ref().unwrap().par_iter().zip(next_matr_1.data_ref().unwrap().par_iter())
            .fold_with(0.0, |acc, (x,y)| { acc + (x-y).powf(2.0)}).sum::<f64>().sqrt()/num_elems as f64;

        let self_norm = next_matr_1.data_ref().unwrap().par_iter()
            .fold_with(0.0, |acc, x| { acc + x.powf(2.0)}).sum::<f64>().sqrt();


        converge_flag = norm <= threshold;

        let change_norm = norm - prev_norm;
        prev_norm = norm;

        let mut dev_symm_norm = 0.0;

        prev_matr_0 = initial_matr.clone();
        

        if converge_flag {
            println!("Iteration after {} steps: 1) diff_norm: {:16.8}; 2) self_norm: {:16.8}; 3) broken symmetry: {:16.8}", &cur_iter, &norm, &self_norm/num_elems as f64, &dev_symm_norm);
            break
        } else {
            if norm <= threshold * 10.0 {
                initial_matr.data_ref_mut().unwrap().par_iter_mut().zip(next_matr_1.data_ref().unwrap().par_iter()).for_each(|(to, from)| {
                    *to = mix_parameter*(*from) + (1.0-mix_parameter)*(*to);
                });
                initial_matr.data_ref_mut().unwrap().par_iter_mut().zip(prev_matr_0.data_ref().unwrap().par_iter()).for_each(|(to, from)| {
                    *to = mix_parameter*(*from) + (1.0-mix_parameter)*(*to);
                });

            } else if change_norm > 1.0e-3 {
                initial_matr.data_ref_mut().unwrap().par_iter_mut().zip(next_matr_1.data_ref().unwrap().par_iter()).for_each(|(to, from)| {
                    *to = mix_parameter*(*from) + (1.0-mix_parameter)*(*to);
                });
                //initial_matr  = (initial_matr.transpose() + initial_matr)*0.5;
                
                let tmp_self_norm = initial_matr.data_ref().unwrap().par_iter()
                    .fold_with(0.0, |acc, x| { acc + x.powf(2.0)}).sum::<f64>().sqrt();
                if tmp_self_norm >= 100.0 {

                    initial_matr.data_ref_mut().unwrap().par_iter_mut().for_each(|x| *x /= tmp_self_norm);
                    //let wr = initial_matr.iter_diagonal().unwrap().map(|x| *x).collect_vec();
                    initial_matr.data_ref_mut().unwrap().par_iter_mut().for_each(|x| *x /= tmp_self_norm);
                    initial_matr.iter_diagonal_mut().unwrap().for_each(|x| *x *= tmp_self_norm);
                }
            }  else if change_norm.abs() < 1.0e-3 {
                //initial_matr = next_matr_1.clone();
                //initial_matr = (initial_matr.transpose() + initial_matr)*0.5;
                //dev_symm_norm = initial_matr.data_ref().unwrap().par_iter().zip(next_matr_1.data_ref().unwrap().par_iter())
                //    .fold_with(0.0, |acc, (x,y)| { acc + (x-y).powf(2.0)}).sum::<f64>();
                initial_matr.iter_diagonal_mut().unwrap().for_each(|to | *to = initial_a);

            }
            else {
                initial_matr = next_matr_1.clone();
                //initial_matr = (initial_matr.transpose() + initial_matr)*0.5;
                dev_symm_norm = initial_matr.data_ref().unwrap().par_iter().zip(next_matr_1.data_ref().unwrap().par_iter())
                    .fold_with(0.0, |acc, (x,y)| { acc + (x-y).powf(2.0)}).sum::<f64>();

            }
        };

        println!("Iteration after {} steps: 1) diff_norm: {:16.8}; 2) self_norm: {:16.8}; 3) broken symmetry: {:16.8}", &cur_iter, &norm, &self_norm/num_elems as f64, &dev_symm_norm);

        final_iter += 1;

        //initial_matr.formated_output(3, "full");


    }
    if converge_flag {
        println!("Iteration is converged after {} steps", final_iter);
    } else {
        println!("WARNNING: Iteration IS NOT converged after {} steps", final_iter);
    }

    //p_matr.iter_diagonal_mut().unwrap().for_each(|to| {
    //    *to = (*to).sqrt()
    //});
    //_dgemm_full(&p_matr, 'N', &next_matr_1, 'N', &mut next_matr_0, 1.0, 0.0);
    //_dgemm_full(&next_matr_0, 'N', &p_matr, 'N', &mut next_matr_1, 1.0, 0.0);


    let rescaled_norm = original_norm.powf(-0.5);
    next_matr_1.data.par_iter_mut().for_each(|x| *x *= rescaled_norm);

    (next_matr_1, n_singular, converge_flag)
}

//pub fn _dspgvx(&mut self,ovlp:MatrixUpperSliceMut<f64>,num_orb:usize) -> Option<(MatrixFull<f64>,Vec<f64>)> {
pub fn _dspgvx<'a, T, Q>(matr_a: &T, matr_b: &Q, num_orb: usize)  -> Option<(MatrixFull<f64>,Vec<f64>, usize)> 
where T: BasicMatUp<'a, f64>,
      Q: BasicMatUp<'a, f64>, 
      //P: BasicMatrix<'a, f64>
{
    ///solve A*x=(lambda)*B*x
    /// A is "self"; B is ovlp
    let mut itype: i32 = 1;
    //let n = ((8.0*self.size.to_owned() as f32+1.0).sqrt()/2.0) as i32;
    let a_size = matr_a.size();
    let b_size = matr_b.size();
    if ! basic_check_shape(&a_size, &b_size) {
        panic!("ERROR:: _dspgvx for BasicMatUp, Matr_A[{:},{:}] and Matr_B[{:},{:}] have different size",
         a_size[0], a_size[1], b_size[0], b_size[1]
        )
    };
    let ndim = a_size[0];
    let n = ndim as i32;
    let mut a = matr_a.data_ref().unwrap().to_vec().clone();
    let mut b = matr_b.data_ref().unwrap().to_vec().clone();
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
    Some((eigenvectors, w, m as usize))
}

/// # a fast solver for the hamiltonian equation:
///        0) -> H \cdot C =  S \cdot C \cdot L 
/// where H is the fock matrix, C the MO coefficient matrix, S the overlap matrix, and L the eigenvalues for this equation
/// The original implementation in REST is to simply invoke the lapack subroutine '_dspgvx'. 
/// Unfortuantely, this subroutine is not well optimized for multi-thread parallization.
/// Therefore, we try another algorithm to solve this problem as follow:
///  1) Define a transformation matrix X = S^{0.5}, such that the original equation can be changed to:
///        a) -> H \cdot C = X \cdot X \cdot C \cdot \lambda
///  2) Then, define a new matrix C' = X \cdot C:
///        b) -> H \cdot C = X \cdot C' \cdot \lambda
///        c) -> H \cdot X^{-1} \cdot C' = X \cdot C' \cdot \lambda
///  3) Then, define a new matrix H' = X^{-1} \cdot H X^{-1}, the original equation changes to:
///        d) -> H' \cdot C' =  C' \cdot \lambda
///  4) Since H' is also a symetric matrix, this standard diagonalization problem could be solved by using the efficient lapack subroutin '_dsyevd'.
///     It yields the eigenvalue L, and the eigenvector matrix C'
///  5) The MO coefficient matrix C can then be obtrain by using the following matrix transformation:
///        e) -> C = X^{-1} \cdot C'
///       
/// 
pub fn _hamiltonian_fast_solver<'a, T, Q>(matr_a: &T, matr_b: &Q, num_orb: &mut usize)  -> Option<(MatrixFull<f64>,Vec<f64>)> 
//                                                         IGOR:: should turn on the use of num_orb for ill-conditioning check and manipulation
where T: BasicMatUp<'a, f64>,
      Q: BasicMatUp<'a, f64>, 
      //P: BasicMatrix<'a, f64>
{
    // first change matr_a and matr_b with matrixup to matrixfull 

    let num_bas = matr_a.size()[0];
    let mut matr_a_full = MatrixFull::new(matr_a.size(), 0.0);
    matr_a_full.iter_matrixupper_mut().unwrap().zip(matr_a.data_ref().unwrap()).for_each(|(to, from)| {
        *to = *from;
    });
    matr_a_full.fill_lower_part_from_upper();
    let mut matr_b_full = MatrixFull::new(matr_b.size(), 0.0);
    matr_b_full.iter_matrixupper_mut().unwrap().zip(matr_b.data_ref().unwrap()).for_each(|(to, from)| {
        *to = *from;
    });
    matr_b_full.fill_lower_part_from_upper();

    let (matr_b_sqrt, matr_b_sqrt_inv, n_singularity) = _get_sqrt_and_inv_sqrt(&matr_b_full, 1.0e-10).unwrap();

    if n_singularity < *num_orb {*num_orb = n_singularity};

    let mut matr_a_full_p= MatrixFull::new(matr_a.size(), 0.0);
    let mut matr_a_tmp= MatrixFull::new(matr_a.size(), 0.0);

    _dgemm_full(&matr_a_full, 'N', &matr_b_sqrt_inv, 'N', &mut matr_a_tmp, 1.0, 0.0);
    _dgemm_full(&matr_b_sqrt_inv, 'N', &matr_a_tmp, 'N', &mut matr_a_full_p, 1.0, 0.0);

    let (mut opt_eigenvector, mut eigenvalues, mut n) = _dsyevd(&matr_a_full_p, 'V');
    matr_a_tmp = opt_eigenvector.unwrap();

    let mut eigenvector= MatrixFull::new([matr_a.size()[0], *num_orb], 0.0);

    _dgemm(&matr_b_sqrt_inv, (0..num_bas, 0..*num_orb), 'T', 
           &matr_a_tmp, (0..num_bas, 0..*num_orb), 'N', 
           &mut eigenvector, (0..num_bas, 0..*num_orb),
           1.0, 0.0);

    Some((eigenvector, eigenvalues))
}

pub fn _get_sqrt_and_inv_sqrt<'a, T>(matr_a: &T, threshold: f64) -> Option<(MatrixFull<f64>, MatrixFull<f64>, usize)> 
where T: BasicMatrix<'a, f64>
{
    let size = matr_a.size();
    if size[0]==size[1] {

        let mut sqrt_om = MatrixFull::new([size[0],size[1]],0.0);
        let mut inv_sqrt_om = MatrixFull::new([size[0],size[1]],0.0);

        // because lapack sorts eigenvalues from small to large, we multiply the elements with -1.0
        sqrt_om.data.par_iter_mut().zip(matr_a.data_ref().unwrap().par_iter()).for_each(|value| {*value.0=*value.1*-1.0f64});
        //inv_sqrt_om.data.par_iter_mut().zip(matr_a.data_ref().unwrap().par_iter()).for_each(|value| {*value.0=*value.1*-1.0f64});


        // diagonalize the matrix
        let (mut opt_eigenvector, mut eigenvalues, mut n) = _dsyevd(&sqrt_om, 'V');
        let mut sqrt_eigenvector = opt_eigenvector.unwrap();
        let mut inv_sqrt_eigenvector = sqrt_eigenvector.clone();

        // now we get the eigenvectors with the eigenvalues from large to small
        let (mut n_nonsigular, mut tmpv) = (0usize,0.0);
        eigenvalues.iter_mut().for_each(|value| {
            *value = *value*(-1.0);
            //if *value >= threshold && *value <= tmpv {
            if *value >= threshold {
                n_nonsigular +=1;
            };
        });

        if n as usize != size[0] {
            panic!("Found unphysical eigenvalues");
        } else if n_nonsigular != size[0] {
            //println!("_power_rayon => n_nonsigular: {}, {}", n_nonsigular, size[0]);
            //println!("{:?}", &eigenvalues);
        }

        &sqrt_eigenvector.data.par_chunks_exact_mut(size[0]).enumerate()
            .for_each(|(i,value)| {
            if i<n_nonsigular {
                if let Some(ev) = eigenvalues.get(i) {
                    let ev_sqrt = ev.sqrt().sqrt();
                    value.iter_mut().for_each(|v| {*v = *v*ev_sqrt});
                }
            } else {
                value.iter_mut().for_each(|v| {*v = 0.0f64});
            }
        });

        &inv_sqrt_eigenvector.data.par_chunks_exact_mut(size[0]).enumerate()
            .for_each(|(i,value)| {
            if i<n_nonsigular {
                if let Some(ev) = eigenvalues.get(i) {
                    let ev_sqrt = ev.sqrt().sqrt();
                    value.iter_mut().for_each(|v| {*v = *v/ev_sqrt});
                }
            } else {
                value.iter_mut().for_each(|v| {*v = 0.0f64});
            }
        });

        
        _dgemm_full(&sqrt_eigenvector, 'N', &sqrt_eigenvector, 'T', &mut sqrt_om, 1.0, 0.0);
        _dgemm_full(&inv_sqrt_eigenvector, 'N', &inv_sqrt_eigenvector, 'T', &mut inv_sqrt_om, 1.0, 0.0);

        Some((sqrt_om, inv_sqrt_om, n_nonsigular))
    } else {
        println!("Error: The matrix for power operations should be NxN");
        None
    }

}

#[cfg(test)]
mod tests {
    use crate::matrix::MatrixFull;
    use super::*;

    #[test]
    fn test_dsvd_square() {
        let matr_a = MatrixFull::from_vec([3, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();

        let (u, s, vt) = _dsvd(&matr_a, 'A', 'A');

        assert_eq!(u.size(), [3, 3]);
        assert_eq!(s.len(), 3);
        assert_eq!(vt.size(), [3, 3]);

        // singular values should be sorted descending
        assert!(s[0] >= s[1] && s[1] >= s[2]);
        assert!(s[0] > 0.0);

        // reconstruct A from U * S * V^T
        let mut us = MatrixFull::new([3, 3], 0.0);
        for j in 0..3 {
            let col: Vec<f64> = u.iter_column(j).map(|x| *x * s[j]).collect();
            for i in 0..3 {
                us[[i, j]] = col[i];
            }
        }
        let mut reconstructed = MatrixFull::new([3, 3], 0.0);
        _dgemm_full(&us, 'N', &vt, 'N', &mut reconstructed, 1.0, 0.0);

        let diff = matr_a.data_ref().unwrap().iter()
            .zip(reconstructed.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff < 1e-10, "Reconstruction error: {}", diff);
    }

    #[test]
    fn test_dsvd_rectangular_m_gt_n() {
        let matr_a = MatrixFull::from_vec([4, 2], vec![
            0.0, 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0, 7.0,
        ]).unwrap();

        let (u, s, vt) = _dsvd(&matr_a, 'A', 'A');
        assert_eq!(u.size(), [4, 4]);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.size(), [2, 2]);

        let mut us = MatrixFull::new([4, 2], 0.0);
        for j in 0..2 {
            let col: Vec<f64> = u.iter_column(j).map(|x| *x * s[j]).collect();
            for i in 0..4 {
                us[[i, j]] = col[i];
            }
        }
        let mut reconstructed = MatrixFull::new([4, 2], 0.0);
        _dgemm_full(&us, 'N', &vt, 'N', &mut reconstructed, 1.0, 0.0);

        let diff = matr_a.data_ref().unwrap().iter()
            .zip(reconstructed.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff < 1e-10, "Reconstruction error: {}", diff);
    }

    #[test]
    fn test_dsvd_rectangular_m_lt_n() {
        let matr_a = MatrixFull::from_vec([2, 4], vec![
            0.0, 1.0,
            2.0, 3.0,
            4.0, 5.0,
            6.0, 7.0,
        ]).unwrap();

        let (u, s, vt) = _dsvd(&matr_a, 'S', 'S');
        assert_eq!(u.size(), [2, 2]);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.size(), [2, 4]);

        let mut us = MatrixFull::new([2, 2], 0.0);
        for j in 0..2 {
            let col: Vec<f64> = u.iter_column(j).map(|x| *x * s[j]).collect();
            for i in 0..2 {
                us[[i, j]] = col[i];
            }
        }
        let mut reconstructed = MatrixFull::new([2, 4], 0.0);
        _dgemm_full(&us, 'N', &vt, 'N', &mut reconstructed, 1.0, 0.0);

        let diff = matr_a.data_ref().unwrap().iter()
            .zip(reconstructed.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff < 1e-10, "Reconstruction error: {}", diff);
    }

    #[test]
    fn test_dsvd_jobu_s_jobvt_s() {
        let matr_a = MatrixFull::from_vec([4, 3], vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ]).unwrap();

        let (u, s, vt) = _dsvd(&matr_a, 'S', 'S');
        assert_eq!(u.size(), [4, 3]);
        assert_eq!(s.len(), 3);
        assert_eq!(vt.size(), [3, 3]);

        let mut us = MatrixFull::new([4, 3], 0.0);
        for j in 0..3 {
            let col: Vec<f64> = u.iter_column(j).map(|x| *x * s[j]).collect();
            for i in 0..4 {
                us[[i, j]] = col[i];
            }
        }
        let mut reconstructed = MatrixFull::new([4, 3], 0.0);
        _dgemm_full(&us, 'N', &vt, 'N', &mut reconstructed, 1.0, 0.0);

        let diff = matr_a.data_ref().unwrap().iter()
            .zip(reconstructed.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff < 1e-10, "Reconstruction error: {}", diff);
    }

    #[test]
    fn test_pinv_full_rank_square() {
        let matr_a = MatrixFull::from_vec([3, 3], vec![
            1.0, 0.0, 2.0,
            0.0, 3.0, 0.0,
            2.0, 0.0, 1.0,
        ]).unwrap();

        let a_pinv = _pinv(&matr_a, None).unwrap();
        assert_eq!(a_pinv.size(), [3, 3]);

        let mut identity_approx = MatrixFull::new([3, 3], 0.0);
        _dgemm_full(&a_pinv, 'N', &matr_a, 'N', &mut identity_approx, 1.0, 0.0);

        let diff = identity_approx.data_ref().unwrap().iter()
            .zip(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0].iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff < 1e-10, "A^+ * A != I, diff: {}", diff);
    }

    #[test]
    fn test_pinv_rectangular_m_gt_n_full_col_rank() {
        let matr_a = MatrixFull::from_vec([3, 2], vec![
            1.0, 3.0, 5.0,
            2.0, 4.0, 6.0,
        ]).unwrap();

        let a_pinv = _pinv(&matr_a, None).unwrap();
        assert_eq!(a_pinv.size(), [2, 3]);

        // A^+ * A should equal I_2
        let mut identity_approx = MatrixFull::new([2, 2], 0.0);
        _dgemm_full(&a_pinv, 'N', &matr_a, 'N', &mut identity_approx, 1.0, 0.0);

        let id_data = vec![1.0, 0.0, 0.0, 1.0];
        let diff = identity_approx.data_ref().unwrap().iter()
            .zip(id_data.iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff < 1e-10, "A^+ * A != I for m>n, diff: {}", diff);
    }

    #[test]
    fn test_pinv_rectangular_m_lt_n_full_row_rank() {
        let matr_a = MatrixFull::from_vec([2, 3], vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]).unwrap();

        let a_pinv = _pinv(&matr_a, None).unwrap();
        assert_eq!(a_pinv.size(), [3, 2]);

        // A * A^+ should equal I_2
        let mut identity_approx = MatrixFull::new([2, 2], 0.0);
        _dgemm_full(&matr_a, 'N', &a_pinv, 'N', &mut identity_approx, 1.0, 0.0);

        let id_data = vec![1.0, 0.0, 0.0, 1.0];
        let diff = identity_approx.data_ref().unwrap().iter()
            .zip(id_data.iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff < 1e-10, "A * A^+ != I for m<n, diff: {}", diff);
    }

    #[test]
    fn test_pinv_rank_deficient() {
        // rank-1 matrix: outer product of [1,2,3] and [1,1,1]
        let matr_a = MatrixFull::from_vec([3, 3], vec![
            1.0, 2.0, 3.0,
            1.0, 2.0, 3.0,
            1.0, 2.0, 3.0,
        ]).unwrap();

        let a_pinv = _pinv(&matr_a, None).unwrap();
        assert_eq!(a_pinv.size(), [3, 3]);

        // Moore-Penrose condition: A * A^+ * A = A
        let mut tmp = MatrixFull::new([3, 3], 0.0);
        let mut reconstructed = MatrixFull::new([3, 3], 0.0);
        _dgemm_full(&matr_a, 'N', &a_pinv, 'N', &mut tmp, 1.0, 0.0);
        _dgemm_full(&tmp, 'N', &matr_a, 'N', &mut reconstructed, 1.0, 0.0);

        let diff = matr_a.data_ref().unwrap().iter()
            .zip(reconstructed.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff < 1e-10, "A * A^+ * A != A, diff: {}", diff);
    }

    #[test]
    fn test_pinv_moore_penrose_conditions() {
        // Full-rank 2x3 matrix: A = | 1 3 5 |
        //                            | 2 4 6 |
        let matr_a = MatrixFull::from_vec([2, 3], vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]).unwrap();

        let a_pinv = _pinv(&matr_a, None).unwrap();

        // 1. A * A^+ * A = A
        let mut tmp = MatrixFull::new([2, 2], 0.0);
        let mut a_aplus_a = MatrixFull::new([2, 3], 0.0);
        _dgemm_full(&matr_a, 'N', &a_pinv, 'N', &mut tmp, 1.0, 0.0);
        _dgemm_full(&tmp, 'N', &matr_a, 'N', &mut a_aplus_a, 1.0, 0.0);
        let diff1 = matr_a.data_ref().unwrap().iter()
            .zip(a_aplus_a.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff1 < 1e-10, "Condition 1 failed: A*A^+*A != A, diff: {}", diff1);

        // 2. A^+ * A * A^+ = A^+
        let mut tmp2 = MatrixFull::new([3, 3], 0.0);
        let mut aplus_a_aplus = MatrixFull::new([3, 2], 0.0);
        _dgemm_full(&a_pinv, 'N', &matr_a, 'N', &mut tmp2, 1.0, 0.0);
        _dgemm_full(&tmp2, 'N', &a_pinv, 'N', &mut aplus_a_aplus, 1.0, 0.0);
        let diff2 = a_pinv.data_ref().unwrap().iter()
            .zip(aplus_a_aplus.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff2 < 1e-10, "Condition 2 failed: A^+*A*A^+ != A^+, diff: {}", diff2);

        // 3. (A * A^+)^T = A * A^+
        let mut tmp3 = MatrixFull::new([2, 2], 0.0);
        _dgemm_full(&matr_a, 'N', &a_pinv, 'N', &mut tmp3, 1.0, 0.0);
        let transposed = tmp3.transpose();
        let diff3 = tmp3.data_ref().unwrap().iter()
            .zip(transposed.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff3 < 1e-10, "Condition 3 failed: A*A^+ not symmetric, diff: {}", diff3);

        // 4. (A^+ * A)^T = A^+ * A
        let mut tmp4 = MatrixFull::new([3, 3], 0.0);
        _dgemm_full(&a_pinv, 'N', &matr_a, 'N', &mut tmp4, 1.0, 0.0);
        let transposed4 = tmp4.transpose();
        let diff4 = tmp4.data_ref().unwrap().iter()
            .zip(transposed4.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff4 < 1e-10, "Condition 4 failed: A^+*A not symmetric, diff: {}", diff4);
    }

    #[test]
    fn test_pinv_custom_rcond() {
        let matr_a = MatrixFull::from_vec([3, 3], vec![
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 0.0,
        ]).unwrap();

        // with rcond=1e-3, only the first 2 singular values (1.0, 2.0) are kept
        let a_pinv = _pinv(&matr_a, Some(1e-3)).unwrap();

        let mut identity_approx = MatrixFull::new([3, 3], 0.0);
        _dgemm_full(&a_pinv, 'N', &matr_a, 'N', &mut identity_approx, 1.0, 0.0);

        // diagonal should be [1, 1, 0] after pinv * A
        let diff = (identity_approx[[0, 0]] - 1.0).abs()
                 + (identity_approx[[1, 1]] - 1.0).abs()
                 + identity_approx[[2, 2]].abs();
        assert!(diff < 1e-10, "Custom rcond pinv failed, diff: {}", diff);
    }

/*    #[test]
    fn test_qrpinv_full_rank_square() {
        let matr_a = MatrixFull::from_vec([3, 3], vec![
            2.0, 1.0, 1.0,
            1.0, 2.0, 1.0,
            1.0, 1.0, 2.0,
        ]).unwrap();

        let a_pinv = _qrpinv(&matr_a, None).unwrap();
        assert_eq!(a_pinv.size(), [3, 3]);

        let mut identity_approx = MatrixFull::new([3, 3], 0.0);
        _dgemm_full(&a_pinv, 'N', &matr_a, 'N', &mut identity_approx, 1.0, 0.0);

        let diff = identity_approx.data_ref().unwrap().iter()
            .zip(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0].iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff < 1e-10, "A^+ * A != I, diff: {}", diff);
    }

    #[test]
    fn test_qrpinv_rank_deficient() {
        let matr_a = MatrixFull::from_vec([3, 3], vec![
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0,
        ]).unwrap();

        let a_pinv = _qrpinv(&matr_a, Some(1e-6)).unwrap();
        assert_eq!(a_pinv.size(), [3, 3]);

        let mut tmp = MatrixFull::new([3, 3], 0.0);
        let mut reconstructed = MatrixFull::new([3, 3], 0.0);
        _dgemm_full(&matr_a, 'N', &a_pinv, 'N', &mut tmp, 1.0, 0.0);
        _dgemm_full(&tmp, 'N', &matr_a, 'N', &mut reconstructed, 1.0, 0.0);

        let diff = matr_a.data_ref().unwrap().iter()
            .zip(reconstructed.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff < 1e-10, "A * A^+ * A != A, diff: {}", diff);
    }

    #[test]
    fn test_qrpinv_moore_penrose_conditions() {
        let matr_a = MatrixFull::from_vec([4, 4], vec![
            4.0, 1.0, 2.0, 3.0,
            1.0, 5.0, 1.0, 2.0,
            2.0, 1.0, 6.0, 1.0,
            3.0, 2.0, 1.0, 7.0,
        ]).unwrap();

        let a_pinv = _qrpinv(&matr_a, None).unwrap();

        // Condition 1: A * A^+ * A = A
        let mut tmp1 = MatrixFull::new([4, 4], 0.0);
        let mut cond1 = MatrixFull::new([4, 4], 0.0);
        _dgemm_full(&matr_a, 'N', &a_pinv, 'N', &mut tmp1, 1.0, 0.0);
        _dgemm_full(&tmp1, 'N', &matr_a, 'N', &mut cond1, 1.0, 0.0);
        let diff1 = matr_a.data_ref().unwrap().iter()
            .zip(cond1.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff1 < 1e-10, "Condition 1 failed, diff: {}", diff1);

        // Condition 2: A^+ * A * A^+ = A^+
        let mut tmp2 = MatrixFull::new([4, 4], 0.0);
        let mut cond2 = MatrixFull::new([4, 4], 0.0);
        _dgemm_full(&a_pinv, 'N', &matr_a, 'N', &mut tmp2, 1.0, 0.0);
        _dgemm_full(&tmp2, 'N', &a_pinv, 'N', &mut cond2, 1.0, 0.0);
        let diff2 = a_pinv.data_ref().unwrap().iter()
            .zip(cond2.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff2 < 1e-10, "Condition 2 failed, diff: {}", diff2);

        // Condition 3: (A * A^+)^T = A * A^+ (symmetric)
        let _tmp3 = tmp1.clone();
        let transposed = tmp1.transpose();
        let diff3 = tmp1.data_ref().unwrap().iter()
            .zip(transposed.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff3 < 1e-10, "Condition 3 failed, diff: {}", diff3);

        // Condition 4: (A^+ * A)^T = A^+ * A (symmetric)
        let transposed4 = tmp2.transpose();
        let diff4 = tmp2.data_ref().unwrap().iter()
            .zip(transposed4.data_ref().unwrap().iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff4 < 1e-10, "Condition 4 failed, diff: {}", diff4);
    }

    #[test]
    fn test_qrpinv_vs_pinv() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for n in [2usize, 3, 5, 8] {
            for _ in 0..5 {
                let data: Vec<f64> = (0..n * n).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let matr_a = MatrixFull::from_vec([n, n], data).unwrap();

                let pinv_svd = _pinv(&matr_a, None).unwrap();
                let pinv_qr = _qrpinv(&matr_a, None).unwrap();

                let diff = pinv_svd.data_ref().unwrap().iter()
                    .zip(pinv_qr.data_ref().unwrap().iter())
                    .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
                assert!(diff < 1e-12,
                    "QR and SVD pinv disagree for n={}, diff={}", n, diff);
            }
        }
    }

    #[test]
    fn test_qrpinv_vs_pinv_rank_deficient() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for n in [3usize, 5, 7] {
            for _ in 0..3 {
                let base: Vec<f64> = (0..(n - 1) * n).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let col0: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let col1: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

                let mut data = vec![0.0f64; n * n];
                // Fill first n-1 columns with random data
                for j in 0..(n - 1) {
                    for i in 0..n {
                        data[i + j * n] = base[i + j * n];
                    }
                }
                // Make column n-1 a linear combination of columns 0 and 1
                for i in 0..n {
                    data[i + (n - 1) * n] = col0[i] + col1[i];
                }

                let matr_a = MatrixFull::from_vec([n, n], data).unwrap();

                let pinv_svd = _pinv(&matr_a, Some(1e-8)).unwrap();
                let pinv_qr = _qrpinv(&matr_a, Some(1e-8)).unwrap();

                let diff = pinv_svd.data_ref().unwrap().iter()
                    .zip(pinv_qr.data_ref().unwrap().iter())
                    .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
                assert!(diff < 1e-10,
                    "QR and SVD rank-deficient pinv disagree for n={}, diff={}", n, diff);
            }
        }
    }

    #[test]
    fn test_qrpinv_custom_rcond() {
        let matr_a = MatrixFull::from_vec([3, 3], vec![
            10.0, 0.0, 0.0,
            0.0, 0.1, 0.0,
            0.0, 0.0, 0.001,
        ]).unwrap();

        let pinv_rcond_high = _qrpinv(&matr_a, Some(0.1)).unwrap();
        let pinv_rcond_low = _qrpinv(&matr_a, Some(1e-6)).unwrap();

        // With rcond=0.1, only the largest sv (10.0) is kept, so pinv* A = diag(1,0,0)
        let mut check = MatrixFull::new([3, 3], 0.0);
        _dgemm_full(&pinv_rcond_high, 'N', &matr_a, 'N', &mut check, 1.0, 0.0);
        let diff = (check[[0, 0]] - 1.0).abs() + check[[1, 1]].abs() + check[[2, 2]].abs();
        assert!(diff < 1e-10, "High rcond: pinv*A should be diag(1,0,0), got diff={}", diff);

        // With rcond=1e-6, all 3 values are kept, pinv * A should be I
        let mut check2 = MatrixFull::new([3, 3], 0.0);
        _dgemm_full(&pinv_rcond_low, 'N', &matr_a, 'N', &mut check2, 1.0, 0.0);
        let id_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let diff2 = check2.data_ref().unwrap().iter()
            .zip(id_data.iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powf(2.0));
        assert!(diff2 < 1e-10, "Low rcond: pinv*A should be I, diff={}", diff2);
    }*/
} 
