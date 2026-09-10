use num_complex::{Complex, ComplexFloat};
use crate::matrix::MatrixFull;
use rayon::prelude::*;
use num_traits::Zero;
use std::ffi::c_char;
use std::ops::Neg;
use num_traits::Num;

impl<T: Clone + Copy + Send + Sync + Zero> MatrixFull<Complex<T>> {
    pub fn from_matrixfull(re_part: &MatrixFull<T>, im_part: &MatrixFull<T>) -> Option<MatrixFull<Complex<T>>> {
        let re_size = re_part.size;
        let im_size = im_part.size;
        if (re_size[0] != im_size[0]) || (re_size[1] != im_size[1]) {
            panic!("Error: mismatch between re part and im part. Find {:?} and {:?}",re_size,im_size);
            None
        } else {
            let mut new_vec: Vec<Complex<T>> = vec![Complex::new(T::zero(),T::zero());re_size[0]*re_size[1]];
            re_part.data.par_iter().zip(im_part.data.par_iter()).zip(new_vec.par_iter_mut()).for_each(|((re, im), comp)| {
                *comp = Complex::new(re.clone(),im.clone());
            });
            MatrixFull::from_vec(re_size,new_vec)
        }
    }

    pub fn from_matrixfull_re(re_part: &MatrixFull<T>) -> Option<MatrixFull<Complex<T>>> {
        let size = re_part.size;
        let mut new_vec: Vec<Complex<T>> = vec![Complex::new(T::zero(),T::zero());size[0]*size[1]];
        re_part.data.par_iter().zip(new_vec.par_iter_mut()).for_each(|(re,comp)| {
            *comp = Complex::new(re.clone(),T::zero());
        });
        MatrixFull::from_vec(size,new_vec)
    }

    pub fn from_matrixfull_im(im_part: &MatrixFull<T>) -> Option<MatrixFull<Complex<T>>> {
        let size = im_part.size;
        let mut new_vec: Vec<Complex<T>> = vec![Complex::new(T::zero(),T::zero());size[0]*size[1]];
        im_part.data.par_iter().zip(new_vec.par_iter_mut()).for_each(|(im,comp)| {
            *comp = Complex::new(T::zero(),im.clone());
        });
        MatrixFull::from_vec(size,new_vec)
    }
}

impl <T: Clone + Copy + Send + Sync + Neg + Num + Neg<Output=T>> MatrixFull<Complex<T>> {
    pub fn hermitian(&self) -> MatrixFull<Complex<T>> {
        //get hermitian matrix
        let [x_len, y_len] = self.size;
        let herm_data: Vec<_> = (0..x_len).into_par_iter().flat_map(|x| {
            (0..y_len).map(|y| {
                let src_index = x + y * x_len;
                self.data[src_index].conj()
            }).collect::<Vec<_>>()}).collect();

        MatrixFull::from_vec([y_len,x_len],herm_data).unwrap()
    }
}

impl MatrixFull<Complex<f64>> {
    //Some LAPACK subroutines
    pub fn lapack_zgemm(&mut self, a: &MatrixFull<Complex<f64>>, b: &MatrixFull<Complex<f64>>, opa: char, opb: char, alpha: Complex<f64>, beta: Complex<f64>) {
        //calculate self = alpha * opa(a) * opb(b) + beta * self 
        let m = if opa=='N' {a.size[0]} else {a.size[1]};
        let n = if opb=='N' {b.size[1]} else {b.size[0]};
        let k = if opa=='N' {a.size[1]} else {a.size[0]};
        let lda = if opa=='N' {m.max(1)} else {k.max(1)};
        let ldb = if opb=='N' {k.max(1)} else {n.max(1)};
        let ldc = m.max(1);
        unsafe {
            zgemm_(
            &(opa as i8) as *const c_char,
            &(opb as i8) as *const c_char,
            &(m as i32) as *const i32,
            &(n as i32) as *const i32,
            &(k as i32) as *const i32,
            &alpha as *const Complex<f64>,
            a.data.as_ptr(),
            &(lda as i32) as *const i32,
            b.data.as_ptr(),
            &(ldb as i32) as *const i32,
            &beta as *const Complex<f64>,
            self.data.as_mut_ptr(),
            &(ldc as i32) as *const i32,
            );
        };
    }

    pub fn lapack_zgeev(&mut self, jobvl: char, jobvr: char, w: &mut MatrixFull<Complex<f64>>, vl: Option<&mut MatrixFull<Complex<f64>>>, vr: Option<&mut MatrixFull<Complex<f64>>>) {
        // The right eigenvector v(j) of A satisfies
        //      A * v(j) = lambda(j) * v(j)
        // where lambda(j) is its eigenvalue.
        // The left eigenvector u(j) of A satisfies
        //      u(j)**H * A = lambda(j) * u(j)**H
        //
        // if jobvl/jobvr = 'N', vl/vr could be None as it's not referenced in LAPACK
        if self.size[0] != self.size[1] {
            println!("Warning: ZGEEV computes N-by-N matrix, found {:?}",self.size);
        }
        let n = self.size[0];
        let mut info = 0;
        let mut work = vec![Complex::new(0.0,0.0);(2*n)];
        let mut rwork = vec![0.0;(2*n)];
        let vl_ptr = match vl {
            Some(m) => m.data.as_mut_ptr(),
            None => std::ptr::null_mut(),
        };
        let vr_ptr = match vr {
            Some(m) => m.data.as_mut_ptr(),
            None => std::ptr::null_mut(),
        };
        unsafe {
            zgeev_(
                &(jobvl as i8) as *const c_char,
                &(jobvr as i8) as *const c_char,
                &(n as i32) as *const i32,
                self.data.as_mut_ptr(),
                &(n as i32) as *const i32,
                w.data.as_mut_ptr(),
                vl_ptr,
                &(n as i32) as *const i32,
                vr_ptr,
                &(n as i32) as *const i32,
                work.as_mut_ptr(),
                &((2*n) as i32) as *const i32,
                rwork.as_mut_ptr(),
                &mut info as *mut i32,
                )
        }
        if info != 0 {
            println!("Warning: ZGEEV returns INFO: {:?}",info);
        }
    }

    pub fn lapack_zggev(&mut self, jobvl: char, jobvr: char, b: &mut MatrixFull<Complex<f64>>, vl: Option<&mut MatrixFull<Complex<f64>>>, vr: Option<&mut MatrixFull<Complex<f64>>>, alpha: &mut MatrixFull<Complex<f64>>, beta: &mut MatrixFull<Complex<f64>>) {
        // The right generalized eigenvector v(j) corresponding to the
        // generalized eigenvalue lambda(j) of (A,B) satisfies
        //
        //              A * v(j) = lambda(j) * B * v(j).
        //
        // The left generalized eigenvector u(j) corresponding to the
        // generalized eigenvalues lambda(j) of (A,B) satisfies
        //
        //              u(j)**H * A = lambda(j) * u(j)**H * B
        //
        // where u(j)**H is the conjugate-transpose of u(j).
        // On exit, ALPHA(j)/BETA(j), j=1,...,N, will be the generalized eigenvalues.
        if self.size[1] != b.size[1] {
            panic!("Matrix A and B should have the same order, found {:?} and {:?}",self.size,b.size);
        }
        let n = self.size[1];
        let mut info = 0;
        let mut work = vec![Complex::new(0.0,0.0);(2*n)];
        let mut rwork = vec![0.0;(8*n)];
        let ldb = b.size[0];
        let lda = self.size[0];
        let vl_ptr = match vl {
            Some(m) => m.data.as_mut_ptr(),
            None => std::ptr::null_mut(),
        };
        let vr_ptr = match vr {
            Some(m) => m.data.as_mut_ptr(),
            None => std::ptr::null_mut(),
        };
        unsafe {
            zggev_(
                &(jobvl as i8) as *const c_char,
                &(jobvr as i8) as *const c_char,
                &(n as i32) as *const i32,
                self.data.as_mut_ptr(),
                &(lda as i32) as *const i32,
                b.data.as_mut_ptr(),
                &(ldb as i32) as *const i32,
                alpha.data.as_mut_ptr(),
                beta.data.as_mut_ptr(),
                vl_ptr,
                &(n as i32) as *const i32,
                vr_ptr,
                &(n as i32) as *const i32,
                work.as_mut_ptr(),
                &((2*n) as i32) as *const i32,
                rwork.as_mut_ptr(),
                &mut info as *mut i32,
                )
        }
        if info != 0 {
            println!("Warning: ZGGEV returns INFO: {:?}",info);
        }
    }

    pub fn lapack_ztrtrs(&mut self, uplo: char, trans: char, diag: char, b: &mut MatrixFull<Complex<f64>>) {
        // ZTRTRS solves a triangular system of the form
        //
        //    A * X = B,  A**T * X = B,  or  A**H * X = B,
        //
        // where A is a triangular matrix of order N, and B is an N-by-NRHS matrix.
        if self.size[0] != self.size[1] {
            println!("Warning: ZTRTRS should apply for N-by-N matrix, found {:?}",self.size);
        }
        let n = self.size[1];
        let nrhs = b.size[1];
        let ldb = b.size[0];
        let mut info = 0;
        unsafe {
            ztrtrs_(
                &(uplo as i8) as *const c_char,
                &(trans as i8) as *const c_char,
                &(diag as i8) as *const c_char,
                &(n as i32) as *const i32,
                &(nrhs as i32) as *const i32,
                self.data.as_mut_ptr(),
                &(n as i32) as *const i32,
                b.data.as_mut_ptr(),
                &(ldb as i32) as *const i32,
                &mut info as *mut i32,
            )
        }
        if info != 0 {
            println!("Warning: ZTRTRS returns INFO: {:?}",info);
        }
    }
    
    pub fn lapack_zpotrf(&mut self, uplo: char) {
        // ZPOTRF computes the Cholesky factorization of a complex Hermitian
        // positive definite matrix A.
        //
        // The factorization has the form
        //    A = U**H * U,  if UPLO = 'U', or
        //    A = L  * L**H,  if UPLO = 'L',
        // where U is an upper triangular matrix and L is lower triangular.
        let n = self.size[1];
        let lda = self.size[0];
        if lda < n.max(1usize) {
            panic!("The leading dimension must larger than the order of matrix");
        }
        let mut info = 0;
        unsafe {
            zpotrf_(
                &(uplo as i8) as *const c_char,
                &(n as i32) as *const i32,
                self.data.as_mut_ptr(),
                &(lda as i32) as *const i32,
                &mut info as *mut i32,
                )
        }
        if info != 0 {
            println!("Warning: ZPOTRF returns INFO: {:?}",info);
        }
    }

    pub fn lapack_zgetrf(&mut self) -> Vec<i32> {
        // The factorization has the form
        //    A = P * L * U
        // where P is a permutation matrix, L is lower triangular with unit
        // diagonal elements (lower trapezoidal if m > n), and U is upper
        // triangular (upper trapezoidal if m < n).
        let m = self.size[0] as i32;
        let n = self.size[1] as i32;
        let lda = m.max(1);
        let ip_dim = m.min(n) as usize;
        let mut ipiv = vec![0 as i32;ip_dim];
        let mut info = 0;
        unsafe {
            zgetrf_(
                &m as *const i32,
                &n as *const i32,
                self.data.as_mut_ptr(),
                &lda as *const i32,
                ipiv.as_mut_ptr(),
                &mut info as *mut i32,
                );
        }
        if info != 0 {
            println!("Warning: ZGETRF returns INFO: {:?}",info);
        }
        ipiv
    }

    pub fn lapack_zgetri(&mut self, ipiv: &Vec<i32>) {
        // This method inverts U and then computes inv(A) by solving the system
        // inv(A)*L = inv(U) for inv(A).
        let n = self.size[1];
        let lda = self.size[0];
        let lwork = n;
        let mut work = vec![Complex::new(0.0,0.0);lwork];
        let mut info = 0 as i32;
        unsafe {
            zgetri_(
                &(n as i32) as *const i32,
                self.data.as_mut_ptr(),
                &(lda as i32) as *const i32,
                ipiv.as_ptr(),
                work.as_mut_ptr(),
                &(lwork as i32) as *const i32,
                &mut info as *mut i32,
                );
        }
        if info != 0 {
            println!("Warning: ZGETRI returns INFO: {:?}",info);
        }
    }

    pub fn lapack_zheev(&mut self, jobz: char, uplo: char, w: &mut [f64]) {
        if self.size[0] != self.size[1] {
            println!("Warning: ZHEEV computes N-by-N matrix, found {:?}",self.size);
        }
        let n = self.size[0];
        let lda = n.max(1);
        // Workspace query: call with LWORK=-1 to get optimal LWORK
        let mut wkopt = Complex::new(0.0, 0.0);
        let lwork_query: i32 = -1;
        let mut rwork = vec![0.0;(3 * n).max(1)];
        let mut info: i32 = 0;
        unsafe {
            zheev_(
                &(jobz as i8) as *const c_char,
                &(uplo as i8) as *const c_char,
                &(n as i32) as *const i32,
                self.data.as_mut_ptr(),
                &(lda as i32) as *const i32,
                w.as_mut_ptr(),
                &mut wkopt as *mut Complex<f64>,
                &lwork_query as *const i32,
                rwork.as_mut_ptr(),
                &mut info as *mut i32,
                );
        }
        let lwork = (wkopt.re as i32).max((2 * n as i32) - 1).max(1);
        let mut work = vec![Complex::new(0.0,0.0); lwork as usize];
        info = 0;
        unsafe {
            zheev_(
                &(jobz as i8) as *const c_char,
                &(uplo as i8) as *const c_char,
                &(n as i32) as *const i32,
                self.data.as_mut_ptr(),
                &(lda as i32) as *const i32,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &(lwork as i32) as *const i32,
                rwork.as_mut_ptr(),
                &mut info as *mut i32,
                );
        }
        if info != 0 {
            println!("Warning: ZHEEV returns INFO: {:?}",info);
        }
    }

    pub fn lapack_zheevd(&mut self, jobz: char, uplo: char, w: &mut [f64]) {
        if self.size[0] != self.size[1] {
            println!("Warning: ZHEEVD computes N-by-N matrix, found {:?}",self.size);
        }
        let n = self.size[0];
        let lda = n.max(1);
        // Workspace query: LWORK=-1, LRWORK=-1, LIWORK=-1
        let mut wkopt = Complex::new(0.0, 0.0);
        let mut rwopt = 0.0f64;
        let mut iwopt: i32 = 0;
        let lwork_query: i32 = -1;
        let lrwork_query: i32 = -1;
        let liwork_query: i32 = -1;
        let mut info: i32 = 0;
        unsafe {
            zheevd_(
                &(jobz as i8) as *const c_char,
                &(uplo as i8) as *const c_char,
                &(n as i32) as *const i32,
                self.data.as_mut_ptr(),
                &(lda as i32) as *const i32,
                w.as_mut_ptr(),
                &mut wkopt as *mut Complex<f64>,
                &lwork_query as *const i32,
                &mut rwopt as *mut f64,
                &lrwork_query as *const i32,
                &mut iwopt as *mut i32,
                &liwork_query as *const i32,
                &mut info as *mut i32,
            );
        }
        let lwork = (wkopt.re as i32).max(1);
        let lrwork = (rwopt as i32).max(1);
        let liwork = (iwopt as i32).max(1);
        let mut work = vec![Complex::new(0.0,0.0); lwork as usize];
        let mut rwork = vec![0.0; lrwork as usize];
        let mut iwork = vec![0i32; liwork as usize];
        info = 0;
        unsafe {
            zheevd_(
                &(jobz as i8) as *const c_char,
                &(uplo as i8) as *const c_char,
                &(n as i32) as *const i32,
                self.data.as_mut_ptr(),
                &(lda as i32) as *const i32,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &(lwork as i32) as *const i32,
                rwork.as_mut_ptr(),
                &(lrwork as i32) as *const i32,
                iwork.as_mut_ptr(),
                &(liwork as i32) as *const i32,
                &mut info as *mut i32,
            );
        }
        if info != 0 {
            println!("Warning: ZHEEVD returns INFO: {:?}",info);
        }
    }

    pub fn lapack_zhegv(&mut self, itype: i32, jobz: char, uplo: char, b: &mut MatrixFull<Complex<f64>>, w: &mut [f64]) {
        if self.size[0] != self.size[1] || b.size[0] != b.size[1] || self.size[0] != b.size[0] {
            panic!("lapack_zhegv: A/B dimensions mismatch: A={:?}, B={:?}", self.size, b.size);
        }
        let n = self.size[0] as i32;
        let lda = self.size[0] as i32;
        let ldb = b.size[0] as i32;

        // workspace query: LWORK=-1
        let mut work_query = vec![Complex::new(0.0, 0.0); 1];
        let mut rwork_query = vec![0.0f64; 1];
        let lwork_q: i32 = -1;
        let mut info: i32 = 0;
        unsafe {
            zhegv_(
                &itype as *const i32,
                &(jobz as i8) as *const c_char,
                &(uplo as i8) as *const c_char,
                &n as *const i32,
                self.data.as_mut_ptr(),
                &lda as *const i32,
                b.data.as_mut_ptr(),
                &ldb as *const i32,
                w.as_mut_ptr(),
                work_query.as_mut_ptr(),
                &lwork_q as *const i32,
                rwork_query.as_mut_ptr(),
                &mut info as *mut i32,
            );
        }
        if info != 0 { panic!("ZHEGV workspace query failed, info={}", info); }
        let lwork = (work_query[0].re as i32).max(1);
        let lrwork = ((3 * n - 2).max(1)) as usize;
        let mut work = vec![Complex::new(0.0, 0.0); lwork as usize];
        let mut rwork = vec![0.0f64; lrwork];
        info = 0;
        unsafe {
            zhegv_(
                &itype as *const i32,
                &(jobz as i8) as *const c_char,
                &(uplo as i8) as *const c_char,
                &n as *const i32,
                self.data.as_mut_ptr(),
                &lda as *const i32,
                b.data.as_mut_ptr(),
                &ldb as *const i32,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork as *const i32,
                rwork.as_mut_ptr(),
                &mut info as *mut i32,
            );
        }
        if info != 0 { panic!("ZHEGV failed, info={}", info); }
    }
}

#[link(name="openblas")]
extern "C" {
    fn zgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const Complex<f64>,
        a: *const Complex<f64>,
        lda: *const i32,
        b: *const Complex<f64>,
        ldb: *const i32,
        beta: *const Complex<f64>,
        c: *mut Complex<f64>,
        ldc: *const i32,
    );
}

#[link(name="openblas")]
extern "C" {
    fn zgeev_(
        jobvl: *const c_char,
        jobvr: *const c_char,
        n: *const i32,
        a: *const Complex<f64>,
        lda: *const i32,
        w: *mut Complex<f64>,
        vl: *mut Complex<f64>,
        ldvl: *const i32,
        vr: *mut Complex<f64>,
        ldvr: *const i32,
        work: *mut Complex<f64>,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
        );
}

#[link(name="openblas")]
extern "C" {
    fn zggev_(
        jobvl: *const c_char,
        jobvr: *const c_char,
        n: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        b: *mut Complex<f64>,
        ldb: *const i32,
        alpha: *mut Complex<f64>,
        beta: *mut Complex<f64>,
        vl: *mut Complex<f64>,
        ldvl: *const i32,
        vr: *mut Complex<f64>,
        ldvr: *const i32,
        work: *mut Complex<f64>,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
        );
}

#[link(name="openblas")]
extern "C" {
    fn zpotrf_(
        uplo: *const c_char,
        n: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        info: *mut i32);
}

#[link(name="openblas")]
extern "C" {
    fn ztrtrs_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        b: *mut Complex<f64>,
        ldb: *const i32,
        info: *mut i32);
}

#[link(name="openblas")]
extern "C" {
    fn zgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
        );
}

#[link(name="openblas")]
extern "C" {
    fn zgetri_(
        n: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        ipiv: *const i32,
        work: *mut Complex<f64>,
        lwork: *const i32,
        info: *mut i32,
        );
}

#[link(name="openblas")]
extern "C" {
    fn zheev_(
        jobz: *const c_char,
        uplo: *const c_char,
        n: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        w: *mut f64,
        work: *mut Complex<f64>,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
    );
}

#[link(name="openblas")]
extern "C" {
    fn zheevd_(
        jobz: *const c_char,
        uplo: *const c_char,
        n: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        w: *mut f64,
        work: *mut Complex<f64>,
        lwork: *const i32,
        rwork: *mut f64,
        lrwork: *const i32,
        iwork: *mut i32,
        liwork: *const i32,
        info: *mut i32,
    );
}

#[link(name="openblas")]
extern "C" {
    fn zhegv_(
        itype: *const i32,
        jobz: *const c_char,
        uplo: *const c_char,
        n: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        b: *mut Complex<f64>,
        ldb: *const i32,
        w: *mut f64,
        work: *mut Complex<f64>,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
    );
}

#[test]
fn test_complexmatrixfull() {
    let a_re = MatrixFull::from_vec([2,3],vec![1.0,2.0,3.0,4.0,5.0,6.0]).unwrap();
    let a_im = MatrixFull::from_vec([2,3],vec![7.0,8.0,9.0,10.0,11.0,12.0]).unwrap();
    let a_complex = MatrixFull::from_matrixfull(&a_re,&a_im).unwrap();
    let herm_a_complex = a_complex.hermitian();
    println!("re = {:?}\nim = {:?}\n a = {:?}\na^H = {:?}",a_re,a_im,a_complex,herm_a_complex);
    let mut a_re_complex = MatrixFull::from_matrixfull_re(&a_re).unwrap();
    let mut a_im_complex = MatrixFull::from_matrixfull_im(&a_im).unwrap();
    println!("{:?}{:?}",a_re_complex,a_im_complex);

    let mut b_re = MatrixFull::from_vec([2,2],vec![2.0,3.0,4.0,5.0]).unwrap();
    let mut b_im = MatrixFull::from_vec([2,2],vec![6.0,7.0,8.0,9.0]).unwrap();
    let mut b_complex = MatrixFull::from_matrixfull(&b_re,&b_im).unwrap();
    let b_copy = b_complex.clone();

    let mut a_re = MatrixFull::from_vec([2,2],vec![1.0,2.0,3.0,4.0]).unwrap();
    let mut a_im = MatrixFull::from_vec([2,2],vec![7.0,8.0,9.0,10.0]).unwrap();
    let mut a_copy = MatrixFull::from_matrixfull(&a_re,&a_im).unwrap();

    println!("Before zgemm: {:?},{:?}",a_copy,b_copy);
    let mut c = MatrixFull::new([2,2],Complex::new(0.0,0.0));
    c.lapack_zgemm(&a_copy,&b_copy,'N','N',Complex::new(1.0,0.0),Complex::new(0.0,0.0));
    println!("After zgemm: {:?}",c);
    let mut c_copy = c.clone();

    let mut w = MatrixFull::new([1,2],Complex::new(0.0,0.0));
    let mut vl = None;
    let mut vr = MatrixFull::new([2,2],Complex::new(0.0,0.0));
    c.lapack_zgeev('N','V',&mut w, vl, Some(&mut vr));
    println!("After zgeev: {:?}, {:?}",w,vr);

    let mut alpha = MatrixFull::new([1,2],Complex::new(0.0,0.0));
    let mut beta = MatrixFull::new([1,2],Complex::new(0.0,0.0));
    let mut vl = None;
    let mut vr = MatrixFull::new([2,2],Complex::new(0.0,0.0));
    let mut d = MatrixFull::from_vec([2,2],vec![Complex::new(2.0,6.0),Complex::new(3.0,7.0),Complex::new(4.0,8.0),Complex::new(5.0,9.0)]).unwrap();
//    let mut d = MatrixFull::from_vec([2,2],vec![Complex::new(1.0,0.0),Complex::new(0.0,0.0),Complex::new(0.0,0.0),Complex::new(1.0,0.0)]).unwrap();
    println!("Before zggev: {:?}, {:?}",c_copy,d);
    c_copy.lapack_zggev('N','V',&mut d, vl, Some(&mut vr), &mut alpha, &mut beta);
    println!("After zggev: {:?}, {:?}, {:?}",alpha,beta,vr);
    let mut theta = vec![Complex::new(0.0,0.0);2];
    alpha.data.iter().zip(beta.data.iter()).zip(theta.iter_mut()).for_each(|((aa,bb),cc)| {
        *cc = *aa / *bb;
    });
    println!("zggev value: {:?}",theta);

    //test inverse by cholesky + ztrtrs
    let mut tmp = MatrixFull::from_vec([2,2],vec![Complex::new(6.0,6.0),Complex::new(3.0,7.0),Complex::new(4.0,8.0),Complex::new(5.0,9.0)]).unwrap();
    let mut herm_tmp = tmp.hermitian();
    let mut target = MatrixFull::new([2,2],Complex::new(0.0,0.0));
    target.data.iter_mut().zip(tmp.data.iter()).zip(herm_tmp.data.iter()).for_each(|((t,aa),bb)| {
        *t = (*aa + *bb)/2.0;
    });
    println!("Before: {:?}",target);

    target.lapack_zpotrf('L');
    println!("After: {:?}",target);

    let mut identity = MatrixFull::from_vec([2,2],vec![Complex::new(1.0,0.0),Complex::new(0.0,0.0),Complex::new(0.0,0.0),Complex::new(1.0,0.0)]).unwrap();
    target.lapack_ztrtrs('L','N','N',&mut identity);
    println!("After: {:?}",identity);

    let mut inv_target = MatrixFull::new([2,2],Complex::new(0.0,0.0));
    inv_target.lapack_zgemm(&identity,&identity,'C','N',Complex::new(1.0,0.0),Complex::new(0.0,0.0));
    println!("inv = {:?}",inv_target);

    let mut tmp = MatrixFull::from_vec([2,2],vec![Complex::new(6.0,6.0),Complex::new(3.0,7.0),Complex::new(4.0,8.0),Complex::new(5.0,9.0)]).unwrap();
    let mut herm_tmp = tmp.hermitian();
    let mut target = MatrixFull::new([2,2],Complex::new(0.0,0.0));
    target.data.iter_mut().zip(tmp.data.iter()).zip(herm_tmp.data.iter()).for_each(|((t,aa),bb)| {
        *t = (*aa + *bb)/2.0;
    });
    let mut ipiv = target.lapack_zgetrf();
    target.lapack_zgetri(&ipiv);
    println!("inv = {:?}",target);
}

#[test]
fn test_zheev_zhegv() {
    // Build a 3x3 Hermitian matrix:
    // H = [[  2,    1+i,    0  ],
    //      [1-i,     3,   2+2i ],
    //      [  0,  2-2i,    4  ]]
    let re = MatrixFull::from_vec([3,3], vec![
        2.0, 1.0, 0.0,
        1.0, 3.0, 2.0,
        0.0, 2.0, 4.0,
    ]).unwrap();
    let im = MatrixFull::from_vec([3,3], vec![
        0.0,  1.0,  0.0,
       -1.0,  0.0,  2.0,
        0.0, -2.0,  0.0,
    ]).unwrap();
    let mut h = MatrixFull::from_matrixfull(&re, &im).unwrap();
    let h_copy = h.clone();

    // Test zheev
    let mut w = vec![0.0; 3];
    h.lapack_zheev('V', 'U', &mut w);
    println!("zheev eigenvalues: {:?}", w);

    // Verify: H * V = V * diag(λ)
    let mut hv = MatrixFull::new([3, 3], Complex::new(0.0, 0.0));
    hv.lapack_zgemm(&h_copy, &h, 'N', 'N',
        Complex::new(1.0, 0.0), Complex::new(0.0, 0.0));
    for j in 0..3 {
        let lambda = Complex::new(w[j], 0.0);
        for i in 0..3 {
            let diff = (hv[(i, j)] - h[(i, j)] * lambda).norm();
            assert!(diff < 1e-10, "Eigenpair check failed at ({}, {}): diff = {}", i, j, diff);
        }
    }
    println!("zheev: all eigenpair checks passed");

    // Test zhegv: A*x = λ*B*x with B = I, should give same eigenvalues as zheev
    let mut a = h_copy.clone();
    let mut b = MatrixFull::from_vec([3, 3], vec![
        Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
    ]).unwrap();
    let mut w2 = vec![0.0; 3];
    a.lapack_zhegv(1, 'V', 'U', &mut b, &mut w2);
    println!("zhegv eigenvalues: {:?}", w2);

    // Compare with zheev results
    for i in 0..3 {
        assert!((w[i] - w2[i]).abs() < 1e-10,
            "zhegv vs zheev eigenvalue mismatch at {}: {} vs {}", i, w[i], w2[i]);
    }
    println!("zhegv: all checks passed");
}
