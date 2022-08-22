//extern crate nalgebra as na;
//
//use eigenvalues::algorithms::davidson::Davidson;
//use eigenvalues::utils::generate_diagonal_dominant;
//use eigenvalues::{DavidsonCorrection, SpectrumTarget};
//use nalgebra::{DMatrix, DMatrixSlice, DVector, DVectorSlice};
//
//pub fn davidson(Ax: Box<dyn FnMut>) -> (DVector<f64>, DMatrix<f64>) {
//    impl MatrixOperations for DMatrix<f64> {
//        fn matrix_vector_prod(&self, vs: DVectorSlice<f64>) -> DVector<f64> {
//            Ax(vs)
//        }
//    }
//    let arr = generate_diagonal_dominant(10, 0.005);
//    //let eig = sort_eigenpairs(nalgebra::linalg::SymmetricEigen::new(arr.clone()), true);
//    let spectrum_target = SpectrumTarget::Lowest;
//    let tolerance = 1.0e-4;
//
//    let dav = Davidson::new(
//        arr.clone(),
//        2,
//        DavidsonCorrection::DPR,
//        spectrum_target.clone(),
//        tolerance,
//    )
//    .unwrap();
//    (dav.eigenvalues, dav.eigenvectors)
//}
use std::default::Default;

#[derive(Debug)]
pub struct DavidsonParams { 
    pub tol: f64,
    pub maxcyc: i32,
    pub maxspace: i32,
    pub lindep: f64,
    pub nroots: i32
    }

impl Default for DavidsonParams {
    fn default() -> Self {
        DavidsonParams{
           tol:      1e-5, 
           maxcyc:   50,
           maxspace: 12,
           lindep:   1e-14,
           nroots:   1
        }
    }
}

pub fn davidson_solve(ax: Box<dyn FnMut(Vec<f64>) -> Vec<f64> + '_>,
                      x0: Vec<f64>,
                      params: &DavidsonParams
                      ){
    let tol = params.tol;
    let maxcyc = params.maxcyc;
    let maxspace = params.maxspace;
    let lindep = params.lindep;
    let nroots = params.nroots;
    println!("{:?}", tol);


}

