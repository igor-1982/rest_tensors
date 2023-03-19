//! # rest_tensors
//!                                                                                                                                                                          
//! **rest_tensors** is a linear algebra library, which aims at providing efficient tensor operations for the Rust-based electronic structure tool (REST).
//! 
//!  ### Using rest_tensors
//! 
//! - Several global environment variables should be specified  
//!   1) REST_BLAS_DIR:           The path to the openblas library: `libopenblas.so`  
//!   2) REST_FORTRAN_COMPILER:   The compiler to build a fortran library for effcient tensor operations:  `restmatr.f90` -> `librestmatr.so`  
//!   3) REST_EXT_DIR:            The path to store the fortran library: `librestmatr.so` after compilation 
//!   4) LD_LIBRARY_PATH:         attach REST_BLAS_DIR and REST_EXT_DIR to LD_LIBRARY_PATH: `export LD_LIBRARY_PATH="$REST_BLAS_DIR:$REST_EXT_DIR:$LD_LIBRARY_PATH"` 
//! 
//! - Simply add the following to your Carto.toml file:
//! ```ignore
//! [dependencies]
//! // replace the * by the latest version
//! rest_tensors = "*"
//! ```
//! 
//!  ### Fetures
//! 
//!    * [`MatrixFull`](MatrixFull): the `column-major` rank-2 tensor, i.e. `matrix`, which is used for the molecular geometries, 
//!                   orbital coefficients, density matrix, and most of intermediate data for REST.  
//! There are several relevant structures for matrix, which share the same trait, namely
//!                   [`BasicMatrix`](BasicMatrix), [`BasicMatrixOpt`](BasicMatrixOpt), [`MathMatrix`](MathMatrix) and so forth. 
//!    * [`MatrixUpper`](MatrixUpper): the structure storing the upper triangle of the matrix, which is used for Hamiltonian matrix, and many other Hermitian matrices in the REST package.
//!    * [`RIFull`](RIFull):  the `column-major` rank-3 tensor structure, which is used for the three-center integrals 
//!                   in the resoution-of-identity approximation (RI). For example, ri3ao, ri3mo, and so forth.   
//! **NOTE**:: Although RIFull is created for very specific purpose use in REST, most of the relevant operations provided here are quite general and can be easily extended to any other 3-rank tensors 
//!    * [`ERIFull`](ERIFull): the `column-major` 4-dimention tensors for electronic repulsive integrals (ERI).  
//! **NOTE**:: ERIFull is created to handle the analytic electronic-repulsive integrals in REST. 
//! Because REST mainly uses the Resolution-of-Identity (RI) technique. The analytic ERI is provided for benchmark, and thus is not fully optimized.
//! 
//! 
//!    *  Detailed usage of [`MatrixFull`](MatrixFull) can be find in the corresponding pages; while those of [`RIFull`] and [`ERIFull`] are not yet ready.
//! 
//!  ### To-Do-List
//! 
//!   * Introduce more LAPACK and BLAS functions to the 2-dimention matrix struct in rest-tensors, like [`MatrixFull`](MatrixFull), [`MatrixFullSlice`](MatrixFullSlice), [`SubMatrixFull`](SubMatrixFull) and so forth.
//!   * Reoptimize the API for the rank-3 tensor, mainly [`RIFull`](RIFull) and complete the detailed usage accordingly.
//!   * Enable the ScaLAPCK (scalable linear algebra package) functions to the 2-dimention matrix struct in rest-tensors, like [`MatrixFull`](MatrixFull).
//!   * Conversions between `rest_tensors` and `numpy` in python
