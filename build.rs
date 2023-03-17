extern crate dunce;
//use std::{env, path::PathBuf};
use std::{env, process::Command};

fn main() {

    let external_dir = if let Ok(external_dir) = env::var("REST_EXT_DIR") {
        external_dir
    } else {"".to_string()};
    let blas_dir = if let Ok(blas_dir) = env::var("REST_BLAS_DIR") {
        blas_dir
    } else {"".to_string()};
    let fortran_compiler = if let Ok(fortran_compiler) = env::var("REST_FORTRAN_COMPILER") {
        fortran_compiler
    } else {"gfortran".to_string()};


    let restmatr_file = format!("{}/restmatr.f90",&external_dir);
    let restmatr_libr = format!("{}/librestmatr.so",&external_dir);
    let restmatr_link = format!("-L{} -lopenblas",&blas_dir);

    Command::new(fortran_compiler)
        .args(&["-shared", "-fpic", "-O2",&restmatr_file,"-o",&restmatr_libr,&restmatr_link])
        .status().unwrap();


    println!("cargo:rustc-link-lib=restmatr");
    println!("cargo:rustc-link-search=native={}",&external_dir);


    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-search=native={}",&blas_dir);

    println!("cargo:rerun-if-changed={}/restmatr.f90", &external_dir);
    println!("cargo:rerun-if-changed={}/librestmatr.so", &external_dir);

}
