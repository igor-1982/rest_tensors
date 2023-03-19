//extern crate dunce;
use std::path::PathBuf;
use std::{fs, env, process::Command};

fn main() -> std::io::Result<()> {

    // the lib directory to store librestmatr.so
    let external_dir = if let Ok(target_dir) = env::var("REST_EXT_DIR") {
        PathBuf::from(target_dir)
    } else {PathBuf::from("".to_string())};

    if ! external_dir.is_dir() {
        fs::create_dir(&external_dir)?
    };

    let blas_dir = if let Ok(blas_dir) = env::var("REST_BLAS_DIR") {
        blas_dir
    } else {"".to_string()};
    let fortran_compiler = if let Ok(fortran_compiler) = env::var("REST_FORTRAN_COMPILER") {
        fortran_compiler
    } else {"gfortran".to_string()};


    let restmatr_file = format!("src/external_libs/restmatr.f90");
    //let restmatr_libr = format!("{}/librestmatr.so",&external_dir.to_str().unwrap());
    let restmatr_libr = format!("src/external_libs/librestmatr.so",&external_dir.to_str().unwrap());
    let restmatr_link = format!("-L{} -lopenblas",&blas_dir);

    Command::new(fortran_compiler)
        .args(&["-shared", "-fpic", "-O2",&restmatr_file,"-o",&restmatr_libr,&restmatr_link])
        .status().unwrap();


    println!("cargo:rustc-link-lib=restmatr");
    println!("cargo:rustc-link-search=native={}",&external_dir.to_str().unwrap());


    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-search=native={}",&blas_dir);

    println!("cargo:rerun-if-changed=src/external_libs/restmatr.f90");
    println!("cargo:rerun-if-changed={}/librestmatr.so", &external_dir.to_str().unwrap());

    Ok(())

}
