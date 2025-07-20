//extern crate dunce;
use std::path::PathBuf;
use std::{fs, env, process::Command};

fn main() -> std::io::Result<()> {

    // the lib directory to store librestmatr.so
    let external_dir = if let Ok(target_dir) = env::var("REST_EXT_DIR") {
        PathBuf::from(target_dir)
    } else {PathBuf::from(".".to_string())};

    if ! external_dir.is_dir() {
        fs::create_dir(&external_dir)?
    };



    #[cfg(not(feature="intel-mkl"))] {
        let restmatr_file = format!("src/external_libs/restmatr.f90");
        let restmatr_libr = format!("{}/librestmatr.so",&external_dir.to_str().unwrap());
        let fortran_compiler = if let Ok(fortran_compiler) = env::var("REST_FORTRAN_COMPILER") {
            fortran_compiler
        } else {"gfortran".to_string()};
        let blas_dir = if let Ok(blas_dir) = env::var("REST_EXT_DIR") {
            PathBuf::from(blas_dir)
        } else {PathBuf::from(".".to_string())};
        let restmatr_link = format!("-L{}",&blas_dir.display());
        Command::new(fortran_compiler)
            .args(&["-shared", "-fpic", "-O2",&restmatr_file,"-o",&restmatr_libr,&restmatr_link, "-lopenblas"])
            .status().unwrap();
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-link-search=native={}",&blas_dir.display());
    }

    #[cfg(feature="intel-mkl")] {
        let restmatr_file = format!("src/external_libs/restmatr.f90");
        let restmatr_libr = format!("{}/librestmatr.so",&external_dir.to_str().unwrap());
        let blas_dir = if let Ok(blas_dir) = env::var("MKLROOT") {
            PathBuf::from(blas_dir)
        } else {panic!("MKLROOT not set for feature intel-mkl")};
        let mkl_link = format!("-L{}/lib", &blas_dir.display());
        let mkl_include = format!("-I{}/include", &blas_dir.display());
        let fortran_compiler = if let Ok(fortran_compiler) = env::var("REST_FORTRAN_COMPILER") {
            fortran_compiler
        } else {"gfortran".to_string()};
        Command::new(fortran_compiler)
            .args(&["-shared", "-fpic", "-O2",&restmatr_file,"-o",&restmatr_libr, "-m64", &mkl_include, &mkl_link, 
            "-Wl,--no-as-needed", "-lmkl_gf_lp64", "-lmkl_gnu_thread", "-lmkl_core", "-lgomp", "-lpthread", "-lm", "-ldl"])
            .status().unwrap();
        println!("cargo:rustc-link-lib=mkl_rt");
        println!("cargo:rustc-link-search=native={}/lib",&blas_dir.display());
    }


    println!("cargo:rustc-link-lib=restmatr");
    println!("cargo:rustc-link-search=native={}",&external_dir.to_str().unwrap());



    println!("cargo:rerun-if-changed=src/external_libs/restmatr.f90");
    println!("cargo:rerun-if-changed={}/librestmatr.so", &external_dir.to_str().unwrap());

    Ok(())

}
