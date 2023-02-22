extern crate dunce;
//use std::{env, path::PathBuf};
use std::{env};

fn main() {

    let library_names = ["restmatr","openblas"];
    library_names.iter().for_each(|name| {
    //    println!("cargo:rustc-link-lib=static={}",*name);
        println!("cargo:rustc-link-lib={}",*name);
    });
    let library_path = [
        //dunce::canonicalize(root.join("external_libs")).unwrap(),
        dunce::canonicalize("/share/home/wenxinzy/export/REST/fdqc/external_libs").unwrap(),
        //dunce::canonicalize("/share/apps/lib/libcint/build").unwrap(),
        //dunce::canonicalize("/share/apps/rust/HDF5/lib").unwrap(),
        //dunce::canonicalize(root.join("src/dft/libxc")).unwrap(),
        dunce::canonicalize("/share/apps/rust/OpenBLAS-0.3.17").unwrap()
    ];
    library_path.iter().for_each(|path| {
        println!("cargo:rustc-link-search=native={}",env::join_paths(&[path]).unwrap().to_str().unwrap())
    });
}