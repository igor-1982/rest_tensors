//extern crate dunce;
use std::path::PathBuf;
use std::{fs, env, process::Command};

#[cfg(feature = "pexsi")]
use std::collections::HashSet;

fn main() -> std::io::Result<()> {

    // the lib directory to store librestmatr.so
    let external_dir = if let Ok(target_dir) = env::var("REST_EXT_DIR") {
        PathBuf::from(target_dir)
    } else {PathBuf::from(".".to_string())};

    if ! external_dir.is_dir() {
        fs::create_dir(&external_dir)?
    };

    let fc_env_candidate = ["REST_FORTRAN_COMPILER", "F90", "FC"];
    let mut fortran_compiler = String::new();

    // iter over possible fortran compiler environment variables
    for fc_env in fc_env_candidate.iter() {
        if let Ok(fc) = env::var(fc_env) {
            fortran_compiler = fc;
            println!("Using Fortran compiler from env var {}: {}", fc_env, fortran_compiler);
            break;
        }
    }
    // if no fortran compiler env var is set, fallback to gfortran
    if fortran_compiler.is_empty() {
        fortran_compiler = "gfortran".to_string();
        println!("No Fortran compiler environment variable set, falling back to default: {}", fortran_compiler);
    }
    let cross_win = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "windows";
    let restmatr_lib;
    #[cfg(target_os = "linux")] {
        restmatr_lib = if ! cross_win {
            "librestmatr.so"
        } else {
            "librestmatr.dll"
        };
    }
    #[cfg(target_os = "macos")] {
        restmatr_lib = "librestmatr.dylib";
    }
    #[cfg(target_os = "windows")] {
        restmatr_lib = "librestmatr.dll";
    }
    
    let restmatr_file = format!("src/external_libs/restmatr.f90");
    let restmatr_libr = format!("{}/{}",&external_dir.to_str().unwrap(), restmatr_lib);

    #[cfg(not(feature="intel-mkl"))] {
        let blas_dir = if let Ok(blas_dir) = env::var("REST_EXT_DIR") {
            PathBuf::from(blas_dir)
        } else {PathBuf::from(".".to_string())};
        let restmatr_link = format!("-L{}",&blas_dir.display());
        
        let mut compile_command = Command::new(&fortran_compiler);
        compile_command.args(&["-shared", "-fpic", "-O2",&restmatr_file,"-o",&restmatr_libr,&restmatr_link, "-lopenblas"]);
        // for cross win or native win
        let mut compile_command2 = Command::new(&fortran_compiler);
        compile_command2.args(&["-shared", "-O2",&restmatr_file,"-o",&restmatr_libr,&restmatr_link, "-lopenblas"]);
        #[cfg(target_os = "windows")]
        let compile_result = compile_command2.status();
        
        #[cfg(not(target_os = "windows"))]
        let compile_result = if !cross_win {
            compile_command.status()
        } else {
            compile_command2.status()
        };
        
        compile_result.expect("Failed to compile restmatr");
        
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-link-search=native={}",&blas_dir.display());
    }

    #[cfg(feature="intel-mkl")] {
        let blas_dir = if let Ok(blas_dir) = env::var("MKLROOT") {
            PathBuf::from(blas_dir)
        } else {panic!("MKLROOT not set for feature intel-mkl")};
        let mkl_link = format!("-L{}/lib", &blas_dir.display());
        let mkl_include = format!("-I{}/include", &blas_dir.display());

        let mut compile_command = Command::new(&fortran_compiler);
        compile_command.args(&["-shared", "-fpic", "-O2",&restmatr_file,"-o",&restmatr_libr, "-m64", &mkl_include, &mkl_link, 
            "-Wl,--no-as-needed", "-lmkl_gf_lp64", "-lmkl_gnu_thread", "-lmkl_core", "-lgomp", "-lpthread", "-lm", "-ldl"]);
        let mut compile_command2 = Command::new(&fortran_compiler);
        compile_command2.args(&["-shared", "-O2",&restmatr_file,"-o",&restmatr_libr, "-m64", &mkl_include, &mkl_link, 
            "-lmkl_rt"]);
        
        #[cfg(target_os = "windows")]
        let compile_result = compile_command2.status();
        
        #[cfg(not(target_os = "windows"))]
        let compile_result = if !cross_win {
            compile_command.status()
        } else {
            compile_command2.status()
        };
        
        compile_result.expect("Failed to compile restmatr with MKL");
        println!("cargo:rustc-link-lib=mkl_rt");
        println!("cargo:rustc-link-search=native={}/lib",&blas_dir.display());
    }


    #[cfg(not(target_os = "windows"))] {
        if !cross_win {
            println!("cargo:rustc-link-lib=restmatr");
        } else {
            println!("cargo:rustc-link-arg={}/{}", external_dir.to_str().unwrap(), restmatr_lib);
            println!("cargo:rustc-link-lib=gomp");
        }
    }
    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-arg={}/{}", external_dir.to_str().unwrap(), restmatr_lib);
    println!("cargo:rustc-link-search=native={}",&external_dir.to_str().unwrap());

    #[cfg(feature = "scalapack")] {
    // `libscalapack.so` is expected in REST_EXT_DIR (already added to the link search
    // path above; e.g. a symlink to the apt package libscalapack-openmpi-dev), or in
    // the standard system path. Optionally, a conda environment providing
    // libscalapack can be selected via CONDA_PREFIX.
    if let Ok(prefix) = env::var("CONDA_PREFIX") {
        println!(
            "cargo:rustc-link-search=native={}/lib",
            PathBuf::from(&prefix).display()
        );
    }
    println!("cargo:rustc-link-lib=scalapack");
}
    println!("cargo:rerun-if-changed=src/external_libs/restmatr.f90");
    // println!("cargo:rerun-if-changed={}/{}", &external_dir.to_str().unwrap(), restmatr_lib);
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=omp");
    
    #[cfg(feature = "pexsi")]
    build_pexsi();

    Ok(())

}

// ===========================================================================
// PEXSI: bindgen bindings + static link stack.
//
// Enabled via `--features pexsi` (declared in Cargo.toml).  Assumes the
// dependencies are reachable through $REST_EXT_DIR (the spack-style view
// maintained by genview.sh), with per-package $PEXSI_DIR / $SUPERLU_DIST_DIR
// / $PARMETIS_DIR / $MPI_DIR honoured as overrides.
// ===========================================================================
#[cfg(feature = "pexsi")]
fn build_pexsi() {
    for var in [
        "REST_EXT_DIR", "PEXSI_DIR", "SUPERLU_DIST_DIR",
        "PARMETIS_DIR", "MPI_DIR", "LIBRARY_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={}", var);
    }
    println!("cargo:rerun-if-changed=build.rs");

    let view = env::var("REST_EXT_DIR")
        .map(PathBuf::from)
        .expect("REST_EXT_DIR must be set when the `pexsi` feature is enabled");

    // Header: prefer $PEXSI_DIR/include, fall back to the view's include/.
    let header = match env::var("PEXSI_DIR") {
        Ok(p) => PathBuf::from(p).join("include").join("c_pexsi_interface.h"),
        Err(_) => view.join("include").join("c_pexsi_interface.h"),
    };
    if !header.is_file() {
        panic!(
            "PEXSI header not found at {:?}.  Set PEXSI_DIR or make sure \
             $REST_EXT_DIR/include/c_pexsi_interface.h exists.",
            header
        );
    }

    // ---- link search paths --------------------------------------------
    let mut dirs: Vec<PathBuf> = Vec::new();
    dirs.push(view.clone());
    dirs.push(view.join("lib"));
    dirs.push(view.join("lib64"));

    for name in ["PEXSI_DIR", "SUPERLU_DIST_DIR", "PARMETIS_DIR", "MPI_DIR"] {
        if let Ok(p) = env::var(name) {
            let p = PathBuf::from(p);
            dirs.push(p.join("lib"));
            dirs.push(p.join("lib64"));
        }
    }
    if let Some(lp) = env::var_os("LIBRARY_PATH") {
        for p in env::split_paths(&lp) {
            dirs.push(p);
        }
    }

    // Dedupe preserving order (view wins).
    let mut seen = HashSet::new();
    dirs.retain(|d| seen.insert(d.clone()));

    for d in &dirs {
        if d.is_dir() {
            println!("cargo:rustc-link-search=native={}", d.display());
        }
    }

    // ---- static stack: dependency order, in a linker group ------------
    // ParMETIS ↔ METIS ↔ GKlib have cyclic references; --start-group /
    // --end-group lets ld iterate to a fixed point.
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    for lib in ["pexsi", "superlu_dist", "parmetis", "metis", "GKlib"] {
        println!("cargo:rustc-link-lib=static={}", lib);
    }
    println!("cargo:rustc-link-arg=-Wl,--end-group");

    // ---- shared libs ---------------------------------------------------
    println!("cargo:rustc-link-lib=dylib=mpi");
    for l in ["mpi_mpifh", "mpi_usempif08", "mpi_usempi_ignore_tkr"] {
        println!("cargo:rustc-link-lib=dylib={}", l);
    }
    for l in ["openblas", "gomp", "stdc++", "gfortran", "m"] {
        println!("cargo:rustc-link-lib=dylib={}", l);
    }

    // ---- bindgen -------------------------------------------------------
    let mut b = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .allowlist_function("PPEXSI.*")
        .allowlist_type("PPEXSI.*")
        .allowlist_var("PPEXSI.*")
        .allowlist_type("MPI_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    for name in ["PEXSI_DIR", "SUPERLU_DIST_DIR", "PARMETIS_DIR", "MPI_DIR"] {
        if let Ok(p) = env::var(name) {
            let inc = PathBuf::from(p).join("include");
            if inc.is_dir() {
                b = b.clang_arg(format!("-I{}", inc.display()));
            }
        }
    }
    let view_inc = view.join("include");
    if view_inc.is_dir() {
        b = b.clang_arg(format!("-I{}", view_inc.display()));
    }

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is always set by cargo"));
    b.generate()
        .expect("bindgen failed on c_pexsi_interface.h")
        .write_to_file(out.join("pexsi_bindings.rs"))
        .expect("failed to write pexsi_bindings.rs");
}