// rest_tensors/src/pexsi.rs
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, improper_ctypes)]

mod sys {
    include!(concat!(env!("OUT_DIR"), "/pexsi_bindings.rs"));
}

pub use sys::*;

use std::mem::MaybeUninit;

impl PPEXSIOptions {
    /// Equivalent to `PPEXSISetDefaultOptions(&mut zeroed)`.
    pub fn with_defaults() -> Self {
        let mut o: Self = unsafe { MaybeUninit::zeroed().assume_init() };
        unsafe { PPEXSISetDefaultOptions(&mut o); }
        o
    }
}

/// RAII wrapper around `PPEXSIPlan`.
pub struct Plan {
    inner: sys::PPEXSIPlan,
}

impl Plan {
    /// `comm` should be the size the caller wants; `num_proc_row * num_proc_col`
    /// must divide the communicator size.
    pub fn new(
        comm: mpi::ffi::MPI_Comm,
        num_proc_row: i32,
        num_proc_col: i32,
        output_file_index: i32,
    ) -> Result<Self, i32> {
        let mut info: i32 = 0;
        let inner = unsafe {
            // `mpi::ffi::MPI_Comm` and bindgen's `MPI_Comm` are both
            // `*mut ompi_communicator_t` at the ABI level.
            let comm_raw: sys::MPI_Comm = std::mem::transmute(comm);
            sys::PPEXSIPlanInitialize(comm_raw, num_proc_row, num_proc_col, output_file_index, &mut info)
        };
        if info != 0 { Err(info) } else { Ok(Self { inner }) }
    }

    pub fn raw(&self) -> sys::PPEXSIPlan { self.inner }
}

impl Drop for Plan {
    fn drop(&mut self) {
        let mut info: i32 = 0;
        unsafe { sys::PPEXSIPlanFinalize(self.inner, &mut info); }
        // PEXSI has no way to surface this during unwind; intentionally ignored.
    }
}

/// Number of MPI ranks expected by a `num_proc_row × num_proc_col` factorization.
pub fn ranks_per_pole(num_proc_row: i32, num_proc_col: i32) -> i32 {
    num_proc_row * num_proc_col
}