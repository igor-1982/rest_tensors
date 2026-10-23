// rest_tensors/src/matrix/matrix_pexsi.rs
//
// Hand-written FFI for PEXSI v2.0.0. Mirrors the style of matrix_scalapack.rs:
// no bindgen, no build-time code generation. Struct layout, index base,
// and by-value-vs-by-pointer conventions are all visible here.

use std::os::raw::c_int;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// `typedef intptr_t PPEXSIPlan;` in c_pexsi_interface.h line 174.
/// `isize` is the Rust spelling of `intptr_t`.
pub type PPEXSIPlanRaw = isize;

// ---------------------------------------------------------------------------
// PPEXSIOptions: exact #[repr(C)] mirror of the C header, lines 180-348.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_snake_case)]
pub struct PPEXSIOptions {
    pub spin:                      f64,
    pub temperature:               f64,
    pub gap:                       f64,
    pub deltaE:                    f64,
    pub numPole:                   c_int,
    pub isInertiaCount:            c_int,
    pub maxPEXSIIter:              c_int,
    pub muMin0:                    f64,
    pub muMax0:                    f64,
    pub mu0:                       f64,
    pub muInertiaTolerance:        f64,
    pub muInertiaExpansion:        f64,
    pub muPEXSISafeGuard:          f64,
    pub numElectronPEXSITolerance: f64,
    pub matrixType:                c_int,
    pub isSymbolicFactorize:       c_int,
    pub isConstructCommPattern:    c_int,
    pub solver:                    c_int,
    pub symmetricStorage:          c_int,
    pub ordering:                  c_int,
    pub rowOrdering:               c_int,
    pub npSymbFact:                c_int,
    pub symmetric:                 c_int,
    pub transpose:                 c_int,
    pub method:                    c_int,
    pub nPoints:                   c_int,
    pub verbosity:                 c_int,
    pub iFLAG:                     c_int,
}

// Compile-time guard: if this fails, the field order above is wrong and
// every PEXSI call is UB. 160 was confirmed against the C side earlier.
const _: () = assert!(std::mem::size_of::<PPEXSIOptions>() == 160);

impl PPEXSIOptions {
    /// Equivalent to `PPEXSISetDefaultOptions(&mut zeroed())`.
    pub fn defaults() -> Self {
        let mut o: Self = unsafe { std::mem::zeroed() };
        unsafe { PPEXSISetDefaultOptions(&mut o); }
        o
    }
}

// ---------------------------------------------------------------------------
// Raw FFI declarations
// ---------------------------------------------------------------------------

unsafe extern "C" {
    pub fn PPEXSIPlanInitialize(
        comm: mpi::ffi::MPI_Comm,
        numProcRow: c_int,
        numProcCol: c_int,
        outputFileIndex: c_int,
        info: *mut c_int,
    ) -> PPEXSIPlanRaw;

    pub fn PPEXSIPlanFinalize(plan: PPEXSIPlanRaw, info: *mut c_int);

    pub fn PPEXSISetDefaultOptions(options: *mut PPEXSIOptions);

    pub fn PPEXSILoadRealHSMatrix(
        plan: PPEXSIPlanRaw,
        options: PPEXSIOptions,             // by value
        nrows: c_int,
        nnz: c_int,
        nnzLocal: c_int,
        numColLocal: c_int,
        colptrLocal: *mut c_int,
        rowindLocal: *mut c_int,
        HnzvalLocal: *mut f64,
        isSIdentity: c_int,
        SnzvalLocal: *mut f64,
        info: *mut c_int,
    );

    pub fn PPEXSISymbolicFactorizeRealSymmetricMatrix(
        plan: PPEXSIPlanRaw,
        options: PPEXSIOptions,             // by value
        info: *mut c_int,
    );

    pub fn PPEXSIDFTDriver2(
        plan: PPEXSIPlanRaw,
        options: *mut PPEXSIOptions,        // by pointer — the only one
        numElectronExact: f64,
        muPEXSI: *mut f64,
        numElectronPEXSI: *mut f64,
        numTotalInertiaIter: *mut c_int,
        info: *mut c_int,
    );

    pub fn PPEXSIRetrieveRealDM(
        plan: PPEXSIPlanRaw,
        DMnzvalLocal: *mut f64,
        totalEnergyH: *mut f64,
        info: *mut c_int,
    );

    pub fn PPEXSISelInvRealSymmetricMatrix(
            plan: PPEXSIPlanRaw,
            options: PPEXSIOptions,
            AnzvalLocal: *mut f64,
            AinvnzvalLocal: *mut f64,
            info: *mut c_int,
    );
}

// ---------------------------------------------------------------------------
// RAII wrapper for the plan handle
// ---------------------------------------------------------------------------

pub struct PexsiPlan {
    raw: PPEXSIPlanRaw,
    // Backing storage for the current H/S load. PEXSI v2.0.0's
    // LoadRealHSMatrix does NOT copy these arrays: on the root rank it
    // wraps the caller's pointers as views (NumVec(..., owndata_ = false)).
    // They must all stay alive until PPEXSIPlanFinalize, or the inertia
    // counting / Fermi operator will read freed memory (SIGSEGV).
    _colptr: Vec<c_int>,
    _rowind: Vec<c_int>,
    _hnzval: Vec<f64>,
    _snzval: Vec<f64>,
}

impl PexsiPlan {
    /// `mpi_comm` should be the raw `MPI_Comm` from rsmpi (`world.as_raw()`).
    /// `num_proc_row * num_proc_col` must divide the communicator's size.
    /// For the 1D column-block distribution REST uses, `num_proc_row = 1`.
    pub fn new(
        mpi_comm: mpi::ffi::MPI_Comm,
        num_proc_row: i32,
        num_proc_col: i32,
        output_file_index: i32,
    ) -> Result<Self, String> {
        let mut info: c_int = 0;
        let raw = unsafe {
            PPEXSIPlanInitialize(
                mpi_comm,
                num_proc_row as c_int,
                num_proc_col as c_int,
                output_file_index as c_int,
                &mut info,
            )
        };
        pexsi_check(info, "PPEXSIPlanInitialize")?;
        Ok(Self { 
            raw: raw, 
            _colptr: Vec::new(), 
            _rowind: Vec::new(), 
            _hnzval: Vec::new(),
            _snzval: Vec::new(),
        })
    }

    #[inline]
    pub fn raw(&self) -> PPEXSIPlanRaw { self.raw }
}

impl Drop for PexsiPlan {
    fn drop(&mut self) {
        let mut info: c_int = 0;
        unsafe { PPEXSIPlanFinalize(self.raw, &mut info); }
        // Deliberately not unwrapping: Drop must not panic.
    }
}

// ---------------------------------------------------------------------------
// Safe wrappers around each FFI call
// ---------------------------------------------------------------------------

/// Returns Ok(()) when info == 0, Err(msg) otherwise.
fn pexsi_check(info: c_int, ctx: &str) -> Result<(), String> {
    if info == 0 { Ok(()) } else { Err(format!("{ctx} failed: info = {info}")) }
}

/// Load H and S into the plan. Both matrices must share the same
/// `SparsePattern` (i.e. identical colptr/rowind).
pub fn pexsi_load_hs<T>(
    plan: &mut PexsiPlan,
    opts: &PPEXSIOptions,
    h: &crate::matrix::distsparsematrixfull::DistSparseMatrixFull<T>,
    s: &crate::matrix::distsparsematrixfull::DistSparseMatrixFull<T>,
) -> Result<(), String>
where
    T: Copy + Into<f64>,
{
    use std::sync::Arc;

    // Verify same pattern.
    let hp: &Arc<crate::matrix::distsparsematrixfull::SparsePattern> = &h.pattern;
    let sp = &s.pattern;
    if !Arc::ptr_eq(hp, sp) {
        // Not necessarily a bug — but the FFI requires the same colptr/rowind.
        // Fall back to comparing the arrays.
        if hp.local_colptr != sp.local_colptr || hp.local_rowind != sp.local_rowind {
            return Err("pexsi_load_hs: H and S patterns differ".into());
        }
    }

    let nrows       = h.nrows()         as c_int;
    let nnz_global  = h.pattern.global_nnz as c_int;
    let nnz_local   = h.nnz_local()     as c_int;
    let ncol_local  = h.num_col_local() as c_int;

    if nnz_global > c_int::MAX || nnz_local > c_int::MAX {
        return Err("pexsi_load_hs: nnz exceeds c_int".into());
    }

    // Refill the plan-owned buffers in place. PEXSI v2.0.0 wraps these
    // pointers (owndata_ = false) rather than copying, so the colptr,
    // rowind AND the H/S values must all live on `plan` for the lifetime
    // of the handle.
    plan._colptr.clear();
    plan._colptr.extend(h.pattern.local_colptr.iter().map(|&v| v as c_int));
    plan._rowind.clear();
    plan._rowind.extend(h.pattern.local_rowind.iter().map(|&v| v as c_int));
    plan._hnzval.clear();
    plan._hnzval.extend(h.nzval.iter().map(|&v| v.into()));
    plan._snzval.clear();
    plan._snzval.extend(s.nzval.iter().map(|&v| v.into()));

    let mut info: c_int = 0;
    unsafe {
        PPEXSILoadRealHSMatrix(
            plan.raw(),
            *opts,
            nrows,
            nnz_global,
            nnz_local,
            ncol_local,
            plan._colptr.as_mut_ptr(),
            plan._rowind.as_mut_ptr(),
            plan._hnzval.as_mut_ptr(),
            0,                 // isSIdentity = 0: S is provided explicitly
            plan._snzval.as_mut_ptr(),
            &mut info,
        );
    }
    pexsi_check(info, "PPEXSILoadRealHSMatrix")
}

pub fn pexsi_symbolic_factorize(
    plan: &PexsiPlan,
    opts: &PPEXSIOptions,
) -> Result<(), String> {
    let mut info: c_int = 0;
    unsafe {
        PPEXSISymbolicFactorizeRealSymmetricMatrix(plan.raw(), *opts, &mut info);
    }
    pexsi_check(info, "PPEXSISymbolicFactorizeRealSymmetricMatrix")
}

pub struct DftDriverResult {
    pub mu:               f64,
    pub num_electrons:    f64,
    pub num_inertia_iter: i32,
}

/// Runs the PEXSI DFT driver: internal inertia counting + Newton for mu,
/// numeric factorization, and selected inversion. Writes back into `opts`
/// (the C API takes `*mut PPEXSIOptions` here).
pub fn pexsi_dft_driver2(
    plan: &PexsiPlan,
    opts: &mut PPEXSIOptions,
    num_electron_exact: f64,
) -> Result<DftDriverResult, String> {
    let mut mu            = 0.0_f64;
    let mut ne            = 0.0_f64;
    let mut n_inertia     = 0 as c_int;
    let mut info: c_int   = 0;
    unsafe {
        PPEXSIDFTDriver2(
            plan.raw(),
            opts as *mut PPEXSIOptions,
            num_electron_exact,
            &mut mu,
            &mut ne,
            &mut n_inertia,
            &mut info,
        );
    }
    pexsi_check(info, "PPEXSIDFTDriver2")?;
    Ok(DftDriverResult {
        mu,
        num_electrons:    ne,
        num_inertia_iter: n_inertia as i32,
    })
}

/// Retrieve the density matrix. The buffer must have length equal to PEXSI's
/// internal `nnzLocal` for the DM. For a first integration, allocating
/// `h.nnz_local()` and growing on failure is acceptable; when PEXSI reports
/// it needs more, the buffer size returned by symbolic factorization should
/// be used (PEXSI v2.0.0 does not expose a direct query, so this is a
/// remaining sharp edge).
pub fn pexsi_retrieve_dm(
    plan: &PexsiPlan,
    dm_nzval: &mut [f64],
) -> Result<f64, String> {
    let mut band_energy: f64 = 0.0;
    let mut info: c_int      = 0;
    unsafe {
        PPEXSIRetrieveRealDM(
            plan.raw(),
            dm_nzval.as_mut_ptr(),
            &mut band_energy,
            &mut info,
        );
    }
    pexsi_check(info, "PPEXSIRetrieveRealDM")?;
    Ok(band_energy)
}

pub fn pexsi_selinv_real_symmetric<T>(
    plan: &PexsiPlan,
    opts: &PPEXSIOptions,
    a: &crate::matrix::distsparsematrixfull::DistSparseMatrixFull<T>,
    ainv_nzval: &mut [f64],
) -> Result<(), String>
where
    T: Copy + Into<f64>,
{
    assert_eq!(ainv_nzval.len(), a.nnz_local());

    let mut avals: Vec<f64> = a.nzval.iter().map(|&v| v.into()).collect();
    let mut info: c_int = 0;
    unsafe {
        PPEXSISelInvRealSymmetricMatrix(
            plan.raw(),
            *opts,
            avals.as_mut_ptr(),
            ainv_nzval.as_mut_ptr(),
            &mut info,
        );
    }
    pexsi_check(info, "PPEXSISelInvRealSymmetricMatrix")
}