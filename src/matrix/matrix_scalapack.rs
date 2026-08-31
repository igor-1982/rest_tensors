#[cfg(feature = "scalapack")]
use std::ffi::CString;

#[cfg(feature = "scalapack")]
use std::os::raw::c_char;

#[cfg(feature = "scalapack")]
#[link(name = "scalapack")]
unsafe extern "C" {
    // -------- utility routines --------
    pub fn numroc_(
        n: *const i32,
        nb: *const i32,
        iproc: *const i32,
        isrcproc: *const i32,
        nprocs: *const i32,
    ) -> i32;

    pub fn descinit_(
        desc: *mut i32,
        m: *const i32,
        n: *const i32,
        mb: *const i32,
        nb: *const i32,
        irsrc: *const i32,
        icsrc: *const i32,
        ictxt: *const i32,
        lld: *const i32,
        info: *mut i32,
    );

    pub fn pdgemm_(
        transa: *const u8,
        transb: *const u8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        ia: *const i32,
        ja: *const i32,
        desca: *const i32,
        b: *const f64,
        ib: *const i32,
        jb: *const i32,
        descb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ic: *const i32,
        jc: *const i32,
        descc: *const i32,
    );

    pub fn pdsygvx_(
        ibtype: *const i32,
        jobz: *const u8,
        range: *const u8,
        uplo: *const u8,
        n: *const i32,
        a: *mut f64,
        ia: *const i32,
        ja: *const i32,
        desca: *const i32,
        b: *mut f64,
        ib: *const i32,
        jb: *const i32,
        descb: *const i32,
        vl: *const f64,
        vu: *const f64,
        il: *const i32,
        iu: *const i32,
        abstol: *const f64,
        m: *mut i32,
        nz: *mut i32,
        w: *mut f64,
        orfac: *const f64,
        z: *mut f64,
        iz: *const i32,
        jz: *const i32,
        descz: *const i32,
        work: *mut f64,
        lwork: *const i32,
        iwork: *mut i32,
        liwork: *const i32,
        ifail: *mut i32,
        iclustr: *mut i32,
        gap: *mut f64,
        info: *mut i32,
    );

    // -------- distributed Cholesky and triangular solve --------
    pub fn pdpotrf_(
        uplo: *const u8,
        n: *const i32,
        a: *mut f64,
        ia: *const i32,
        ja: *const i32,
        desca: *const i32,
        info: *mut i32,
    );

    pub fn pdtrtrs_(
        uplo: *const u8,
        trans: *const u8,
        diag: *const u8,
        n: *const i32,
        nrhs: *const i32,
        a: *const f64,
        ia: *const i32,
        ja: *const i32,
        desca: *const i32,
        b: *mut f64,
        ib: *const i32,
        jb: *const i32,
        descb: *const i32,
        info: *mut i32,
    );

    // -------- BLACS routines --------
    pub fn Cblacs_get(icontxt: i32, what: i32, val: *mut i32);

    pub fn Cblacs_gridinit(icontxt: *mut i32, order: *const c_char, nprow: i32, npcol: i32);

    pub fn Cblacs_gridinfo(
        icontxt: i32,
        nprow: *mut i32,
        npcol: *mut i32,
        myrow: *mut i32,
        mycol: *mut i32,
    );

    pub fn Cblacs_gridexit(icontxt: i32);
}

#[cfg(feature = "scalapack")]
pub struct CblacsGrid {
    pub ictxt: i32,
    pub nprow: i32,
    pub npcol: i32,
    pub myrow: i32,
    pub mycol: i32,
}

#[cfg(feature = "scalapack")]
impl CblacsGrid {
    pub fn new(size: usize, order: &str) -> Self {
        let mut ictxt = 0;

        unsafe {
            Cblacs_get(-1, 0, &mut ictxt);
        }

        let (nprow, npcol) = best_grid_dims(size as i32);

        let order = CString::new(order)
            .expect("BLACS grid order string must not contain interior null bytes");

        unsafe {
            Cblacs_gridinit(&mut ictxt, order.as_ptr(), nprow, npcol);
        }

        let mut nprow_out = 0;
        let mut npcol_out = 0;
        let mut myrow = 0;
        let mut mycol = 0;

        unsafe {
            Cblacs_gridinfo(
                ictxt,
                &mut nprow_out,
                &mut npcol_out,
                &mut myrow,
                &mut mycol,
            );
        }
        Self {
            ictxt,
            nprow: nprow_out,
            npcol: npcol_out,
            myrow,
            mycol,
        }
    }

    #[inline]
    pub fn rank_coords(&self) -> (i32, i32) {
        (self.myrow, self.mycol)
    }

    #[inline]
    pub fn dims(&self) -> (i32, i32) {
        (self.nprow, self.npcol)
    }
}

#[cfg(feature = "scalapack")]
impl Drop for CblacsGrid {
    fn drop(&mut self) {
        unsafe {
            Cblacs_gridexit(self.ictxt);
        }
    }
}

#[cfg(feature = "scalapack")]
pub fn best_grid_dims(size: i32) -> (i32, i32) {
    if size <= 0 {
        return (0, 0);
    }

    let mut nprow = (size as f64).sqrt().floor() as i32;
    while nprow > 1 && size % nprow != 0 {
        nprow -= 1;
    }

    let npcol = size / nprow;
    (nprow, npcol)
}
