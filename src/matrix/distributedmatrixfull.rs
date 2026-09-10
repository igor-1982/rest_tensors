use crate::matrix::{
    BasicMatrix, BasicMatrixOpt, MatFormat, MathMatrix, MatrixFull, ParMathMatrix,
};
use crate::matrix::matrixupper::*;
use crate::matrix::matrix_blas_lapack::{_dgemm_full, _hamiltonian_fast_solver, _get_sqrt_and_inv_sqrt};
#[cfg(feature = "scalapack")]
use crate::matrix::matrix_scalapack::*;
#[cfg(feature = "mpi")]
use mpi::collective::SystemOperation;
#[cfg(feature = "mpi")]
use mpi::environment::Universe;
#[cfg(feature = "mpi")]
use mpi::request::WaitGuard;
#[cfg(feature = "mpi")]
use mpi::topology::{Rank, SimpleCommunicator};
#[cfg(feature = "mpi")]
use mpi::traits::*;
#[cfg(feature = "mpi")]
use mpi::datatype::*;

use rand::Rng;


#[cfg(feature = "scalapack")]
#[derive(Clone, Debug, PartialEq)]
pub struct DistributedMatrixFull<T> {
    /// the number of LOCAL row and column, column major
    pub size: [usize; 2],
    /// LOCAL indicing is defined to facilitate the element nevigation, in particular for 3-rank tensors, RIFull
    pub indicing: [usize; 2],
    /// the data stored in the [`Vec`](Vec) struct
    pub data: Vec<T>,
    /// ScaLAPACK descriptor array (9 integers)
    pub desc: [i32; 9],
    /// Leading dimension (lld) used in desc
    pub lld: i32,
}

#[cfg(feature = "scalapack")]
#[cfg(feature = "mpi")]
#[derive(Clone, Debug)]
pub struct GatherDistributedMsg<T> where T: Copy + Clone + Default + Send {
    pub myrow: i32,
    pub mycol: i32,
    pub local_rows: i32,
    pub local_cols: i32,
    pub lld: i32,
    pub data: Vec<T>,
}

#[cfg(feature = "scalapack")]
impl<'a, T> BasicMatrix<'a, T> for DistributedMatrixFull<T> {
    #[inline]
    /// `matr_a.size()' return &matr_a.size;
    fn size(&self) -> [usize; 2] {
        self.size.clone()
    }
    #[inline]
    /// `matr_a.indicing()' return &matr_a.indicing;
    fn indicing(&self) -> [usize; 2] {
        self.indicing.clone()
    }
    #[inline]
    fn data_ref(&self) -> Option<&[T]> {
        Some(&self.data[..])
    }
    #[inline]
    fn data_ref_mut(&mut self) -> Option<&mut [T]> {
        Some(&mut self.data[..])
    }

    fn is_matr(&self) -> bool {
        self.size().len() == 2 && self.indicing().len() == 2
    }

    fn is_contiguous(&self) -> bool {
        false
    }
}
#[cfg(feature = "scalapack")]
impl<'a, T> BasicMatrixOpt<'a, T> for DistributedMatrixFull<T> where T: Copy + Clone {}

//impl<'a, T> MathMatrix<'a, T> for DistributedMatrixFull<T> where T: Copy + Clone {}

//impl<'a, T> ParMathMatrix<'a, T> for DistributedMatrixFull<T> where T: Copy + Clone + Send + Sync {}
//
#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
impl<T> DistributedMatrixFull<T>
where
    T: Copy + Clone + Default + Send + Buffer + 'static + mpi::datatype::Equivalence,
{
    pub fn new(
        grid: &CblacsGrid,
        global_rows: i32,
        global_cols: i32,
        block_rows: i32,
        block_cols: i32,
        irsrc: i32,
        icsrc: i32,
        new_default: T,
    ) -> Self {
        assert!(block_rows < global_rows);
        assert!(block_cols < global_cols);
        let local_rows =
            unsafe { numroc_(&global_rows, &block_rows, &grid.myrow, &irsrc, &grid.nprow) };
        let local_cols =
            unsafe { numroc_(&global_cols, &block_cols, &grid.mycol, &icsrc, &grid.npcol) };
        let lld = std::cmp::max(1, local_rows);
        let mut desc = [0; 9];
        let mut info = 0;
        unsafe {
            descinit_(
                desc.as_mut_ptr(),
                &global_rows,
                &global_cols,
                &block_rows,
                &block_cols,
                &irsrc,
                &icsrc,
                &grid.ictxt,
                &lld,
                &mut info,
            );
        }
        if info != 0 {
            panic!("Failed with descinit with info{}", info)
        } else {
            //println!("grid:{},{}, desc:{:?}", grid.myrow, grid.mycol, desc);
        };
        let data = vec![new_default.clone(); (local_rows * local_cols) as usize];
        let size = [local_rows as usize, local_cols as usize];
        let mut indicing = [0usize; 2];
        let mut len = size
            .iter()
            .zip(indicing.iter_mut())
            .fold(1usize, |len, (di, ii)| {
                *ii = len;
                len * di
            });

        Self {
            size,
            indicing,
            data,
            desc,
            lld,
        }
    }

    pub fn from_matrixfull(
        grid: &CblacsGrid,
        mat: &MatrixFull<T>,
        block_rows: i32,
        block_cols: i32,
        irsrc: i32,
        icsrc: i32,
    ) -> DistributedMatrixFull<T>
    {
        let global_rows = mat.size()[0] as i32;
        let global_cols = mat.size()[1] as i32;
        assert!(block_rows < global_rows);
        assert!(block_cols < global_cols);

        let nprow = grid.nprow;
        let npcol = grid.npcol;
        let myrow = grid.myrow;
        let mycol = grid.mycol;

        let mut result = Self::new(
            grid,
            mat.size()[0] as i32,
            mat.size()[1] as i32,
            block_rows,
            block_cols,
            irsrc,
            icsrc,
            T::default(),
        );
        let lld = result.lld;
        let global_data_ref = mat.data_ref().expect("global matrix should not be empty");
        assert_eq!(global_data_ref.len() as i32, global_rows * global_cols);
        let local_data_ref = result.data_ref_mut().unwrap();

        for i in 0..global_rows {
            let block_i = i / block_rows;
            let proc_i = (irsrc + block_i) % nprow;
            if proc_i != myrow {
                continue;
            }
            let local_i = (block_i / nprow) * block_rows + (i % block_rows);

            for j in 0..global_cols {
                let block_j = j / block_cols;
                let proc_j = (icsrc + block_j) % npcol;
                if proc_j != mycol {
                    continue;
                }
                let local_j = (block_j / npcol) * block_cols + (j % block_cols);
                let local_idx = (local_j * lld + local_i) as usize;
                let global_idx = (j * global_rows + i) as usize;
                assert!(local_idx < local_data_ref.len());
                assert!(global_idx < global_data_ref.len());
                local_data_ref[local_idx] = global_data_ref[global_idx].clone();
            }
        }
        result
    }

    pub fn from_matrixupper(
        grid: &CblacsGrid,
        mat: &MatrixUpper<T>,
        block_rows: i32,
        block_cols: i32,
        irsrc: i32,
        icsrc: i32,
    ) -> DistributedMatrixFull<T>
    {
        let mat_full = mat.to_matrixfull().unwrap();
        let result = Self::from_matrixfull(grid, &mat_full, block_rows, block_cols, irsrc, icsrc);
        result
    }

    pub fn gather_to_root(&self, root: i32, grid:& CblacsGrid, world: &SimpleCommunicator) -> Vec<Option<GatherDistributedMsg<T>>>
        where [T]: mpi::datatype::Buffer {
        let rank = world.rank();
        let size = world.size();
        let mut msgs = Vec::with_capacity(size as usize);

        if rank == root {
            let own_msg = GatherDistributedMsg {
                myrow: grid.myrow,
                mycol: grid.mycol,
                local_rows: self.size()[0].try_into().unwrap(),
                local_cols: self.size()[1].try_into().unwrap(),
                lld: self.lld,
                data: self.data.clone(),
            };
            msgs.push(Some(own_msg));

            for r in 0..size {
                if r == root {continue;}
                let (metadata_recv, _status) = world.process_at_rank(r).receive_vec::<i32>();
                let metadata: [i32; 6] = metadata_recv.try_into().unwrap();
                let myrow_recv = metadata[0];
                let mycol_recv = metadata[1];
                let local_rows_recv = metadata[2];
                let local_cols_recv = metadata[3];
                let lld_recv = metadata[4];
                let data_len = metadata[5] as usize;

                let (data_to_recv, _status) = world.process_at_rank(r).receive_vec::<T>();
                assert_eq!(data_to_recv.len(), data_len);

                let msg = GatherDistributedMsg {
                    myrow: myrow_recv,
                    mycol: mycol_recv,
                    local_rows: local_rows_recv,
                    local_cols: local_cols_recv,
                    lld: lld_recv,
                    data: data_to_recv,
                };
                msgs.push(Some(msg));
            }

            if root == 0 {return msgs;}
            let mut ordered = Vec::with_capacity(size as usize);
            ordered.resize(size as usize, None);
            ordered[0] = msgs[0].take();
            for (i, msg) in msgs.into_iter().skip(1).enumerate() {
                let rank_idx = i + 1;
                ordered[rank_idx] = msg;
            }
            ordered
        }
        else {
            let metadata = [
                grid.myrow,
                grid.mycol,
                self.size()[0].try_into().unwrap(),
                self.size()[1].try_into().unwrap(),
                self.lld,
                self.data.len() as i32,
            ];

            world.process_at_rank(0).send(&metadata[..]);
            world.process_at_rank(0).send(&self.data[..]);
            Vec::new()
        }
    }
}

#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn reconstruct_from_gathered<T>(
    msgs: &[Option<GatherDistributedMsg<T>>],
    global_rows: i32,
    global_cols: i32,
    block_rows: i32,
    block_cols: i32,
    nprow: i32,
    npcol: i32,
    irsrc: i32,
    icsrc: i32) -> MatrixFull<T>
    where T: Copy + Clone + Default + Send {
        let mut global_mat = MatrixFull::new([global_rows as usize, global_cols as usize], T::default());

        for (rank, opt) in msgs.iter().enumerate() {
            let msg = opt.as_ref().expect("Message missing for rank {rank}");
            let myrow = msg.myrow;
            let mycol = msg.mycol;
            let lld = msg.lld;
            let data = &msg.data;

            for idx in (0..data.len()) {
                let local_i = (idx % lld as usize) as i32;
                let local_j = (idx / lld as usize) as i32;
                let block_i = (local_i / block_rows) * nprow + ((myrow - irsrc + nprow) % nprow);
                let global_i = block_i * block_rows + (local_i % block_rows);
                let block_j = (local_j / block_cols) * npcol + ((mycol - icsrc + npcol) % npcol);
                let global_j = block_j * block_cols + (local_j % block_cols);
                let global_idx = (global_i + global_j * global_rows) as usize;
                global_mat.data[global_idx] = data[idx];
            }
        }
        global_mat
    }


#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
fn check_pdgemm_shape(
    transa: char,
    transb: char,
    a: &DistributedMatrixFull<f64>,
    ia: i32,
    ja: i32,
    b: &DistributedMatrixFull<f64>,
    ib: i32,
    jb: i32,
    c: &DistributedMatrixFull<f64>,
    ic: i32,
    jc: i32,
) {
    // Extract global dimensions from descriptors
    let a_rows = if transa == 'N' { a.desc[2] } else { a.desc[3] };
    let a_cols = if transa == 'N' { a.desc[3] } else { a.desc[2] };

    let b_rows = if transb == 'N' { b.desc[2] } else { b.desc[3] };
    let b_cols = if transb == 'N' { b.desc[3] } else { b.desc[2] };

    let c_rows = c.desc[2];
    let c_cols = c.desc[3];

    // Compatibility checks
    assert!(
        a_cols == b_rows,
        "pdgemm: op(A) columns ({}) must equal op(B) rows ({})",
        a_cols, b_rows
    );
    assert!(
        a_rows == c_rows,
        "pdgemm: op(A) rows ({}) must equal C rows ({})",
        a_rows, c_rows
    );
    assert!(
        b_cols == c_cols,
        "pdgemm: op(B) columns ({}) must equal C columns ({})",
        b_cols, c_cols
    );

    // Starting index bounds (1‑based)
    assert!(ia >= 1 && ia <= a.desc[2], "ia out of range (1..{})", a.desc[2]);
    assert!(ja >= 1 && ja <= a.desc[3], "ja out of range (1..{})", a.desc[3]);
    assert!(ib >= 1 && ib <= b.desc[2], "ib out of range (1..{})", b.desc[2]);
    assert!(jb >= 1 && jb <= b.desc[3], "jb out of range (1..{})", b.desc[3]);
    assert!(ic >= 1 && ic <= c.desc[2], "ic out of range (1..{})", c.desc[2]);
    assert!(jc >= 1 && jc <= c.desc[3], "jc out of range (1..{})", c.desc[3]);
}

#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pdgemm(
    transa: char,
    transb: char,
    alpha: f64,
    a: &DistributedMatrixFull<f64>,
    ia: i32, ja: i32,     // 1‑based indices
    b: &DistributedMatrixFull<f64>,
    ib: i32, jb: i32,
    beta: f64,
    c: &mut DistributedMatrixFull<f64>,
    ic: i32, jc: i32,
) {

    check_pdgemm_shape(transa, transb, a, ia, ja, b, ib, jb, c, ic, jc);
    // Extract global dimensions from descriptors (or from stored fields)
    let m = c.desc[2];   // global rows of C
    let n = c.desc[3];   // global cols of C
    let k = if transa == 'N' { a.desc[3] } else { a.desc[2] };

    // Convert char to bytes (Fortran expects single characters)
    let ta = transa as u8;
    let tb = transb as u8;

    unsafe {
        pdgemm_(
            &ta, &tb,
            &m, &n, &k,
            &alpha,
            a.data.as_ptr(), &ia, &ja, a.desc.as_ptr(),
            b.data.as_ptr(), &ib, &jb, b.desc.as_ptr(),
            &beta,
            c.data.as_mut_ptr(), &ic, &jc, c.desc.as_ptr(),
        );
    }
}

#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pdsygvx(
    grid: &CblacsGrid,
    ibtype: i32,
    jobz: char,
    range: char,
    uplo: char,
    a: &mut DistributedMatrixFull<f64>,
    ia: i32, ja: i32,
    b: &mut DistributedMatrixFull<f64>,
    ib: i32, jb: i32,
    vl: f64, vu: f64,
    il: i32, iu: i32,
    abstol: f64,
    orfac: f64,
    z: &mut DistributedMatrixFull<f64>,
    iz: i32, jz: i32,
) -> (Vec<f64>, i32, i32) // returns (eigenvalues, m, nz)
{
    let n = a.desc[2]; // global size
    let (nprocs, myrow, mycol) = (grid.nprow, grid.myrow, grid.mycol);

    // Workspace query
    let mut m = 0;
    let mut nz = 0;
    let mut w = vec![0.0; n as usize];
    let mut info = 0;
    let mut lwork = -1;
    let mut liwork = -1;
    let mut work = vec![0.0; 1];
    let mut iwork = vec![0; 1];
    let mut ifail = vec![0; n as usize];
    let mut iclustr = vec![0; 2 * (n as usize + 1)];
    let mut gap = vec![0.0; n as usize];

    unsafe {
        pdsygvx_(
            &ibtype,
            &(jobz as u8),
            &(range as u8),
            &(uplo as u8),
            &n,
            a.data.as_mut_ptr(), &ia, &ja, a.desc.as_ptr(),
            b.data.as_mut_ptr(), &ib, &jb, b.desc.as_ptr(),
            &vl, &vu, &il, &iu,
            &abstol,
            &mut m, &mut nz,
            w.as_mut_ptr(),
            &orfac,
            z.data.as_mut_ptr(), &iz, &jz, z.desc.as_ptr(),
            work.as_mut_ptr(), &lwork,
            iwork.as_mut_ptr(), &liwork,
            ifail.as_mut_ptr(),
            iclustr.as_mut_ptr(),
            gap.as_mut_ptr(),
            &mut info,
        );
    }
    // info should be 0 on query
    if info != 0 { panic!("Query failed"); }

    let lwork_opt = work[0] as i32;
    let liwork_opt = iwork[0] as i32;
    //println!("lwork_opt:{}, liwork_opt:{}", lwork_opt, liwork_opt);
    work.resize(lwork_opt as usize, 0.0);
    iwork.resize(liwork_opt as usize, 0);

    // Second call
    unsafe {
        pdsygvx_(
            &ibtype,
            &(jobz as u8),
            &(range as u8),
            &(uplo as u8),
            &n,
            a.data.as_mut_ptr(), &ia, &ja, a.desc.as_ptr(),
            b.data.as_mut_ptr(), &ib, &jb, b.desc.as_ptr(),
            &vl, &vu, &il, &iu,
            &abstol,
            &mut m, &mut nz,
            w.as_mut_ptr(),
            &orfac,
            z.data.as_mut_ptr(), &iz, &jz, z.desc.as_ptr(),
            work.as_mut_ptr(), &lwork_opt,
            iwork.as_mut_ptr(), &liwork_opt,
            ifail.as_mut_ptr(),
            iclustr.as_mut_ptr(),
            gap.as_mut_ptr(),
            &mut info,
        );
    }

    if info != 0 {
        if info > 0 {
            // Eigenvectors failed to converge (small/ill-conditioned systems, e.g. ECP
            // molecules, can hit this in the distributed driver while the serial driver
            // converges fine). Signal failure via m = -info so the caller can fall back.
            eprintln!("WARNING: pdsygvx failed to converge, info = {}, ifail = {:?}", info, &ifail[..info.min(ifail.len() as i32) as usize]);
            return (w, -info, nz);
        } else {
            panic!("pdsygvx error, info = {}", info);
        }
    }

    (w, m, nz)
}

/// Distributed Cholesky factorization (`pdpotrf`): overwrites the upper triangle of
/// the distributed matrix `a` with the factor `U` such that `A = U^T U`.
///
/// Returns the LAPACK-style `info` (0 on success).
#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pdpotrf(
    grid: &CblacsGrid,
    uplo: char,
    a: &mut DistributedMatrixFull<f64>,
    ia: i32,
    ja: i32,
) -> i32 {
    let n = a.desc[2];
    let mut info = 0;
    unsafe {
        pdpotrf_(
            &(uplo as u8),
            &n,
            a.data.as_mut_ptr(),
            &ia,
            &ja,
            a.desc.as_ptr(),
            &mut info,
        );
    }
    info
}

/// Distributed LU factorization (`pdgetrf`): in-place `A = P·L·U` (partial pivoting)
/// on the block-cyclic distributed matrix `a`.
///
/// `ipiv` must hold at least `pdgetrf_ipiv_len` entries. Returns LAPACK-style `info`
/// (0 on success; > 0 means U(i,i) is exactly zero — singular matrix).
#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pdgetrf(
    grid: &CblacsGrid,
    a: &mut DistributedMatrixFull<f64>,
    ia: i32,
    ja: i32,
    ipiv: &mut [i32],
) -> i32 {
    let m = a.desc[2];
    let n = a.desc[3];
    let mut info = 0;
    unsafe {
        pdgetrf_(
            &m,
            &n,
            a.data.as_mut_ptr(),
            &ia,
            &ja,
            a.desc.as_ptr(),
            ipiv.as_mut_ptr(),
            &mut info,
        );
    }
    info
}

/// Distributed LU inverse (`pdgetri`): computes A⁻¹ in place from the LU factorization
/// produced by `pdgetrf`. The `ipiv` and workspace sizes follow ScaLAPACK docs:
/// `ipiv` same length as for `pdgetrf`; `work ≥ max(1, lwork)` with `lwork` from query;
/// `iwork ≥ max(1, liwork)`.
///
/// Returns LAPACK-style `info` (0 on success).
#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pdgetri(
    grid: &CblacsGrid,
    a: &mut DistributedMatrixFull<f64>,
    ia: i32,
    ja: i32,
    ipiv: &mut [i32],
) -> i32 {
    let n = a.desc[2]; // global order (square)
    let mut info = 0;

    // Workspace query
    let mut lwork = -1_i32;
    let mut liwork = -1_i32;
    let mut work = vec![0.0_f64; 1];
    let mut iwork = vec![0_i32; 1];
    unsafe {
        pdgetri_(
            &n,
            a.data.as_mut_ptr(),
            &ia,
            &ja,
            a.desc.as_ptr(),
            ipiv.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        );
    }
    if info != 0 {
        eprintln!("pdgetri workspace query failed, info = {}", info);
        return info;
    }
    let lwork_opt = work[0] as i32;
    let liwork_opt = iwork[0] as i32;
    work.resize(lwork_opt.max(1) as usize, 0.0);
    iwork.resize(liwork_opt.max(1) as usize, 0);

    // Actual inverse
    unsafe {
        pdgetri_(
            &n,
            a.data.as_mut_ptr(),
            &ia,
            &ja,
            a.desc.as_ptr(),
            ipiv.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork_opt,
            iwork.as_mut_ptr(),
            &liwork_opt,
            &mut info,
        );
    }
    info
}

/// Minimal `ipiv` length for `pdgetrf` on `a` with global row count `m_global` and
/// row blocksize `mb` (ScaLAPACK doc: numroc(m, mb, iarow, 0, nprow) + mb; ia = 1 ⇒ iarow = 0).
#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pdgetrf_ipiv_len(grid: &CblacsGrid, m_global: i32, mb: i32) -> usize {
    let nloc = unsafe {
        numroc_(&m_global, &mb, &grid.myrow, &0i32, &grid.nprow)
    };
    (nloc + mb) as usize
}

/// Local extent of a block-cyclic dimension for an *arbitrary* grid coordinate
/// (`iproc` over `nprocs`, blocksize `nb`, src 0) — e.g. the row count of the
/// band owned by BLACS row `i` before assembly.
#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pd_numroc_iproc(n_global: i32, nb: i32, iproc: i32, nprocs: i32) -> usize {
    unsafe { numroc_(&n_global, &nb, &iproc, &0i32, &nprocs) as usize }
}

/// Number of rows of a block-cyclic (mb × nb, src 0) matrix stored locally on this grid rank.
#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pd_numroc(grid: &CblacsGrid, n_global: i32, nb: i32, dim: i32) -> usize {
    let (iproc, nprocs) = if dim == 0 { (grid.myrow, grid.nprow) } else { (grid.mycol, grid.npcol) };
    unsafe { numroc_(&n_global, &nb, &iproc, &0i32, &nprocs) as usize }
}

/// Sum of ln|U(i,i)| over the local diagonal of a matrix already factorized by `pdgetrf`.
/// Global element (r, r) is stored locally iff the local (row, col) pair maps to the
/// same global index; U diagonal lives in `a` (L has unit diagonal, so det = Π U_ii).
/// Returns the LOCAL sum of ln|U_ii| — the caller must all-reduce across the grid.
#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pdgetrf_local_lnabsdet(
    grid: &CblacsGrid,
    a: &DistributedMatrixFull<f64>,
    mb: i32,
    nb: i32,
) -> f64 {
    let m_global = a.desc[2] as usize;
    let local_rows = a.size()[0];
    let local_cols = a.size()[1];
    let mut ln_det = 0.0_f64;
    for ir in 0..local_rows {
        let bi = (ir as i32) / mb;
        let off = (ir as i32) % mb;
        let g_row = (bi * grid.nprow + grid.myrow) * mb + off;
        if g_row >= m_global as i32 {
            continue;
        }
        // diagonal element (g_row, g_row): its local column
        let bj = g_row / nb;
        let c_off = g_row % nb;
        // local column index of global column g_row on this rank's mycol
        // cyclic: block bj belongs to mycol iff (bj % npcol) == mycol
        if (bj % grid.npcol) != grid.mycol {
            continue;
        }
        let ic = (bj / grid.npcol) * nb + c_off;
        if ic >= local_cols as i32 {
            continue;
        }
        ln_det += a.data[(ir as usize) + (ic as usize) * local_rows].abs().ln();
    }
    ln_det
}

/// Trace of a block-cyclic distributed matrix (sum of local diagonal elements).
#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pd_trace_local(
    grid: &CblacsGrid,
    a: &DistributedMatrixFull<f64>,
    mb: i32,
    nb: i32,
) -> f64 {
    let m_global = a.desc[2] as usize;
    let local_rows = a.size()[0];
    let local_cols = a.size()[1];
    let mut tr = 0.0_f64;
    for ir in 0..local_rows {
        let bi = (ir as i32) / mb;
        let off = (ir as i32) % mb;
        let g_row = (bi * grid.nprow + grid.myrow) * mb + off;
        if g_row >= m_global as i32 {
            continue;
        }
        let bj = g_row / nb;
        if (bj % grid.npcol) != grid.mycol {
            continue;
        }
        let c_off = g_row % nb;
        let ic = (bj / grid.npcol) * nb + c_off;
        if ic >= local_cols as i32 {
            continue;
        }
        tr += a.data[(ir as usize) + (ic as usize) * local_rows];
    }
    tr
}

/// In-place integrand preparation on the local block-cyclic slice: replicates the
/// serial `evaluate_rpa_integrand` step `diag -= 1.0; *= -1.0` — i.e. off-diagonal
/// elements are negated and diagonal elements become `-(a_ii - 1)`. Purely local.
#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pd_neg_shift_local(
    grid: &CblacsGrid,
    a: &mut DistributedMatrixFull<f64>,
    mb: i32,
    nb: i32,
) {
    let m_global = a.desc[2] as usize;
    let local_rows = a.size()[0];
    let local_cols = a.size()[1];
    for ir in 0..local_rows {
        let bi = (ir as i32) / mb;
        let off = (ir as i32) % mb;
        let g_row = (bi * grid.nprow + grid.myrow) * mb + off;
        let is_diag_row = g_row < m_global as i32;
        for ic in 0..local_cols {
            let bj = (ic as i32) / nb;
            let c_off = (ic as i32) % nb;
            let g_col = (bj * grid.npcol + grid.mycol) * nb + c_off;
            let is_diag = is_diag_row && (g_row == g_col);
            let v = &mut a.data[(ir as usize) + (ic as usize) * local_rows];
            if is_diag {
                *v = -(*v - 1.0);
            } else {
                *v = -*v;
            }
        }
    }
}

/// Distributed triangular solve (`pdtrtrs`): solves `op(A) · X = B` in place on the
/// distributed right-hand side `b`, where `A` is a distributed triangular matrix of
/// which only the specified triangle (`uplo`) is read, and `op(A)` is `A` for
/// `trans = 'N'` or `A^T` for `trans = 'T'`/`'C'`.
///
/// Returns the LAPACK-style `info` (0 on success).
#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pdtrtrs(
    grid: &CblacsGrid,
    uplo: char,
    trans: char,
    diag: char,
    a: &DistributedMatrixFull<f64>,
    ia: i32,
    ja: i32,
    b: &mut DistributedMatrixFull<f64>,
    ib: i32,
    jb: i32,
) -> i32 {
    let n = a.desc[2];
    let nrhs = b.desc[3];
    let mut info = 0;
    unsafe {
        pdtrtrs_(
            &(uplo as u8),
            &(trans as u8),
            &(diag as u8),
            &n,
            &nrhs,
            a.data.as_ptr(),
            &ia,
            &ja,
            a.desc.as_ptr(),
            b.data.as_mut_ptr(),
            &ib,
            &jb,
            b.desc.as_ptr(),
            &mut info,
        );
    }
    info
}

#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn pdsyevd(
    jobz: char,
    uplo: char,
    a: &DistributedMatrixFull<f64>,
    ia: i32,
    ja: i32,
    grid: &CblacsGrid,          // added for constructing Z
) -> Option<(DistributedMatrixFull<f64>, Vec<f64>)> {
    let n = a.desc[2];              // global size
    let block_rows = a.desc[4];     // row block size
    let block_cols = a.desc[5];     // col block size
    let irsrc = a.desc[6];          // process row owning first block
    let icsrc = a.desc[7];          // process col owning first block

    // Create output matrix Z with identical distribution as A
    let mut z = DistributedMatrixFull::new(
        grid,
        n,
        n,
        block_rows,
        block_cols,
        irsrc,
        icsrc,
        0.0,
    );

    let mut eigenvalues = vec![0.0f64; n as usize];

    // Workspace query
    let mut info = 0;
    let mut lwork = -1i32;
    let mut liwork = -1i32;
    let mut work = vec![0.0f64; 1];
    let mut iwork = vec![0i32; 1];

    unsafe {
        pdsyevd_(
            &(jobz as u8),
            &(uplo as u8),
            &n,
            a.data.as_ptr() as *mut f64,   // A is not modified by pdsyevd (input only)
            &ia,
            &ja,
            a.desc.as_ptr(),
            eigenvalues.as_mut_ptr(),
            z.data.as_mut_ptr(),
            &ia,
            &ja,
            z.desc.as_ptr(),
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        );
    }

    if info != 0 {
        eprintln!("pdsyevd workspace query failed, info = {}", info);
        return None;
    }

    let lwork_opt = work[0] as i32;
    let liwork_opt = iwork[0] as i32;

    work.resize(lwork_opt as usize, 0.0);
    iwork.resize(liwork_opt as usize, 0);

    // Actual computation
    unsafe {
        pdsyevd_(
            &(jobz as u8),
            &(uplo as u8),
            &n,
            a.data.as_ptr() as *mut f64,
            &ia,
            &ja,
            a.desc.as_ptr(),
            eigenvalues.as_mut_ptr(),
            z.data.as_mut_ptr(),
            &ia,
            &ja,
            z.desc.as_ptr(),
            work.as_mut_ptr(),
            &lwork_opt,
            iwork.as_mut_ptr(),
            &liwork_opt,
            &mut info,
        );
    }

    if info != 0 {
        eprintln!("pdsyevd failed, info = {}", info);
        return None;
    }

    if jobz == 'V' {
        Some((z, eigenvalues))
    } else {
        // Return an empty matrix for consistency
        let empty = DistributedMatrixFull::new(grid, n, n, block_rows, block_cols, irsrc, icsrc, 0.0);
        Some((empty, eigenvalues))
    }
}

#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn _hamiltonian_distributed_solver<'_a, T, Q>(
    matr_a: &T,
    matr_b: &Q,
    num_orb: &mut usize,
    grid: &CblacsGrid,
    world: &SimpleCommunicator,
) -> Option<(MatrixFull<f64>, Vec<f64>)>
where T: BasicMatUp<'_a, f64>, Q: BasicMatUp<'_a, f64>, {

    let n = matr_a.size()[0];
    if matr_b.size()[0] != n {panic!("Size mismatch");}
    // Adaptive block size: ScaLAPACK requires the block size to be smaller than
    // the global dimension (and non-zero). 256 is the default for large matrices;
    // small systems (e.g. forced-on tests with ~50 orbitals) get a smaller block.
    let nprocs_max = grid.nprow.max(grid.npcol);
    let block_rows = (256i32).min((n as i32) / (2 * nprocs_max)).max(1);
    let block_cols = block_rows;

    let mut matr_a_full = MatrixFull::new(matr_a.size(), 0.0);
    matr_a_full.iter_matrixupper_mut().unwrap().zip(matr_a.data_ref().unwrap()).for_each(|(to, from)| {
        *to = *from;
    });
    matr_a_full.fill_lower_part_from_upper_unsafe();
    let mut matr_b_full = MatrixFull::new(matr_b.size(), 0.0);
    matr_b_full.iter_matrixupper_mut().unwrap().zip(matr_b.data_ref().unwrap()).for_each(|(to, from)| {
        *to = *from;
    });
    matr_b_full.fill_lower_part_from_upper_unsafe();

    let mut dist_matr_a = DistributedMatrixFull::from_matrixfull(grid, &matr_a_full, block_rows, block_cols, 0, 0);
    let mut dist_matr_b = DistributedMatrixFull::from_matrixfull(grid, &matr_b_full, block_rows, block_cols, 0, 0);
    let mut dist_matr_c = DistributedMatrixFull::new(grid, n as i32, n as i32, block_rows, block_cols, 0, 0, 0.0 as f64);

    let (eigenvalues, m, nz) = pdsygvx(
            grid,
            1,               // ibtype: A*x = λ*B*x
            'V',             // jobz: compute eigenvectors
            'A',             // range: all eigenvalues
            'U',             // uplo: upper triangle stored
            &mut dist_matr_a, 1, 1,
            &mut dist_matr_b, 1, 1,
            0.0, 0.0,       // VL, VU (unused)
            0, 0,           // IL, IU (unused)
            // abstol ~ sqrt(eps) as recommended by LAPACK. orfac controls the
            // reorthogonalization threshold for clustered eigenvectors: with the
            // LAPACK-default 1e-3, pdsygvx repeatedly fails to converge (info > 0)
            // on ill-conditioned/clustered problems (author-verified on random
            // 2048x2048 PSD matrices: orfac must be < 1e-8). Use 1e-10.
            1e-8,           // abstol
            1e-10,          // orfac
            &mut dist_matr_c, 1, 1,
        );

    if m < 0 {
        // distributed solve failed (non-convergence); let the caller fall back
        return None;
    }


    let root = 0;
    let gathered_msgs_z = dist_matr_c.gather_to_root(root as i32, grid, world);
    let rank = world.rank();
    let mut full_c = if rank == root {
        let matr_c = reconstruct_from_gathered(&gathered_msgs_z, n as i32, n as i32, block_rows, block_cols, grid.nprow, grid.npcol, 0, 0);
        matr_c.data
    }
    else {
        vec![0.0; n * n]
    };

    world.process_at_rank(root).broadcast_into(&mut full_c);
    let matr_c_full = MatrixFull::from_vec([n, n], full_c).unwrap();
    Some((matr_c_full, eigenvalues))

}

#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
fn _distributed_get_sqrt_and_inv_sqrt(matr_a: &DistributedMatrixFull<f64>, threshold: f64, grid: &CblacsGrid)
-> Option<(DistributedMatrixFull<f64>, DistributedMatrixFull<f64>, usize)>
{
    let n = matr_a.desc[2] as usize;          // global dimension (square)
    let block_rows = matr_a.desc[4] as i32;   // row block size
    let block_cols = matr_a.desc[5] as i32;   // col block size

    // 1. Diagonalize the overlap matrix to get eigenvectors and eigenvalues.
    //    pdsyevd returns eigenvalues in ascending order.
    let (eigvec, eigenvalues) = pdsyevd('V', 'U', matr_a, 1, 1, grid)?;

    // 2. Count singular eigenvalues (those below threshold).
    let n_singular = eigenvalues.iter().filter(|&&v| v < threshold).count();
    let n_non_singular = n - n_singular;

    // 3. Create two matrices to hold scaled eigenvectors.
    let mut sqrt_eigvec = DistributedMatrixFull::new(
        grid,
        n as i32,
        n as i32,
        block_rows,
        block_cols,
        matr_a.desc[6],   // irsrc
        matr_a.desc[7],   // icsrc
        0.0,
    );
    let mut inv_sqrt_eigvec = DistributedMatrixFull::new(
        grid,
        n as i32,
        n as i32,
        block_rows,
        block_cols,
        matr_a.desc[6],
        matr_a.desc[7],
        0.0,
    );

    // 4. Scale columns of eigvec and copy into the two matrices.
    //    For each local column, compute its global column index.
    let myrow = grid.myrow;
    let mycol = grid.mycol;
    let nprow = grid.nprow;
    let npcol = grid.npcol;
    let icsrc = matr_a.desc[7];

    let local_rows = eigvec.size()[0];
    let local_cols = eigvec.size()[1];

    for local_col in 0..local_cols {
        // Compute global column index for this local column.
        // Block-cyclic mapping: local_col = local_block * block_cols + offset_in_block
        let local_block = (local_col / block_cols as usize) as i32;
        let offset_in_block = (local_col % block_cols as usize) as i32;
        // Global block index for this local block on this process:
        // The k-th block for this process (k = local_block) has global block index:
        //   b = icsrc + ((mycol - icsrc) mod npcol) + local_block * npcol
        let mycol_mod = (mycol - icsrc).rem_euclid(npcol);
        let global_block = icsrc + mycol_mod + local_block * npcol;
        let global_col = (global_block * block_cols + offset_in_block) as usize;

        assert!(global_col < n);
        let ev = eigenvalues[global_col];
        let (sqrt_factor, inv_sqrt_factor) = if ev >= threshold {
            let ev_sqrt = ev.sqrt().sqrt();   // λ^(1/4)
            (ev_sqrt, 1.0 / ev_sqrt)
        } else {
            (0.0, 0.0)   // singular direction: zero out
        };

        // Scale the entire column (column-major storage)
        let col_offset = local_col * local_rows;
        for local_row in 0..local_rows {
            let val = eigvec.data[col_offset + local_row];
            sqrt_eigvec.data[col_offset + local_row] = val * sqrt_factor;
            inv_sqrt_eigvec.data[col_offset + local_row] = val * inv_sqrt_factor;
        }
    }

    // 5. Compute final matrices:
    //    sqrt_mat      = sqrt_eigvec * sqrt_eigvec^T
    //    inv_sqrt_mat  = inv_sqrt_eigvec * inv_sqrt_eigvec^T
    let mut sqrt_mat = DistributedMatrixFull::new(
        grid,
        n as i32,
        n as i32,
        block_rows,
        block_cols,
        matr_a.desc[6],
        matr_a.desc[7],
        0.0,
    );
    let mut inv_sqrt_mat = DistributedMatrixFull::new(
        grid,
        n as i32,
        n as i32,
        block_rows,
        block_cols,
        matr_a.desc[6],
        matr_a.desc[7],
        0.0,
    );

    pdgemm('N', 'T', 1.0, &sqrt_eigvec, 1, 1, &sqrt_eigvec, 1, 1, 0.0, &mut sqrt_mat, 1, 1);
    pdgemm('N', 'T', 1.0, &inv_sqrt_eigvec, 1, 1, &inv_sqrt_eigvec, 1, 1, 0.0, &mut inv_sqrt_mat, 1, 1);

    Some((sqrt_mat, inv_sqrt_mat, n_non_singular))
}

#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
pub fn _hamiltonian_distributed_solver_inv<'_a, T, Q>(
    matr_a: &T,
    matr_b: &Q,
    num_orb: &mut usize,
    grid: &CblacsGrid,
    world: &SimpleCommunicator,
) -> Option<(MatrixFull<f64>, Vec<f64>)>
where T: BasicMatUp<'_a, f64>, Q: BasicMatUp<'_a, f64>,
{
    let n = matr_a.size()[0];
    if matr_b.size()[0] != n {panic!("Size mismatch");}
    // Adaptive block size: ScaLAPACK requires the block size to be smaller than
    // the global dimension (and non-zero). 256 is the default for large matrices;
    // small systems (e.g. forced-on tests with ~50 orbitals) get a smaller block.
    let nprocs_max = grid.nprow.max(grid.npcol);
    let block_rows = (256i32).min((n as i32) / (2 * nprocs_max)).max(1);
    let block_cols = block_rows;

    let mut matr_a_full = MatrixFull::new(matr_a.size(), 0.0);
    matr_a_full.iter_matrixupper_mut().unwrap().zip(matr_a.data_ref().unwrap()).for_each(|(to, from)| {
        *to = *from;
    });
    matr_a_full.fill_lower_part_from_upper_unsafe();
    let mut matr_b_full = MatrixFull::new(matr_b.size(), 0.0);
    matr_b_full.iter_matrixupper_mut().unwrap().zip(matr_b.data_ref().unwrap()).for_each(|(to, from)| {
        *to = *from;
    });
    matr_b_full.fill_lower_part_from_upper_unsafe();

    let mut dist_matr_a = DistributedMatrixFull::from_matrixfull(grid, &matr_a_full, block_rows, block_cols, 0, 0);
    let mut dist_matr_b = DistributedMatrixFull::from_matrixfull(grid, &matr_b_full, block_rows, block_cols, 0, 0);

    let (dist_matr_b_sqrt, dist_matr_b_sqrt_inv, n_singularity) = _distributed_get_sqrt_and_inv_sqrt(&dist_matr_b, 1.0e-10, grid).unwrap();

    if n_singularity < *num_orb {*num_orb = n_singularity};

    let mut dist_matr_tmp = DistributedMatrixFull::new(grid, n as i32, n as i32, block_rows, block_cols, 0, 0, 0.0 as f64);
    let mut dist_matr_a_transformed = DistributedMatrixFull::new(grid, n as i32, n as i32, block_rows, block_cols, 0, 0, 0.0 as f64);

    pdgemm('N', 'N', 1.0, &dist_matr_a, 1, 1, &dist_matr_b_sqrt_inv, 1, 1, 0.0, &mut dist_matr_tmp, 1, 1);
    pdgemm('N', 'N', 1.0, &dist_matr_b_sqrt_inv, 1, 1, &dist_matr_tmp, 1, 1, 0.0, &mut dist_matr_a_transformed, 1, 1);

    let (dist_eigenvec, eigenvalues) = pdsyevd('V', 'U', &dist_matr_a_transformed, 1, 1, grid)?;

    let mut dist_final_eigenvec = DistributedMatrixFull::new(grid, n as i32, n as i32, block_rows, block_cols, 0, 0, 0.0 as f64);
    pdgemm('N', 'N', 1.0, &dist_matr_b_sqrt_inv, 1, 1, &dist_eigenvec, 1, 1, 0.0, &mut dist_final_eigenvec, 1, 1);

    let root = 0;
    let gathered_msgs_z = dist_final_eigenvec.gather_to_root(root as i32, grid, world);
    let rank = world.rank();
    let mut full_c = if rank == root {
        let matr_c = reconstruct_from_gathered(&gathered_msgs_z, n as i32, n as i32, block_rows, block_cols, grid.nprow, grid.npcol, 0, 0);
        matr_c.data
    }
    else {
        vec![0.0; n * n]
    };

    world.process_at_rank(root).broadcast_into(&mut full_c);
    let matr_c_full = MatrixFull::from_vec([n, n], full_c).unwrap();

    // Extract only the first num_orb columns
    let mut eigenvector = MatrixFull::new([n, *num_orb], 0.0);
    for j in 0..*num_orb {
        for i in 0..n {
            eigenvector[[i, j]] = matr_c_full[[i, j]];
        }
    }
    Some((eigenvector, eigenvalues))
}


#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
#[test]
fn distribute_from_full() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as usize;
    let rank = world.rank() as usize;
    assert_eq!(size, 4);
    let cblacsgrid = CblacsGrid::new(size, "R");
    let global_mat =
        MatrixFull::from_vec([10, 10], (0..100).map(|x| x as i32).collect::<Vec<i32>>()).unwrap();
    let local_mat = DistributedMatrixFull::from_matrixfull(&cblacsgrid, &global_mat, 2, 2, 0, 0);
    let local_data_ref = local_mat.data_ref().unwrap();
    if (cblacsgrid.myrow == 0) && (cblacsgrid.mycol == 0) {
        assert_eq!(local_mat.size()[0], 6);
        assert_eq!(local_mat.size()[1], 6);
        assert_eq!(local_data_ref[1], 1);
        assert_eq!(local_data_ref[33], 95);
    }
    if (cblacsgrid.myrow == 0) && (cblacsgrid.mycol == 1) {
        assert_eq!(local_mat.size()[0], 6);
        assert_eq!(local_mat.size()[1], 4);
        assert_eq!(local_data_ref[1], 21);
        assert_eq!(local_data_ref[14], 64);
    }
    if (cblacsgrid.myrow == 1) && (cblacsgrid.mycol == 0) {
        assert_eq!(local_mat.size()[0], 4);
        assert_eq!(local_mat.size()[1], 6);
        assert_eq!(local_data_ref[1], 3);
        assert_eq!(local_data_ref[17], 83);
    }
    if (cblacsgrid.myrow == 1) && (cblacsgrid.mycol == 1) {
        assert_eq!(local_mat.size()[0], 4);
        assert_eq!(local_mat.size()[1], 4);
        assert_eq!(local_data_ref[1], 23);
        assert_eq!(local_data_ref[15], 77);
    }

    let global_mat2 =
        MatrixFull::from_vec([7, 7], (0..49).map(|x| x as i32).collect::<Vec<i32>>()).unwrap();
    let local_mat2 = DistributedMatrixFull::from_matrixfull(&cblacsgrid, &global_mat2, 2, 2, 0, 0);
    let local_data_ref2 = local_mat2.data_ref().unwrap();
    if (cblacsgrid.myrow == 0) && (cblacsgrid.mycol == 0) {
        assert_eq!(local_mat2.size()[0], 4);
        assert_eq!(local_mat2.size()[1], 4);
        assert_eq!(local_data_ref2[1], 1);
        assert_eq!(local_data_ref2[9], 29);
    }
    if (cblacsgrid.myrow == 0) && (cblacsgrid.mycol == 1) {
        assert_eq!(local_mat2.size()[0], 4);
        assert_eq!(local_mat2.size()[1], 3);
        assert_eq!(local_data_ref2[1], 15);
        assert_eq!(local_data_ref2[10], 46);
    }
    if (cblacsgrid.myrow == 1) && (cblacsgrid.mycol == 0) {
        assert_eq!(local_mat2.size()[0], 3);
        assert_eq!(local_mat2.size()[1], 4);
        assert_eq!(local_data_ref2[1], 3);
        assert_eq!(local_data_ref2[5], 13);
    }
    if (cblacsgrid.myrow == 1) && (cblacsgrid.mycol == 1) {
        assert_eq!(local_mat2.size()[0], 3);
        assert_eq!(local_mat2.size()[1], 3);
        assert_eq!(local_data_ref2[1], 17);
        assert_eq!(local_data_ref2[5], 27);
    }
}

#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
#[test]
fn distribute_from_upper() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as usize;
    let rank = world.rank() as usize;
    assert_eq!(size, 4);
    let cblacsgrid = CblacsGrid::new(size, "R");
    let mat_upper = MatrixUpper::from_vec(55, (0..55).map(|x| x as i32).collect::<Vec<i32>>()).unwrap();
    let local_mat = DistributedMatrixFull::from_matrixupper(&cblacsgrid, &mat_upper, 2, 2, 0, 0);
    let local_data_ref = local_mat.data_ref().unwrap();
    if (cblacsgrid.myrow == 0) && (cblacsgrid.mycol == 0) {
        assert_eq!(local_mat.size()[0], 6);
        assert_eq!(local_mat.size()[1], 6);
        assert_eq!(local_data_ref[1], 1);
        assert_eq!(local_data_ref[33], 50);
    }
    if (cblacsgrid.myrow == 0) && (cblacsgrid.mycol == 1) {
        assert_eq!(local_mat.size()[0], 6);
        assert_eq!(local_mat.size()[1], 4);
        assert_eq!(local_data_ref[1], 4);
        assert_eq!(local_data_ref[14], 25);
    }
    if (cblacsgrid.myrow == 1) && (cblacsgrid.mycol == 0) {
        assert_eq!(local_mat.size()[0], 4);
        assert_eq!(local_mat.size()[1], 6);
        assert_eq!(local_data_ref[1], 6);
        assert_eq!(local_data_ref[17], 39);
    }
    if (cblacsgrid.myrow == 1) && (cblacsgrid.mycol == 1) {
        assert_eq!(local_mat.size()[0], 4);
        assert_eq!(local_mat.size()[1], 4);
        assert_eq!(local_data_ref[1], 8);
        assert_eq!(local_data_ref[15], 35);
    }
}

#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
#[test]
fn gather_from_distributed() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as usize;
    let rank = world.rank() as usize;
    assert_eq!(size, 4);
    let cblacsgrid = CblacsGrid::new(size, "R");
    let global_mat =
        MatrixFull::from_vec([10, 10], (0..100).map(|x| x as i32).collect::<Vec<i32>>()).unwrap();
    let local_mat = DistributedMatrixFull::from_matrixfull(&cblacsgrid, &global_mat, 2, 2, 0, 0);
    let root = 0;
    let gathered = local_mat.gather_to_root(root, &cblacsgrid, &world);
    let recv_size = gathered.len();
    if rank == (root as usize) {
        assert_eq!(recv_size, size);
        let reconstructed_global_mat = reconstruct_from_gathered(&gathered, 10, 10, 2, 2, cblacsgrid.nprow, cblacsgrid.npcol, 0, 0);
        assert_eq!(global_mat, reconstructed_global_mat);
    }
    else {
        assert_eq!(recv_size, 0);
    }
}

#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
#[test]
fn test_pdgemm() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let grid = CblacsGrid::new(world.size() as usize, "R");  // adjust to world size
    let block_rows = 64 as i32;
    let block_cols = 64 as i32;
    let irsrc = 0_i32;
    let icsrc = 0_i32;
    let m = 128 as usize;
    let n = 128 as usize;
    let k = 128 as usize;
    let offset = 35 as usize;
    // Global matrices (column‑major)
    let global_a = MatrixFull::from_vec([m, n], (0..(m * n)).map(|x| x as f64).collect::<Vec<f64>>()).unwrap();
    let global_b = MatrixFull::from_vec([n, k], (offset..(offset + n * k)).map(|x| x as f64).collect::<Vec<f64>>()).unwrap();

    let dist_a = DistributedMatrixFull::from_matrixfull(&grid, &global_a, block_rows, block_cols, irsrc, icsrc);
    let dist_b = DistributedMatrixFull::from_matrixfull(&grid, &global_b, block_rows, block_cols, irsrc, icsrc);
    let mut dist_c = DistributedMatrixFull::new(&grid, m as i32, k as i32, block_rows, block_cols, irsrc, icsrc, 0.0 as f64);

    pdgemm('N', 'N', 1.0, &dist_a, 1, 1, &dist_b, 1, 1, 0.0, &mut dist_c, 1, 1);

    // Gather dist_c on root and compare with serial dgemm
    let gathered_c = dist_c.gather_to_root(0, &grid, &world);
    if world.rank() == 0 {
        let mut serial_c = MatrixFull::new([m,k], 0.0 as f64);
        _dgemm_full(&global_a, 'N', &global_b, 'N', &mut serial_c, 1.0, 0.0);

        let mut reconstructed_c = reconstruct_from_gathered(&gathered_c, m as i32, k as i32, block_rows, block_cols, grid.nprow, grid.npcol, irsrc, icsrc);
        assert_eq!(serial_c, reconstructed_c);

    }
}


#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
#[test]
fn eigen_solver() {
    let n = 64;
    let mut rng = rand::thread_rng();

    // 1. Generate a random symmetric H (upper triangle only)
    let mut h_packed = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in i..n {
            h_packed.push(rng.gen_range(-5.0..5.0));
        }
    }
    let mat_h = MatrixUpper::from_vec(h_packed.len(), h_packed).unwrap();

    // 2. Generate a random positive definite S
    //    Create a random n×n matrix M, then S = M * M^T (positive definite).
    let mut m = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            m[j * n + i] = rng.gen_range(-2.0..2.0);
        }
    }
    let mat_m = MatrixFull::from_vec([n, n], m).unwrap();
    let mut mat_s_full = MatrixFull::new([n, n], 0.0 as f64);
    _dgemm_full(&mat_m, 'N', &mat_m, 'T', &mut mat_s_full, 1.0, 0.0);

    let mut mat_s = MatrixUpper::new((n * (n + 1) / 2), 0.0);
    /*
    mat_s_full
        .iter_matrixupper()
        .unwrap()
        .zip(mat_s.iter_mut())
        .for_each(|(from, to)| *to = *from);
    */
    let vec_diag = (1..(n+1)).map(|x| x as f64).collect::<Vec<f64>>();
    vec_diag.iter().zip(mat_s.iter_diagonal_mut()).for_each(|(from, to)| *to = *from);
    let mat_s_full = mat_s.to_matrixfull().unwrap();
    // 3. Solve
    let mut num_orb = n; // compute all eigenvectors
    let (eigenvectors, eigenvalues) = _hamiltonian_fast_solver(&mat_h, &mat_s, &mut num_orb).unwrap();
    assert_eq!(num_orb, n);
    assert_eq!(eigenvectors.size(), [n, num_orb]);
    assert_eq!(eigenvalues.len(), n);

    let mut lhs = MatrixFull::new([n, n], 0.0 as f64);
    let mut rhs_tmp = MatrixFull::new([n, n], 0.0 as f64);
    let mut rhs = MatrixFull::new([n, n], 0.0 as f64);
    let mut mat_e = MatrixFull::new([n, n], 0.0 as f64);
    let mat_h_full = mat_h.to_matrixfull().unwrap();
    _dgemm_full(&mat_h_full, 'N', &eigenvectors, 'N', &mut lhs, 1.0, 0.0);
    _dgemm_full(&mat_s_full, 'N', &eigenvectors, 'N', &mut rhs_tmp, 1.0, 0.0);
    eigenvalues.iter().zip(mat_e.iter_diagonal_mut().unwrap()).for_each(|(from, to)| *to = *from);
    _dgemm_full(&rhs_tmp, 'N', &mat_e, 'N', &mut rhs, 1.0, 0.0);
    let max_diff = lhs.data.iter().zip(rhs.data.iter()).map(|(a, b)| (a-b).abs()).fold(0.0, f64::max);
    assert!(max_diff < 1e-10, "max difference too large: {}", max_diff);

}

#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
#[test]
fn distributed_hamiltonian_solver() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let grid = CblacsGrid::new(world.size() as usize, "R");  // adjust to world size

    let n = 2000;
    let mut rng = rand::thread_rng();
    let mut h_packed_vec = vec![];
    let mut s_packed_vec = vec![];
    if world.rank() == 0 {
        h_packed_vec = Vec::with_capacity(n * (n + 1) / 2);
        for i in 0..n {
            for j in i..n {
                h_packed_vec.push(rng.gen_range(-5.0..5.0));
            }
        }

        let mut m = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                m[j * n + i] = rng.gen_range(-2.0..2.0);
            }
        }
        let mat_m = MatrixFull::from_vec([n, n], m).unwrap();
        let mut mat_s_full = MatrixFull::new([n, n], 0.0 as f64);
        _dgemm_full(&mat_m, 'N', &mat_m, 'T', &mut mat_s_full, 1.0, 0.0);

        s_packed_vec = vec![0.0; n * (n + 1) / 2];
        mat_s_full
            .iter_matrixupper()
            .unwrap()
            .zip(s_packed_vec.iter_mut())
            .for_each(|(from, to)| *to = *from);
    }

    let mut h_packed_buf = if world.rank() == 0 {
        h_packed_vec
        }
        else {
            vec![0.0; (n * (n + 1)) / 2]
        };
    world.process_at_rank(0).broadcast_into(&mut h_packed_buf);
    let mat_h = MatrixUpper::from_vec(h_packed_buf.len(), h_packed_buf).unwrap();

    let mut s_packed_buf = if world.rank() == 0 {
        s_packed_vec
        }
        else {
            vec![0.0; n * (n + 1) / 2]
        };
    world.process_at_rank(0).broadcast_into(&mut s_packed_buf);
    let mat_s = MatrixUpper::from_vec(s_packed_buf.len(), s_packed_buf).unwrap();

    /*
    let mut mat_s = MatrixUpper::new((n * (n + 1) / 2), 0.0);
    let vec_s = (1..n + 1).map(|x| x as f64).collect::<Vec<f64>>();
    mat_s.iter_diagonal_mut().zip(vec_s.iter()).for_each(|(from, to)| *from = *to);
    */
    let mut num_orb = n; // compute all eigenvectors

    let (eigenvectors, eigenvalues) = _hamiltonian_distributed_solver_inv(&mat_h, &mat_s, &mut num_orb, &grid, &world).unwrap();
    let mut num_orb_s = n;
    let (eigenvectors_serial, eigenvalues_serial) = _hamiltonian_fast_solver(&mat_h, &mat_s, &mut num_orb_s).unwrap();
    let max_eig = eigenvalues.iter().fold(0.0, |a, &b| f64::max(a, b.abs()));
    let max_diff = eigenvalues.iter().zip(eigenvalues_serial.iter()).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max);
    let rel_diff_eig = max_diff / max_eig.max(1.0); // avoid division by zero
    assert!(rel_diff_eig < 1e-5,
            "Relative eigenvalue error too large: {}", rel_diff_eig);

    let mut lhs = MatrixFull::new([n, n], 0.0 as f64);
    let mut rhs_tmp = MatrixFull::new([n, n], 0.0 as f64);
    let mut rhs = MatrixFull::new([n, n], 0.0 as f64);
    let mut mat_e = MatrixFull::new([n, n], 0.0 as f64);
    let mat_h_full = mat_h.to_matrixfull().unwrap();
    let mat_s_full = mat_s.to_matrixfull().unwrap();
    _dgemm_full(&mat_h_full, 'N', &eigenvectors, 'N', &mut lhs, 1.0, 0.0);
    _dgemm_full(&mat_s_full, 'N', &eigenvectors, 'N', &mut rhs_tmp, 1.0, 0.0);

    eigenvalues.iter().zip(mat_e.iter_diagonal_mut().unwrap()).for_each(|(from, to)| *to = *from);
    _dgemm_full(&rhs_tmp, 'N', &mat_e, 'N', &mut rhs, 1.0, 0.0);

    let tol = 1e-8; // adjust based on your accuracy needs
    let norm_lhs = lhs.data.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_rhs = rhs.data.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_diff = lhs.data.iter()
        .zip(rhs.data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    let scale = 1.0 + norm_lhs.max(norm_rhs);
    assert!(norm_diff < tol * scale,
            "Frobenius norm of difference too large: {} (tolerance: {})",
            norm_diff, tol * scale);
}


#[cfg(feature = "mpi")]
#[cfg(feature = "scalapack")]
#[test]
fn distributed_inv() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let grid = CblacsGrid::new(world.size() as usize, "R");  // adjust to world size


    let n = 512; // matrix dimension
    let threshold = 1.0e-10;

    // Generate a random symmetric positive semi-definite matrix S = M*M^T on rank 0.
    let mut s_full_data = vec![0.0f64; n * n];
    if world.rank() == 0 {
        let mut rng = rand::thread_rng();
        let mut m = vec![0.0f64; n * n];
        for val in m.iter_mut() {
            *val = rng.gen_range(-2.0..2.0);
        }
        let mat_m = MatrixFull::from_vec([n, n], m).unwrap();
        let mut mat_s = MatrixFull::new([n, n], 0.0);
        _dgemm_full(&mat_m, 'N', &mat_m, 'T', &mut mat_s, 1.0, 0.0);
        s_full_data = mat_s.data;
    }

    // Broadcast the full S matrix data (column-major) to all ranks.
    world.process_at_rank(0).broadcast_into(&mut s_full_data);

    // Each rank constructs its own MatrixFull for S.
    let mat_s_full = MatrixFull::from_vec([n, n], s_full_data).unwrap();

    // Distribute S using the same block size as in the real solver.
    let nprocs_max = grid.nprow.max(grid.npcol);
    let block_rows = (256i32).min((n as i32) / (2 * nprocs_max)).max(1);
    let block_cols = block_rows;

    let dist_s = DistributedMatrixFull::from_matrixfull(
        &grid,
        &mat_s_full,
        block_rows,
        block_cols,
        0,
        0,
    );

    // Compute distributed S^{1/2} and S^{-1/2}.
    let (dist_sqrt, dist_inv_sqrt, n_non_singular_dist) =
        _distributed_get_sqrt_and_inv_sqrt(&dist_s, threshold, &grid)
            .expect("Distributed sqrt/inv_sqrt failed");

    // Compute serial S^{1/2} and S^{-1/2} for comparison (on every rank; cheap for test).
    let (serial_sqrt, serial_inv_sqrt, n_non_singular_serial) =
        _get_sqrt_and_inv_sqrt(&mat_s_full, threshold).expect("Serial sqrt/inv_sqrt failed");

    assert_eq!(n_non_singular_dist, n_non_singular_serial,
               "Number of non-singular directions mismatch");

    // Gather distributed results to root.
    let gathered_sqrt = dist_sqrt.gather_to_root(0, &grid, &world);
    let gathered_inv = dist_inv_sqrt.gather_to_root(0, &grid, &world);

    if world.rank() == 0 {
        // Reconstruct full matrices on root.
        let full_sqrt = reconstruct_from_gathered(
            &gathered_sqrt, n as i32, n as i32,
            block_rows, block_cols, grid.nprow, grid.npcol, 0, 0,
        );
        let full_inv = reconstruct_from_gathered(
            &gathered_inv, n as i32, n as i32,
            block_rows, block_cols, grid.nprow, grid.npcol, 0, 0,
        );

        // Compare element-wise with serial results.
        let tol = 1e-8;
        for i in 0..n {
            for j in 0..n {
                let diff_sqrt = (full_sqrt[[i, j]] - serial_sqrt[[i, j]]).abs();
                let diff_inv = (full_inv[[i, j]] - serial_inv_sqrt[[i, j]]).abs();
                assert!(
                    diff_sqrt < tol,
                    "sqrt mismatch at ({}, {}): distributed={}, serial={}",
                    i, j, full_sqrt[[i, j]], serial_sqrt[[i, j]]
                );
                assert!(
                    diff_inv < tol,
                    "inv_sqrt mismatch at ({}, {}): distributed={}, serial={}",
                    i, j, full_inv[[i, j]], serial_inv_sqrt[[i, j]]
                );
            }
        }
    }

    // Ensure all processes finish before MPI finalization.
    world.barrier();

}
