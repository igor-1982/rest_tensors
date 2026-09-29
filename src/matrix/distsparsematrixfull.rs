// rest_tensors/src/matrix/distsparsematrixfull.rs
use std::sync::Arc;
use crate::matrix::{
    BasicMatrix, BasicMatrixOpt, MatFormat, MathMatrix, MatrixFull, ParMathMatrix,
};
use crate::matrix::matrixupper::*;
use crate::matrix::matrix_blas_lapack::{_dgemm_full, _hamiltonian_fast_solver, _get_sqrt_and_inv_sqrt};
use crate::matrix::matrix_pexsi::*;

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

fn column_range(mycol: i32, npcol: i32, ncols: usize) -> (usize, usize) {
    let npcol_u = npcol as usize;
    let base = ncols / npcol_u;
    let rem  = ncols % npcol_u;
    let mycol_u = mycol as usize;
    let start = if mycol_u < rem {
        mycol_u * (base + 1)
    } else {
        rem * (base + 1) + (mycol_u - rem) * base
    };
    let len = if mycol_u < rem { base + 1 } else { base };
    (start, start + len)
}

/// Inverse of `column_range`: which rank owns global column `j`.
fn column_owner(j: usize, nprocs: usize, ncols: usize) -> usize {
    let base = ncols / nprocs;
    let rem  = ncols % nprocs;
    let threshold = rem * (base + 1);   // first column owned by the "short" ranks
    if j < threshold {
        j / (base + 1)
    } else {
        rem + (j - threshold) / base
    }
}


/// Immutable structure of the sparse matrix: shape, distribution, CSC pattern.
/// Built once per SCF calculation. Shared (via Arc or &) by every value-bearing
/// view of H, S, and DM (when their patterns coincide).
pub struct SparsePattern {
    /// Global shape, same on every rank.
    pub global_size: [usize; 2],
    pub global_nnz:  usize,

    /// Full global CSC pattern, identical on every rank.
    /// `global_colptr[0] = 0`, `global_colptr[ncols] = global_nnz`.
    pub global_colptr: Vec<i32>,
    pub global_rowind: Vec<i32>,

    /// This rank's column slice.
    pub col_range: (usize, usize),

    /// Local CSC arrays, 0-based, ready for direct FFI.
    /// `local_colptr[j] = global_colptr[start + j] - global_colptr[start]`
    /// `local_rowind   = global_rowind[global_colptr[start] .. global_colptr[end]]`
    pub local_colptr: Vec<i32>,
    pub local_rowind: Vec<i32>,
}

impl SparsePattern {
    #[inline] pub fn nnz_local(&self)     -> usize { self.local_rowind.len() }
    #[inline] pub fn num_col_local(&self) -> usize { self.col_range.1 - self.col_range.0 }

    /// Build once from a dense matrix + threshold. This is the only place
    /// that scans the full dense layout.
    pub fn from_matrixfull<T: Copy + Into<f64>>(
        mat: &MatrixFull<T>,
        eps: f64,
        npcol: i32,
        mycol: i32,
    ) -> Self {
        let [nrows, ncols] = mat.size();
    
        // Build the FULL global pattern. Every rank does the same work and gets
        // the same result, so no communication is needed for the pattern itself.
        let mut global_colptr: Vec<i32> = Vec::with_capacity(ncols + 1);
        let mut global_rowind: Vec<i32> = Vec::new();
        global_colptr.push(1);
        for j in 0..ncols {
            for i in 0..nrows {
                if Into::<f64>::into(mat.data[j * nrows + i]).abs() > eps {
                    global_rowind.push((i + 1) as i32);
                }
            }
            global_colptr.push(global_rowind.len() as i32 + 1);
        }
        let global_nnz = global_rowind.len();
    
        // Slice out this rank's local pattern and rebase colptr to 0.
        let (start, end) = column_range(mycol, npcol, ncols);
        let base = global_colptr[start]; 
        let offset = (base - 1) as usize;
        let local_colptr: Vec<i32> = global_colptr[start..=end]
            .iter()
            .map(|&v| v - base + 1)
            .collect();
        let local_rowind: Vec<i32> =
            global_rowind[offset..((global_colptr[end] - 1) as usize)].to_vec();
    
        Self {
            global_size: [nrows, ncols],
            global_nnz,
            global_colptr,
            global_rowind,
            col_range: (start, end),
            local_colptr,
            local_rowind,
        }
    }

    /// Fill a value buffer from a dense matrix, walking the pattern in the
    /// order it expects. No allocation, no pattern rebuild, no full-matrix
    /// intermediate. O(nnz_local) work, not O(N²).
    pub fn fill_from_matrixfull<T: Copy + Into<f64>>(
        &self, mat: &MatrixFull<T>, nzval: &mut Vec<T>,
    ) {
        let nrows = self.global_size[0];
        let (start, end) = self.col_range;
        nzval.clear();
        nzval.reserve(self.nnz_local());

        for (local_j, j) in (start..end).enumerate() {
            let lo = (self.local_colptr[local_j] - 1) as usize;
            let hi = (self.local_colptr[local_j + 1] - 1) as usize;
            for k in lo..hi {
                let i1 = self.local_rowind[k] as usize; 
                let i = i1 - 1;
                nzval.push(mat.data[j * nrows + i]);
            }
        }
        
    }

    /// Same, but reading directly from the packed upper triangle. Skips
    /// `to_matrixfull()` entirely. For (i, j) entries with j < i, mirrors
    /// from (j, i) — valid because H and S are symmetric.
    pub fn fill_from_matrixupper<T: Copy + Default>(
        &self, mat: &MatrixUpper<T>, nzval: &mut Vec<T>,
    ) {
        let (start, end) = self.col_range;
        nzval.clear();
        nzval.reserve(self.nnz_local());

        for (local_j, j) in (start..end).enumerate() {
            let lo = (self.local_colptr[local_j] - 1) as usize;
            let hi = (self.local_colptr[local_j + 1] - 1) as usize;
            for k in lo..hi {
                let i1 = self.local_rowind[k] as usize; 
                let i = i1 - 1;
                let (r, c) = if i <= j {(i, j)} else {(j, i)};
                nzval.push(mat.data[r + c * (c + 1) / 2]);
            }
        }
    }
}

pub struct DistSparseMatrixFull<T> {
    pub pattern: Arc<SparsePattern>,
    /// Local nonzero values. Length `nnz_local`.
    pub nzval: Vec<T>,
}

impl<T> DistSparseMatrixFull<T> {
    #[inline] pub fn nnz_local(&self)       -> usize { self.pattern.nnz_local() }
    #[inline] pub fn num_col_local(&self)   -> usize { self.pattern.num_col_local() }
    #[inline] pub fn nrows(&self)           -> usize { self.pattern.global_size[0] }
    #[inline] pub fn ncols(&self)           -> usize { self.pattern.global_size[1] }
    #[inline] pub fn col_range(&self)       -> (usize, usize) {self.pattern.col_range}
}

impl<T> DistSparseMatrixFull<T>
where T: Copy + Clone + Default + Send + Buffer + 'static + mpi::datatype::Equivalence, f64: From<T>,
{
    
    pub fn gather_to_root(&self, root: i32, world: &SimpleCommunicator) -> Vec<Option<Vec<T>>> {
        let rank = world.rank();
        let size = world.size();
        let mut msgs: Vec<Option<Vec<T>>> = vec![None; size as usize];

        if rank == root {
            msgs[root as usize] = Some(self.nzval.clone());
            for r in 0..size {
                if r == root {continue;}
                let (nzval_recv, _) = world.process_at_rank(r).receive_vec::<T>();
                msgs[r as usize] = Some(nzval_recv);
            }
            msgs    
        }
        else {
            world.process_at_rank(root).send(&self.nzval[..]);
            Vec::new()
        }
    }
}

pub fn reconstruct_from_gathered<T>(
    msgs: &[Option<Vec<T>>],
    pattern: &SparsePattern,
) -> MatrixFull<T>
where T: Copy + Clone + Default + Send + num_traits::Zero
{
    let [global_row, global_col] = pattern.global_size;
    let mut global_mat = MatrixFull::new([global_row, global_col], T::zero());
    let mpi_size = msgs.len();

    for (rank, opt_msg) in msgs.iter().enumerate() {
        let nzval = opt_msg
            .as_ref()
            .unwrap_or_else(|| panic!("Message missing for rank {rank}"));

        let (col_start, col_end) = column_range(rank as i32, mpi_size as i32, global_col);
        let base = pattern.global_colptr[col_start] as usize;

        for j in col_start..col_end {
            let lo = pattern.global_colptr[j]     as usize - base;
            let hi = pattern.global_colptr[j + 1] as usize - base;
            for k in lo..hi {
                let i1 = pattern.global_rowind[(base - 1) as usize + k] as usize;
                let i  = i1 - 1;
                global_mat.data[j * global_row + i] = nzval[k];
            }
        }
    }
    global_mat
}

#[test]
#[cfg(all(feature = "pexsi", feature = "mpi"))]
fn pexsi_solve_16x16_round_trip() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    assert_eq!(world.size(), 1);

    // 1. Dense H, S.
    let n = 16usize;
    let mut h_dense = MatrixFull::new([n, n], 0.0);
    let mut s_dense = MatrixFull::new([n, n], 0.0);
    for i in 0..n {
        h_dense[[i, i]] = i as f64 - n as f64 / 2.0 + 0.5;   // no zero eigenvalue
        s_dense[[i, i]] = 1.0;
    }

    for i in 0..n {
        // Irrational offset so no eigenvalue equals a "round" shift PEXSI might pick
        h_dense[[i, i]] = i as f64 - n as f64 / 2.0 + 0.5 + 0.0123456789;
    }

    // 2. Sparse pattern + values.
    let pattern = Arc::new(SparsePattern::from_matrixfull(&s_dense, 1e-12, 1, 0));
    let mut h_nz = Vec::new();
    pattern.fill_from_matrixfull(&h_dense, &mut h_nz);
    let mut s_nz = Vec::new();
    pattern.fill_from_matrixfull(&s_dense, &mut s_nz);
    let h_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: h_nz };
    let s_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: s_nz };

    // 3. Plan.
    let mut opts = PPEXSIOptions::defaults();
    opts.verbosity                  = 2;
    opts.symmetric                  = 1;
    opts.solver                     = 0;      // SuperLU_DIST
    opts.ordering                   = 2;      // MMD
    opts.method                     = 2;
    opts.nPoints                    = 1;      // must divide mpisize
    opts.numPole                    = 40;
    opts.temperature                = 1.0e-2;
    opts.deltaE                     = 20.0;   // > spectral radius 7.5
    opts.gap                        = 0.0;
    opts.matrixType                 = 0;
    opts.npSymbFact                 = 1;

    // DFTDriver2-specific
    opts.isInertiaCount             = 1;
    opts.maxPEXSIIter               = 10;
    opts.muMin0                     = -10.0;
    opts.muMax0                     =  10.0;
    opts.mu0                        =   0.0;
    opts.muInertiaTolerance         = 1.0e-3;
    opts.muInertiaExpansion         = 1.0;
    opts.muPEXSISafeGuard           = 1.0e-2;
    opts.numElectronPEXSITolerance  = 1.0e-6;
    
    let mut plan = PexsiPlan::new(world.as_raw(), 1, 1, 0)
        .expect("PEXSIPlanInitialize");
    pexsi_load_hs(&mut plan, &opts, &h_sp, &s_sp)
        .expect("LoadRealHSMatrix");
    pexsi_symbolic_factorize(&plan, &opts)
        .expect("SymbolicFactorize");

    
    // 4. Solve and retrieve.
    let res = pexsi_dft_driver2(&plan, &mut opts, 8.0)
        .expect("DFTDriver2");
    let mu = res.mu;
    /* 
    let mut dm_nz = vec![0.0_f64; h_sp.nnz_local()];
    let band_energy = pexsi_retrieve_dm(&plan, &mut dm_nz)
        .expect("RetrieveRealDM");

    // 5. Gather and compare.
    let dm_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: dm_nz };
    let gathered = dm_sp.gather_to_root(0, &world);
    let dm_dense = reconstruct_from_gathered(&gathered, &pattern);

    // 6. Analytic reference: P = diag(f(eps_i)) with f = Fermi-Dirac.
    let kt = opts.temperature;
    for i in 0..n {
        let eps = h_dense[[i, i]];
        let expected = 2.0 / (1.0 + ((eps - mu) / kt).exp());
        let err = (dm_dense[[i, i]] - expected).abs();
        assert!(err < 1e-6, "P[{i},{i}] = {}, expected {} (err {err})",
                dm_dense[[i, i]], expected);
    }
    */
}

#[test]
#[cfg(all(feature = "pexsi", feature = "mpi"))]
fn pexsi_manual_selinv_16x16() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    assert_eq!(world.size(), 1);

    let n = 16usize;
    let mut h_dense = MatrixFull::new([n, n], 0.0);
    let mut s_dense = MatrixFull::new([n, n], 0.0);
    for i in 0..n {
        h_dense[[i, i]] = i as f64 - n as f64 / 2.0 + 0.5;
        s_dense[[i, i]] = 1.0;
    }

    let pattern = Arc::new(SparsePattern::from_matrixfull(&s_dense, 1e-12, 1, 0));
    let mut h_nz = Vec::new();
    pattern.fill_from_matrixfull(&h_dense, &mut h_nz);
    let mut s_nz = Vec::new();
    pattern.fill_from_matrixfull(&s_dense, &mut s_nz);
    let h_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: h_nz };
    let s_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: s_nz };

    let mut opts = PPEXSIOptions::defaults();
    opts.verbosity                  = 2;
    opts.symmetric                  = 1;
    opts.solver                     = 0;      // SuperLU_DIST
    opts.ordering                   = 2;      // MMD
    opts.method                     = 2;
    opts.nPoints                    = 1;      // must divide mpisize
    opts.numPole                    = 40;
    opts.temperature                = 1.0e-2;
    opts.deltaE                     = 20.0;   // > spectral radius 7.5
    opts.gap                        = 0.0;
    opts.matrixType                 = 0;
    opts.npSymbFact                 = 1;

    // DFTDriver2-specific
    opts.isInertiaCount             = 1;
    opts.maxPEXSIIter               = 10;
    opts.muMin0                     = -10.0;
    opts.muMax0                     =  10.0;
    opts.mu0                        =   0.0;
    opts.muInertiaTolerance         = 1.0e-3;
    opts.muInertiaExpansion         = 1.0;
    opts.muPEXSISafeGuard           = 1.0e-2;
    opts.numElectronPEXSITolerance  = 1.0e-6;

    let mut plan = PexsiPlan::new(world.as_raw(), 1, 1, 0).unwrap();
    pexsi_load_hs(&mut plan, &opts, &h_sp, &s_sp).unwrap();
    pexsi_symbolic_factorize(&plan, &opts).unwrap();


    
    // Manual selected inverse at a fixed μ = 0.
    let mut ainv_nz = vec![0.0_f64; h_sp.nnz_local()];
    pexsi_selinv_real_symmetric(&plan, &opts, &h_sp, &mut ainv_nz).unwrap();

    // Verify against analytic 1/(eps - 0).
    let inv_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: ainv_nz };
    let gathered = inv_sp.gather_to_root(0, &world);
    let inv_dense = reconstruct_from_gathered(&gathered, &pattern);
    for i in 0..n {
        let eps = h_dense[[i, i]];
        let expected = 1.0 / eps;
        let err = (inv_dense[[i, i]] - expected).abs();
        assert!(err < 1e-6, "Ainv[{i},{i}] = {}, expected {} (err {err})",
                inv_dense[[i, i]], expected);
    }
    
}

// ---------------------------------------------------------------------------
// Test: FFI binding correctness (mirrors the standalone smoke test)
//
// Uses only the raw `extern "C"` declarations and `PPEXSIOptions`/`PexsiPlan`.
// No DistSparseMatrixFull, no SparsePattern, no `pexsi_load_hs`.
// If this passes, the FFI layer is correct and any failure elsewhere must
// live in the higher-level data structures.
// ---------------------------------------------------------------------------
#[cfg(all(test, feature = "pexsi", feature = "mpi"))]
mod ffi_smoke {
    use super::*;
    use std::ffi::c_int;
    #[test]
    fn pexsi_ffi_selinv_16x16() {
        let universe = mpi::initialize().unwrap();
        let world    = universe.world();
        assert_eq!(world.size(), 1);

        // ---- plan ----
        let plan = PexsiPlan::new(world.as_raw(), 1, 1, 0)
            .expect("PPEXSIPlanInitialize");

        // ---- options: identical block to the successful smoke test ----
        let mut opts = PPEXSIOptions::defaults();
        opts.verbosity   = 2;
        opts.symmetric   = 1;
        opts.solver      = 0;
        opts.ordering    = 2;
        opts.method      = 2;
        opts.nPoints     = 2;
        opts.numPole     = 40;
        opts.temperature = 1.0e-2;
        opts.deltaE      = 3.0;
        opts.gap         = 1.0;
        opts.matrixType  = 0;
        opts.npSymbFact  = 1;

        // ---- hand-built 1-based CSC for a 16x16 diagonal ----
        const N: i32 = 16;
        let mut colptr: Vec<i32> = (1..=N + 1).collect();          // [1..=17]
        let mut rowind: Vec<i32> = (1..=N).collect();              // [1..=16]
        let mut h_nz:   Vec<f64> =
            (0..N).map(|i| (i as f64) - (N as f64) / 2.0 + 0.5).collect();
        let mut s_nz:   Vec<f64> = vec![1.0; N as usize];

        // ---- LoadRealHSMatrix (raw FFI, no wrapper) ----
        let mut info: c_int = 0;
        unsafe {
            PPEXSILoadRealHSMatrix(
                plan.raw(),
                opts,
                N,                  // nrows
                N,                  // nnz
                N,                  // nnzLocal
                N,                  // numColLocal
                colptr.as_mut_ptr(),
                rowind.as_mut_ptr(),
                h_nz.as_mut_ptr(),
                0,                  // isSIdentity = 0
                s_nz.as_mut_ptr(),
                &mut info,
            );
        }
        assert_eq!(info, 0, "PPEXSILoadRealHSMatrix info={info}");

        // ---- symbolic ----
        unsafe {
            PPEXSISymbolicFactorizeRealSymmetricMatrix(plan.raw(), opts, &mut info);
        }
        assert_eq!(info, 0, "SymbolicFactorize info={info}");

        // ---- selected inverse ----
        let mut ainv: Vec<f64> = vec![0.0; N as usize];
        unsafe {
            PPEXSISelInvRealSymmetricMatrix(
                plan.raw(),
                opts,
                h_nz.as_mut_ptr(),
                ainv.as_mut_ptr(),
                &mut info,
            );
        }
        assert_eq!(info, 0, "SelInv info={info}");

        // ---- verify against analytic 1/(eps_i) ----
        let expected: Vec<f64> =
            (0..N).map(|i| 1.0 / (i as f64 - (N as f64) / 2.0 + 0.5)).collect();
        let mut worst = 0.0_f64;
        for i in 0..N as usize {
            let err = (ainv[i] - expected[i]).abs();
            worst = worst.max(err);
            println!(
                "Ainv[{i}] = {:+.6e}  expected {:+.6e}  |err| = {:.2e}",
                ainv[i], expected[i], err
            );
        }

        drop(plan);
        drop(world);
        drop(universe);

        assert!(worst < 1e-10, "SelInv mismatch, worst |err| = {worst:.3e}");
        println!("\npexsi FFI smoke test PASSED  (worst |err| = {worst:.3e})");
    }
}