// rest_tensors/src/matrix/distsparsematrixfull.rs
//!
//! Provides [`SparsePattern`] (immutable CSC structure + 1D column-block
//! distribution) and [`DistSparseMatrixFull`] (pattern + values) for use with
//! the PEXSI solver. The distribution follows PEXSI's
//! `PPEXSILoadRealHSMatrix` convention, which is **1D column-block**: rank
//! `r` of `npcol` owns global columns `[start_r, end_r)` with the standard
//! balanced split.
//!
//! ## Distribution model
//!
//! PEXSI's C API expects each rank to supply a contiguous range of columns
//! in CSC format. Rows are global (not distributed). The partition is:
//!
//! ```text
//! base = ncols / npcol
//! rem  = ncols % npcol
//! rank r owns columns [start_r, end_r) where
//!   start_r = r < rem ? r*(base+1) : rem*(base+1) + (r-rem)*base
//!   end_r   = start_r + (r < rem ? base+1 : base)
//! ```
//!
//! Every rank also builds the **full global pattern** (`global_colptr`,
//! `global_rowind`) so that gathering requires no pattern metadata on the
//! wire — only values. This costs `O(nnz_global)` memory per rank, which is
//! well within budget for N up to ~10⁴. Above that, a root-only pattern cache
//! would be the next optimization.
//!
//! ## 1-based CSC convention
//!
//! Despite the `int*` signature in `c_pexsi_interface.h`, PEXSI v2.0.0
//! requires **1-based** column and row indices:
//! - `colptr[0] == 1`
//! - `colptr[ncols] == nnz + 1`
//! - every `rowind[k] >= 1`
//!
//! This contradicts the obvious reading of the C header and matches the text
//! file format (`lap2dc_real.matrix`) used by PEXSI's reference driver.
//! `SparsePattern::from_matrixfull` produces arrays in this convention;
//! `local_colptr`/`local_rowind` are passed to the FFI verbatim.
//!
//! Note the internal `global_colptr`/`global_rowind` also carry 1-based
//! indices, so the arithmetic in `reconstruct_from_gathered` includes a
//! `(base - 1)` offset when indexing into the global arrays. This is the
//! only place where the two conventions (0-based Rust iteration vs.
//! 1-based PEXSI storage) have to be reconciled — do not "simplify" it.
//!
//! ## PEXSI v2.0.0 constraints (learned empirically, 2026-10)
//!
//! These are not documented in the PEXSI header or tutorial; each cost a day
//! of debugging. Record them so the next integrator doesn't re-discover them.
//!
//! ### (1) `nrows` must be a multiple of the process grid size
//!
//! SuperLU_DIST's internal block-cyclic redistribution requires
//! `nrows % (numProcRow * numProcCol) == 0`. Confirmed:
//!
//! | n    | npcol | result |
//! |------|-------|--------|
//! | 2048 | 1,2,4 | ok     |
//! | 2048 | 5,6   | fail   |
//! | 2040 | 5,6   | ok     |
//!
//! Failure mode depends on ordering:
//! - `ordering = 0` (ParMETIS): `ParSymbFact encounters zero diagonal`
//! - `ordering = 2` (MMD): `MPI_Alltoallv` truncated in `Distribute()`
//!
//! **Production code must pad the basis to a multiple of the effective
//! rank count.** See `PexsiContext` in the SCF module for the padding
//! scheme (identity S, `H_pad = c·I` with `c` above the spectrum).
//!

//! ### (3) Exactly-diagonal patterns trigger distinct SuperLU_DIST failures
//!
//! For N < ~128, `RealSuperLUData::NumericalFactorize` returns `info = N`
//! (zero pivot at the last column) on an edgeless graph. For N ≥ 128 the
//! same pattern succeeds (verified at N=2048). Real H/S from a Gaussian
//! basis always have off-diagonal support, so this is synthetic-test-only.
//!
//! ### (5) `isSIdentity = 1` does not omit `SnzvalLocal`
//!
//! The header's doc comment says the S array is optional when S is the
//! identity. The v2.0.0 implementation still requires a valid array of
//! length `nnzLocal`. Always pass `isSIdentity = 0` with real S values.
//!
//! ### (6) `nPoints × numProcRow × numProcCol` must divide `mpisize`
//!
//! Checked inside `DFTDriver2`; not checked by any of the lower-level
//! entry points. With a single rank, `nPoints` must be 1. With multiple
//! ranks, it may be any divisor of `mpisize / (numProcRow * numProcCol)`.
//!
//! ## Test inventory
//!
//! | test | what it verifies |
//! |------|------------------|
//! | [`pexsi_manual_selinv_2048_multirank`] | end-to-end manual `SelInv` path; requires `mpisize` to divide 2048 |
//! | [`pexsi_banded_workmatrix_multirank`] | sparse banded input with realistic row/column sparsity; requires `mpisize` to divide 2040 |
//!
//! ## History
//!
//! Written 2026-09 during the initial PEXSI integration into `rest_tensors`.
//! The `SparsePattern` / `DistSparseMatrixFull` split was introduced so that
//! the pattern (immutable during SCF) can be cached while only the values
//! change per iteration. See `matrix_pexsi.rs` for the FFI layer.
//! 
use std::sync::Arc;
use crate::matrix::{
    BasicMatrix, BasicMatrixOpt, MatFormat, MathMatrix, MatrixFull, ParMathMatrix,
};
use crate::matrix::matrix_blas_lapack::{_dgemm_full};
use crate::matrix::matrixupper::*;
use crate::matrix::matrix_pexsi::*;

#[cfg(feature = "mpi")]
use mpi::topology::{SimpleCommunicator};
#[cfg(feature = "mpi")]
use mpi::traits::*;
#[cfg(feature = "mpi")]
use mpi::datatype::*;

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

/// Immutable structure of the sparse matrix: shape, distribution, CSC pattern.
/// Built once per SCF calculation. Shared (via Arc or &) by every value-bearing
/// view of H, S, and DM (when their patterns coincide).
pub struct SparsePattern {
    /// Global shape, same on every rank.
    pub global_size: [usize; 2],
    pub global_nnz:  usize,

    /// Full global CSC pattern, identical on every rank.
    /// `global_colptr[0] = 1`, `global_colptr[ncols] = global_nnz`.
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

        debug_assert_eq!(global_colptr[0], 1, "PEXSI CSC must be 1-based");
        debug_assert_eq!(
            *global_colptr.last().unwrap(),
            global_nnz as i32 + 1,
            "global_colptr[ncols] must equal nnz_global + 1"
        );
        debug_assert_eq!(local_colptr[0], 1, "local_colptr must be 1-based");
        debug_assert_eq!(
            *local_colptr.last().unwrap(),
            local_rowind.len() as i32 + 1,
            "local_colptr[n] must equal nnz_local + 1"
        );
        debug_assert!(global_rowind.iter().all(|&i| i >= 1 && (i as usize) <= nrows));
    
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

impl SparsePattern {
    /// Maximum absolute difference between two dense reconstructions,
    /// restricted to entries that lie inside this pattern.
    ///
    /// `a_pexsi_dense` is the dense reconstruction of the selected inverse
    /// returned by SelInv; `a_ref_dense` is the exact inverse from LAPACK.
    /// Off-pattern entries are ignored: SelInv does not compute them, so
    /// comparing them would test the pattern, not the solver.
    ///
    /// Returns `(max_abs_diff, (i, j))`.
    pub fn max_on_pattern_diff(
        &self,
        a_pexsi_dense: &MatrixFull<f64>,
        a_ref_dense:   &MatrixFull<f64>,
    ) -> (f64, (usize, usize)) {
        let [n, m] = self.global_size;
        assert_eq!(n, m, "pattern must be square");
        assert_eq!(a_pexsi_dense.size(), [n, n]);
        assert_eq!(a_ref_dense.size(),   [n, n]);

        let mut worst   = 0.0_f64;
        let mut worst_ij = (0usize, 0usize);

        // global_colptr / global_rowind use PEXSI's 1-based convention.
        for j in 0..n {
            let col_lo = (self.global_colptr[j]     - 1) as usize;
            let col_hi = (self.global_colptr[j + 1] - 1) as usize;
            for k in col_lo..col_hi {
                let i = (self.global_rowind[k] - 1) as usize;
                let diff = (a_pexsi_dense[[i, j]] - a_ref_dense[[i, j]]).abs();
                if diff > worst {
                    worst = diff;
                    worst_ij = (i, j);
                }
            }
        }
        (worst, worst_ij)
    }

    /// Maximum magnitude of the reference inverse outside this pattern.
    /// A diagnostic for pattern adequacy: if this is much smaller than the
    /// on-pattern magnitude, the pattern captures the physics and SelInv's
    /// truncation error is negligible.
    pub fn max_off_pattern_abs(
        &self,
        a_ref_dense: &MatrixFull<f64>,
    ) -> f64 {
        let [n, m] = self.global_size;
        assert_eq!(n, m, "pattern must be square");
        assert_eq!(a_ref_dense.size(), [n, n]);

        let mut worst = 0.0_f64;

        for j in 0..n {
            let col_lo = (self.global_colptr[j]     - 1) as usize;
            let col_hi = (self.global_colptr[j + 1] - 1) as usize;

            // Collect row indices of this column's on-pattern entries.
            // Linear scan is fine here: the test is one-shot, not hot.
            for i in 0..n {
                let on_pattern = self.global_rowind[col_lo..col_hi]
                    .iter()
                    .any(|&r| (r - 1) as usize == i);
                if !on_pattern {
                    let v = a_ref_dense[[i, j]].abs();
                    if v > worst { worst = v; }
                }
            }
        }
        worst
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
fn pexsi_solve_round_trip() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let npcol = world.size() as i32;
    let mycol = world.rank() as i32;

    // 1. Dense H, S.
    let n = 2048usize;
    let mut h_dense = MatrixFull::new([n, n], 0.0);
    let mut s_dense = MatrixFull::new([n, n], 0.0);
    
    for i in 0..n {
        h_dense[[i, i]] = i as f64 - n as f64 / 2.0 + 0.5;   // no zero eigenvalue
        s_dense[[i, i]] = 1.0;
        /* 
        if i + 1 < n {
            // Tridiagonal coupling — enough to give MMD a non-trivial graph
            h_dense[[i, i + 1]] = 0.1;
            h_dense[[i + 1, i]] = 0.1;
            s_dense[[i, i + 1]] = 0.05;
            s_dense[[i + 1, i]] = 0.05;
        }
        */
    }

    // 2. Sparse pattern + values.
    let pattern = Arc::new(SparsePattern::from_matrixfull(&s_dense, 1e-12, npcol, mycol));
    let mut h_nz = Vec::new();
    pattern.fill_from_matrixfull(&h_dense, &mut h_nz);
    let mut s_nz = Vec::new();
    pattern.fill_from_matrixfull(&s_dense, &mut s_nz);
    let h_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: h_nz };
    let s_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: s_nz };
    
    // need initial mu from H and S
    let mut trace_h = 0.0_f64;
    let mut trace_s = 0.0_f64;
    for i in 0..n {
        trace_h += h_dense[[i, i]];
        trace_s += s_dense[[i, i]];
    }
    let mu_guess = trace_h / trace_s;          
    let half_width = 20.0; 

    // 3. Plan.
    let mut opts = PPEXSIOptions::defaults();
    opts.verbosity                  = 2;
    opts.symmetric                  = 1;
    opts.solver                     = 0;        // SuperLU_DIST
    opts.ordering                   = 2;        
    opts.method                     = 2;        // Moussa
    opts.nPoints                    = 1;        // ← REQUIRED, must divide mpisize
    opts.numPole                    = 60;
    opts.temperature                = 0.05;
    opts.deltaE                     = 2500.0;   // > spectral radius ~1024
    opts.gap                        = 0.0;
    opts.matrixType                 = 0;
    opts.npSymbFact                 = 1;
    opts.isSymbolicFactorize        = 0;
    
    // DFTDriver2-specific
    opts.isInertiaCount             = 1;
    opts.maxPEXSIIter               = 20;       // ← was 10; more iterations for a 2048-wide spectrum
    opts.muMin0                     = mu_guess - half_width;
    opts.muMax0                     = mu_guess + half_width;
    opts.mu0                        = mu_guess;
    opts.muInertiaTolerance         = 1.0e-2;   // ← will be clamped to 2.0 by PEXSI
    opts.muInertiaExpansion         = 1.0;
    opts.muPEXSISafeGuard           = 1.0e-2;
    opts.numElectronPEXSITolerance  = 1.0e-6;
    
    let mut plan = PexsiPlan::new(world.as_raw(), 1, npcol, mycol)
        .expect("PEXSIPlanInitialize");
    pexsi_load_hs(&mut plan, &opts, &h_sp, &s_sp)
        .expect("LoadRealHSMatrix");
    pexsi_symbolic_factorize(&plan, &opts)
        .expect("SymbolicFactorize");

    
    // 4. Solve and retrieve.
    let num_electrons = 1024.0;
    let res = pexsi_dft_driver2(&plan, &mut opts, num_electrons).expect("DFTDriver2");
    let mu = res.mu;
    println!("mu = {mu:.6e}  (raw return from PEXSI)");
    
    // (a) self-consistency of the returned mu
    let kt = opts.temperature;
    let mut ne_check = 0.0_f64;
    for i in 0..n {
        let eps = h_dense[[i, i]];
        ne_check += 1.0 / (1.0 + ((eps - mu) / kt).exp());
    }
    let ne_err = (ne_check - num_electrons).abs();
    println!("N(mu) = {ne_check:.6}, |N - N_e| = {ne_err:.3e}");
    assert!(ne_err < 1e-2, "electron count off by {ne_err:.3e}");
    
    // (b) DM consistency with the same mu
    let mut dm_nz = vec![0.0_f64; h_sp.nnz_local()];
    let _band_energy = pexsi_retrieve_dm(&plan, &mut dm_nz).expect("RetrieveRealDM");
    let dm_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: dm_nz };
    let gathered = dm_sp.gather_to_root(0, &world);
    let dm_dense = reconstruct_from_gathered(&gathered, &pattern);
    if mycol == 0 {
        let mut worst = 0.0_f64;
        for i in 0..n {
            let eps = h_dense[[i, i]];
            let expected = 1.0 / (1.0 + ((eps - mu) / kt).exp());
            worst = worst.max((dm_dense[[i, i]] - expected).abs());
        }
        println!("worst |P - P_ref| = {worst:.3e}");
        assert!(worst < 1e-3, "DM mismatch");
    }
    
    // (c) optional sanity: mu should be within a few kT of 0
    assert!(mu.abs() < 5.0 * kt, "mu = {mu} far from expected zero");
    
}

// PEXSI v2.0.0's PPEXSIDFTDriver2 aborts inside SuperLU_DIST's
// NumericalFactorize on exactly-diagonal H/S patterns when nrows < ~128.
// The failure is not reproduced at larger N (empirically verified at
// N = 2048) nor on any pattern with at least one off-diagonal entry,
// which is what every real REST calculation produces. The threshold
// appears to correlate with an internal small-matrix code path in
// SuperLU_DIST; the exact cutoff is not documented.
//
#[test]
#[cfg(all(feature = "pexsi", feature = "mpi"))]
fn pexsi_manual_selinv_2048_multirank() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let n = 2048usize;
    let npcol = world.size() as i32;
    let mycol = world.rank() as i32;

    // Constraint (1): nrows must be a multiple of the process grid size.
    // 2048 = 2^11, so this test only runs at power-of-2 mpisize.
    if !(2048usize % npcol as usize == 0) {
        println!("skipping: mpisize={npcol} does not divide 2048");
        return;
    }

    // Every rank builds the same dense H and S. The pattern construction
    // slices out this rank's column block.
    let mut h_dense = MatrixFull::new([n, n], 0.0);
    let mut s_dense = MatrixFull::new([n, n], 0.0);

    for i in 0..n {
        h_dense[[i, i]] = i as f64 - n as f64 / 2.0 + 0.5;
        s_dense[[i, i]] = 1.0;
        if i + 1 < n {
            h_dense[[i, i+1]] = 0.01;
            h_dense[[i+1, i]] = 0.01;
            s_dense[[i, i+1]] = 0.001;
            s_dense[[i+1, i]] = 0.001;
        }
    }

    let pattern = Arc::new(SparsePattern::from_matrixfull(&s_dense, 1e-12, npcol, mycol));
    let mut h_nz = Vec::new();
    pattern.fill_from_matrixfull(&h_dense, &mut h_nz);
    let mut s_nz = Vec::new();
    pattern.fill_from_matrixfull(&s_dense, &mut s_nz);
    let h_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: h_nz };
    let s_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: s_nz };

    let mut opts = PPEXSIOptions::defaults();
    opts.verbosity   = 2;
    opts.symmetric   = 1;
    opts.solver      = 0; //SUPER_LU
    opts.ordering    = 0; //PARMETIS
    opts.method      = 2;
    //opts.nPoints     = 1;
    opts.numPole     = 40;
    opts.temperature = 1.0e-2;
    opts.deltaE      = 2500.0;   // > spectral radius ~1024
    opts.gap         = 0.0;
    opts.matrixType  = 0;
    opts.npSymbFact  = npcol;

    let mut plan = PexsiPlan::new(world.as_raw(), 1, npcol, mycol).unwrap();
    pexsi_load_hs(&mut plan, &opts, &h_sp, &s_sp).unwrap();
    pexsi_symbolic_factorize(&plan, &opts).unwrap();

    let mut ainv_nz = vec![0.0_f64; h_sp.nnz_local()];
    pexsi_selinv_real_symmetric(&plan, &opts, &h_sp, &mut ainv_nz).unwrap();

    let inv_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: ainv_nz };
    let gathered = inv_sp.gather_to_root(0, &world);
    if world.rank() == 0 {
        let inv_dense = reconstruct_from_gathered(&gathered, &pattern);
        let mut worst = 0.0_f64;
        for i in 0..n {
            let eps = h_dense[[i, i]];
            let expected = 1.0 / eps;
            let err = (inv_dense[[i, i]] - expected).abs();
            worst = worst.max(err);
        }
        println!("worst |err| = {worst:.3e}");
        assert!(worst < 1e-10);
    }
}

#[test]
#[cfg(all(feature = "pexsi", feature = "mpi"))]
fn pexsi_banded_workmatrix_multirank() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let npcol = world.size() as i32;
    let mycol = world.rank() as i32;

    let n = 2048usize;
    let bandwidth = 32usize;  
    // Constraint (1): nrows must be a multiple of the process grid size.
    // 2048 = 2^11, so this test only runs at power-of-2 mpisize.
    if !(n % npcol as usize == 0) {
        println!("skipping: mpisize={npcol} does not divide {}", n);
        return;
    }

    let mut h_dense = MatrixFull::new([n, n], 0.0);
    let mut s_dense = MatrixFull::new([n, n], 0.0);

    for i in 0..n {
        // Shift by 0.5 so no diagonal entry is exactly zero (matrix nonsingular)
        h_dense[[i, i]] = (i as f64 - n as f64 / 2.0 + 0.5) * 0.5;
        s_dense[[i, i]] = 1.0;
    
        for d in 1..=bandwidth {
            if i + d >= n { break; }
            let decay = (-(d as f64).powi(2) / 50.0).exp();
            let h_val = decay * 0.1;
            let s_val = decay * 0.05;
            h_dense[[i, i + d]] = h_val;
            h_dense[[i + d, i]] = h_val;
            s_dense[[i, i + d]] = s_val;
            s_dense[[i + d, i]] = s_val;
        }
    }

    let pattern = Arc::new(SparsePattern::from_matrixfull(&s_dense, 1e-12, npcol, mycol));
    let mut h_nz = Vec::new();
    pattern.fill_from_matrixfull(&h_dense, &mut h_nz);
    let mut s_nz = Vec::new();
    pattern.fill_from_matrixfull(&s_dense, &mut s_nz);
    let h_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: h_nz.clone() };
    let s_sp = DistSparseMatrixFull { pattern: pattern.clone(), nzval: s_nz };

    if mycol == 0 {
        println!(
            "npcol={} n={} bandwidth={} nnz_local_rank0={} nnz_global={}",
            npcol, n, bandwidth, h_sp.nnz_local(), pattern.global_nnz
        );
    }

    let mut opts = PPEXSIOptions::defaults();
    opts.verbosity   = 1;
    opts.symmetric   = 1;
    opts.solver      = 0;      // SuperLU_DIST
    opts.ordering    = 2;      
    opts.method      = 2;
    //opts.nPoints     = 1;
    opts.numPole     = 60;
    opts.temperature = 1.0e-2;
    opts.deltaE      = 600.0;  // > spectral radius (~512)
    opts.gap         = 0.0;
    opts.matrixType  = 0;
    opts.npSymbFact  = 1;

    let mut plan = PexsiPlan::new(world.as_raw(), 1, npcol, mycol)
        .expect("PEXSIPlanInitialize");
    pexsi_load_hs(&mut plan, &opts, &h_sp, &s_sp)
        .expect("LoadRealHSMatrix");
    pexsi_symbolic_factorize(&plan, &opts)
        .expect("SymbolicFactorize");

    let mut ainv_nz = vec![0.0_f64; h_sp.nnz_local()];
    pexsi_selinv_real_symmetric(&plan, &opts, &h_sp, &mut ainv_nz)
        .expect("SelInvRealSymmetric");

    if mycol == 0 {
        println!("npcol={} -> SelInv SUCCESS", npcol);
    }
    
    let inv_sp = DistSparseMatrixFull {
        pattern: pattern.clone(),
        nzval:   ainv_nz.clone(),
    };
    let a_gathered   = h_sp.gather_to_root(0, &world);
    let inv_gathered = inv_sp.gather_to_root(0, &world);
    
    if mycol == 0 {
        let a_dense   = reconstruct_from_gathered(&a_gathered,   &pattern);
        let inv_dense = reconstruct_from_gathered(&inv_gathered, &pattern);
        let mut dmin = f64::INFINITY;
        let mut dmax = 0.0_f64;
        for i in 0..n {
            let d = inv_dense[[i, i]].abs();
            if d < dmin { dmin = d; }
            if d > dmax { dmax = d; }
        }
        println!("diag(A⁻¹): min={:.3e} max={:.3e}", dmin, dmax);
        assert!(dmax < 1e6, "diag(A⁻¹) has magnitude {:.3e}, matrix likely singular", dmax);

        // r = A * A^{-1}  (dense n x n)
        let mut r = MatrixFull::new([n, n], 0.0);
        _dgemm_full(&a_dense, 'N', &inv_dense, 'N', &mut r, 1.0, 0.0);

        // For every (i, j) in the pattern, compare r[i, j] to δ_ij.
        // Off-pattern entries of the true A^{-1} are unknown to SelInv,
        // so we only check entries on the pattern.
        let (start, end) = pattern.col_range;  // rank 0 owns all columns when npcol == 1
        // Only check on-pattern entries (|i - j| <= bandwidth)
        let mut worst = 0.0_f64;
        let mut worst_ij = (0, 0);
        for j in 0..n {
            let i_lo = j.saturating_sub(bandwidth);
            let i_hi = (j + bandwidth + 1).min(n);
            for i in i_lo..i_hi {
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (r[[i, j]] - expected).abs();
                if err > worst { worst = err; worst_ij = (i, j); }
            }
        }
        println!("max |A·A⁻¹ - I| on pattern (band) = {:.3e} at {:?}", worst, worst_ij);
        assert!(worst < 1.0e-6, "SelInv on-pattern residual too large: {:.3e}", worst);
        
    }
    
}
