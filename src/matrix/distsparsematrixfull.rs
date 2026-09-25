// rest_tensors/src/matrix/distsparsematrixfull.rs
use crate::matrix::{
    BasicMatrix, BasicMatrixOpt, MatFormat, MathMatrix, MatrixFull, ParMathMatrix,
};
use crate::matrix::matrixupper::*;
use crate::matrix::matrix_blas_lapack::{_dgemm_full, _hamiltonian_fast_solver, _get_sqrt_and_inv_sqrt};

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

pub struct LocalSparseSlice<T> {
    pub col_range: Vec<usize>,
    pub colptr: Vec<i32>,
    pub rowind: Vec<i32>,
    pub nzval: Vec<T>,
}

pub struct DistSparseMatrixFull<T> {
    /// Global matrix shape `[nrows, ncols]`.
    pub global_size: [usize; 2],

    /// Total nnz over all ranks. Only used for sanity checks and reporting.
    pub global_nnz: usize,

    /// pexsi expects a 1d block distribution on col
    pub col_range: (usize, usize),
    
    /// Local CSC storage. Length `num_col_local + 1`. **0-based**, monotone
    /// non-decreasing, `colptr[0] == 0`, `colptr[num_col_local] == rowind.len()`.
    pub colptr: Vec<i32>,

    /// Local row indices (global). **0-based**. Length `nnz_local`.
    pub rowind: Vec<i32>,

    /// Local nonzero values. Length `nnz_local`.
    pub nzval: Vec<T>,
}

impl<T> DistSparseMatrixFull<T> {
    #[inline] pub fn nnz_local(&self)       -> usize { self.rowind.len() }
    #[inline] pub fn num_col_local(&self)   -> usize { self.col_range.1 - self.col_range.0 }
    #[inline] pub fn nrows(&self)           -> usize { self.global_size[0] }
    #[inline] pub fn ncols(&self)           -> usize { self.global_size[1] }
}

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

impl<T> DistSparseMatrixFull<T>
where T: Copy + Clone + Default + Send + Buffer + 'static + mpi::datatype::Equivalence, f64: From<T>,
{
    pub fn from_matrixfull(
        mat: &MatrixFull<T>,
        global_nnz: usize,
        eps: f64, 
        npcol: usize, 
        mycol: usize,
    ) -> Self {
        let [nrows, ncols] = mat.size();
        let (start, end) = column_range(mycol as i32, npcol as i32, ncols);
        let local_cols = end - start;
        let mut colptr: Vec<i32> = Vec::with_capacity(local_cols + 1);
        let mut rowind: Vec<i32> = Vec::new();
        let mut nzval:  Vec<T>   = Vec::new();

        colptr.push(0);

        for j in start..end {
            for i in 0..nrows {
                let v = mat.data[j * nrows + i];
                let abs: f64 = Into::<f64>::into(v).abs();
                if abs > eps {
                    rowind.push(i as i32);
                    nzval.push(v);
                }
            }
            colptr.push(rowind.len() as i32);
        }

        Self {
            global_size: [nrows, ncols],
            global_nnz,
            col_range: (start, end),
            colptr, 
            rowind,
            nzval,
        }
    }

    pub fn from_matrixupper(
        mat: &MatrixUpper<T>,
        global_nnz: usize,
        eps: f64,
        npcol : usize,
        mycol: usize,
    ) -> Self {
        let mat_full = mat.to_matrixfull().unwrap();
        let result = Self::from_matrixfull(&mat_full, global_nnz, eps, npcol, mycol);
        result
    }

    pub fn gather_to_root(&self, root: i32, world: &SimpleCommunicator) -> Vec<Option<LocalSparseSlice<T>>> {
        let rank = world.rank();
        let size = world.size();
        let mut msgs = Vec::with_capacity(size as usize);

        if rank == root {
            
            let own_msg = LocalSparseSlice {
                col_range: [self.col_range.0.clone(), self.col_range.1.clone()].to_vec(),
                colptr: self.colptr.clone(),
                rowind: self.rowind.clone(),
                nzval: self.nzval.clone(),
            };
            msgs.push(Some(own_msg));

            for r in 0..size {
                let (col_range_recv, _) = world.process_at_rank(r).receive_vec::<usize>();
                let (colptr_recv, _) = world.process_at_rank(r).receive_vec::<i32>();
                let (rowind_recv, _) = world.process_at_rank(r).receive_vec::<i32>();
                let (nzval_recv, _) = world.process_at_rank(r).receive_vec::<T>();
                let msg = LocalSparseSlice {
                    col_range: col_range_recv,
                    colptr: colptr_recv,
                    rowind: rowind_recv,
                    nzval: nzval_recv,
                };
                msgs.push(Some(msg));
            }
            msgs    
        }
        else {
            let col_range_send = [self.col_range.0.clone(), self.col_range.1.clone()].to_vec();
            world.process_at_rank(root).send(&col_range_send[..]);
            world.process_at_rank(root).send(&self.colptr[..]);
            world.process_at_rank(root).send(&self.rowind[..]);
            world.process_at_rank(root).send(&self.nzval[..]);
            Vec::new()
        }
    }
}

pub fn reconstruct_from_gathered<T>(msgs: &Vec<Option<LocalSparseSlice<T>>>, global_row: usize, global_col: usize) -> MatrixFull<T>
where T: Copy + Clone + Default + Send + num_traits::Zero
{
    let mut global_mat = MatrixFull::new([global_row, global_col], T::zero());
    for (rank, msg) in msgs.iter().enumerate() {
        let msg_ctx = msg.as_ref().expect("Message missing for rank {rank}");
        let col_start = msg_ctx.col_range[0].clone();
        let col_end = msg_ctx.col_range[1].clone();
        let colptr = msg_ctx.colptr.clone();
        let rowind = msg_ctx.rowind.clone();
        let nzval = msg_ctx.nzval.clone();

        let mut tmp: usize = 0;
        for j in col_start..col_end {
            let rowind_start = colptr[tmp];
            let rowind_end = colptr[tmp + 1];
            for k in rowind_start..rowind_end {
                let k_u: usize = k as usize;
                let i = rowind[k_u] as usize;
                global_mat.data[j * global_row + i] = nzval[k_u];
            }
            tmp += 1;
        }
    }
    global_mat
}