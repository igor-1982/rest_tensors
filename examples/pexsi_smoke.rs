use mpi::raw::AsRaw;
#[cfg(feature = "pexsi")]
fn main() {
    use mpi::traits::Communicator;
    use rest_tensors::pexsi::{self, PPEXSIOptions, Plan};

    let universe = mpi::initialize().expect("MPI_Init failed");
    let world    = universe.world();
    let comm_raw: mpi::ffi::MPI_Comm = world.as_raw();
    let rank     = world.rank();

    let plan = Plan::new(comm_raw, 1, 1, rank)
        .expect("PPEXSIPlanInitialize failed");

    let mut opts: PPEXSIOptions = PPEXSIOptions::with_defaults();
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
    opts.npSymbFact = 1;

    const N: i32 = 16;
    let mut colptr: Vec<i32> = (1..=N+1).collect();  // [1, 2, ..., 17]
    let mut rowind: Vec<i32> = (1..=N).collect();    // [1, 2, ..., 16]
    let mut anz: Vec<f64> = (0..N).map(|i| (i as f64) - (N as f64)/2.0 + 0.5).collect();
    let mut snz:    Vec<f64> = vec![1.0; N as usize];
    let is_s_identity: i32 = 0;

    let mut info: i32 = 0;
    
    unsafe {
        pexsi::PPEXSILoadRealHSMatrix(
            plan.raw(), opts,
            N,    // nrows
            N,    // nnz
            N,    // nnzLocal
            N,    // numColLocal  ← "all columns local" sentinel
            colptr.as_mut_ptr(), rowind.as_mut_ptr(),
            anz.as_mut_ptr(), is_s_identity, snz.as_mut_ptr(),
            &mut info,
        );
    }
    assert_eq!(info, 0, "PPEXSILoadRealHSMatrix info={info}");

    unsafe { pexsi::PPEXSISymbolicFactorizeRealSymmetricMatrix(plan.raw(), opts, &mut info); }
    assert_eq!(info, 0, "SymbolicFactorize info={info}");

    let mut ainv: Vec<f64> = vec![0.0; N as usize];
    unsafe {
        pexsi::PPEXSISelInvRealSymmetricMatrix(
            plan.raw(), opts, anz.as_mut_ptr(), ainv.as_mut_ptr(), &mut info,
        );
    }
    assert_eq!(info, 0, "SelInv info={info}");

    let expected: Vec<f64> =
        (0..N).map(|i| 1.0 / (i as f64 - N as f64/2.0 + 0.5)).collect();
    let mut worst = 0.0_f64;
    for i in 0..N as usize {
        let err = (ainv[i] - expected[i]).abs();
        worst = worst.max(err);
        println!("Ainv[{i}] = {:+.6e}  expected {:+.6e}  |err| = {:.2e}",
                 ainv[i], expected[i], err);
    }

    drop(plan); drop(world); drop(universe);

    assert!(worst < 1e-10, "SelInv mismatch, worst |err| = {worst:.3e}");
    println!("\npexsi smoke test PASSED  (worst |err| = {worst:.3e})");
}