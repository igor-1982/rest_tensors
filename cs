[33mcommit 354b6fadf9ff87c71797f605193ac2a6ca0e13e7[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mfeature/pexsi[m[33m, [m[1;31morigin/feature/pexsi[m[33m)[m
Author: yangqian <16210220043@fudan.edu.cn>
Date:   Wed Oct 14 23:40:03 2026 +0800

    FIX: DFTDRIVER2 back on track

[33mcommit 027825016ec45ff32a47f9638d7488ce00b3fa0a[m
Author: yangqian <16210220043@fudan.edu.cn>
Date:   Wed Oct 14 21:43:14 2026 +0800

    dftdriver2 crash on h2o20

[33mcommit 155f650c383556377365c4e084abeccad111e295[m
Author: yangqian <16210220043@fudan.edu.cn>
Date:   Tue Oct 13 20:02:10 2026 +0800

    add numerical stability test on 1CRN

[33mcommit 6278cdcab43dfe5809dcf48deac211f4c966ec82[m
Author: yangqian <16210220043@fudan.edu.cn>
Date:   Thu Oct 8 21:59:44 2026 +0800

    chore: remove /examples from cargo.toml

[33mcommit 24814f897ec41a0efc9d795c761b0b84195df65c[m
Author: yangqian <16210220043@fudan.edu.cn>
Date:   Thu Oct 8 20:19:50 2026 +0800

    chore: remove examples directory

[33mcommit 05110ce0304347b10afae424217bce5c4c13544c[m
Author: yangqian <16210220043@fudan.edu.cn>
Date:   Thu Oct 8 20:18:25 2026 +0800

    add mpisize safety check to workmatrix test

[33mcommit 7bef00d439f3805886b0f4400de442e06435115d[m
Author: yangqian <16210220043@fudan.edu.cn>
Date:   Thu Oct 8 17:08:10 2026 +0800

    DEBUG: DFTDRIVER2, SELINV, MAT SIZE

[33mcommit 81768ed488f145ec79d046395ae940ca083f69e8[m
Author: yangqian <16210220043@fudan.edu.cn>
Date:   Tue Sep 29 16:19:50 2026 +0800

    corrected dftdriver test (diag H/S fails)

[33mcommit 323f8043089707218f8e7a3d6d4914ec59d88e07[m
Author: yangqian <16210220043@fudan.edu.cn>
Date:   Tue Sep 29 15:44:09 2026 +0800

    PEXSI ffi bindings complete

[33mcommit 527cbac0b87fca3180098de11149b6acfd57fce6[m
Author: yangqian <16210220043@fudan.edu.cn>
Date:   Fri Sep 25 23:03:09 2026 +0800

    tmp work on distsparsematrixfull.rs

[33mcommit 9f1bd707506d2b52ecf1ff5163656d644d40d24d[m
Author: yangqian <16210220043@fudan.edu.cn>
Date:   Thu Sep 24 00:02:26 2026 +0800

    WIP: pexsi unfinished work

[33mcommit 7cbaeeff203ad1151b8dc6ea740be2825bd42cbf[m
Author: 钱洋 <16210220043@fudan.edu.cn>
Date:   Wed Sep 23 15:24:16 2026 +0800

    FIX: datatype i8 mismatch on aarch64

[33mcommit fbea339493d4a6d3acd47bb36445ed6f16a87731[m[33m ([m[1;32mmaster[m[33m)[m
Merge: 31c0126 1f2aa75
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Thu Sep 17 15:22:24 2026 +0000

    !21 Fix: 依 OpenBLAS v0.3.34 线程行为更新修该 `omp_set_num_threads_global_wrapper`
    
    Merge pull request !21 from Zhenyu Zhu ajz34/260916/fix-openblas-0.3.34

[33mcommit 1f2aa75860a79ef3d17f65a5be1df9e33e426c80[m
Author: ajz34 <ajz34@outlook.com>
Date:   Wed Sep 16 21:01:16 2026 +0800

    fix: adapt `omp_set_num_threads_global_wrapper` to OpenBLAS v0.3.34 change of threading

[33mcommit 31c012677fabb96ed855a0f2ca9367fa5f69ca4b[m
Merge: b96c778 ff9f032
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Tue Sep 15 14:50:27 2026 +0000

    !20 Add zheev, zheevd and zhegv to complexmatrixfull.rs
    
    Merge pull request !20 from jp zou/feat/x2c-spinor

[33mcommit ff9f03220c7eb5356872327718e8f2d51d667beb[m
Author: jp-zou <likeat2best@mail.ustc.edu.cn>
Date:   Tue Sep 15 16:52:56 2026 +0800

    Add zheev, zheevd and zhegv to complexmatrixfull.rs

[33mcommit 21035367ad8bb1f3bd5569b23b944d72f3ff2c8e[m
Author: jp-zou <likeat2best@mail.ustc.edu.cn>
Date:   Tue Sep 15 16:16:15 2026 +0800

    Add zheev, zheevd and zhegv to complexmatrixfull.rs

[33mcommit f224242711463705a392a4a3a7bdb28a84467314[m
Author: jp-zou <likeat2best@mail.ustc.edu.cn>
Date:   Tue Sep 15 16:05:54 2026 +0800

    Add zheevd and zhegv to complexmatrixfull.rs

[33mcommit b96c7781780188e0d5bac1cbdcf6b2f3e1881d4d[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Fri Sep 11 03:27:17 2026 +0800

    feat: pdgetri FFI + wrapper (distributed LU inverse for SCS-RPA lambda-integration)

[33mcommit a2f14df6735c4b9ebe589f584bc78c723edb24d5[m
Author: jp-zou <likeat2best@mail.ustc.edu.cn>
Date:   Thu Sep 10 22:22:18 2026 +0800

    feat: complex matrix operations for the two-component X2C spinor SCF
    
    - extend complexmatrixfull with complex BLAS/LAPACK-level operations
      (zgemm-based products, Hermitian assembly) required by the spinor RI-J/K
      and spinor Vxc builders

[33mcommit 022b0fcdbe3c7cd23122e175f663162ff8f264f2[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Mon Sep 7 00:06:41 2026 +0800

    pdgetrf FFI + distributed helpers (pdgetrf_ipiv_len/pd_numroc/pd_trace_local/pdgetrf_local_lnabsdet/pd_neg_shift_local) for distributed dRPA response

[33mcommit 3df708b2e72c2e5b652f5a597140e3756e76dde2[m
Merge: 2cc2ea8 dba8bb0
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Sat Sep 5 15:04:20 2026 +0000

    !17 Modifications for implementation of AC-GW and optimization of CD-GW
    
    Merge pull request !17 from qrgao/master

[33mcommit dba8bb0d38dfcf3c62e4986a4f28a9d82cace8e3[m
Author: qiruigao <gaochenyuen@163.com>
Date:   Sat Sep 5 22:06:06 2026 +0800

    docs: fix stale _qrpinv doc (doc example still used pre-inplace API; upstream doc bug)

[33mcommit 763ddf276b3f73054bd886279514cf21ce3965c7[m
Author: qiruigao <gaochenyuen@163.com>
Date:   Wed Aug 5 20:03:32 2026 +0800

    ri: zero-copy RIFull->MatrixFull (into_matfull_*) and in-place dsyev for GW low-rank path

[33mcommit 5a9d6f124477011fdbfcab2af1173c5d987ccd94[m
Author: Zongye Tang <brook_tang@163.com>
Date:   Tue Jul 21 08:37:02 2026 +0000

    Fixed minor problem in _dsvd encountered in NTO analysis

[33mcommit 8975b54501c4099bc36301003fd81a462a31bbd9[m
Author: qirui-gao-2025 <gaochenyuen@163.com>
Date:   Thu Jul 16 02:11:01 2026 +0000

    Added an in-place _dinverse for G0W0

[33mcommit 2cc2ea82aaea57557d2845b4ff8d590e0979cec0[m
Merge: 46c33b2 4987046
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Sat Sep 5 07:48:35 2026 +0000

    !19 make mpi optional
    
    Merge pull request !19 from Shirong Wang/mpi-optional

[33mcommit 498704612dca17c7769b2455d056253e46fadd10[m
Author: Shirong_Wang <srwang20@fudan.edu.cn>
Date:   Fri Sep 4 21:09:14 2026 +0800

    make mpi optional

[33mcommit 46c33b23a9ab2f697637f1626b78f86d20ab92f4[m
Merge: 31fa2b2 3c41c3d
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Fri Sep 4 11:22:51 2026 +0000

    !18 fix: add new distributed hamiltonian solver to address singularity
    
    Merge pull request !18 from 钱洋/debug-scalapack-bug

[33mcommit 31fa2b21c87081f2eec09ffa1f36bb4631fa489c[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Fri Sep 4 19:20:34 2026 +0800

    IYZ:: pdsygvx 重正交化系数 orfac 修正（1e-3 → 1e-10）

[33mcommit 3c41c3d3fb00ee6f183e99bdcdbd7d3f53e7f396[m
Author: atlaschan1310 <491811899@qq.com>
Date:   Fri Sep 4 14:20:49 2026 +0800

    fix: add new distributed hamiltonian solver to address singularity

[33mcommit 8ee89ab1fd64b3f836b8f0fdaa2f9b7f98fa105d[m
Merge: 5203380 29e6bc7
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Wed Sep 2 08:32:19 2026 +0000

    !16 change-qr: rework QR-based pseudo-inverse (pass x explicitly, refactor…
    
    Merge pull request !16 from DeltaH/change-qr

[33mcommit 5203380939207c880371e9029c4a74f1cb4a62fb[m
Merge: 8d0e65f cd557fa
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Mon Aug 31 07:53:26 2026 +0000

    !15 IYZ: scalapack relevant changes
    
    Merge pull request !15 from 张颖/master

[33mcommit cd557faa8e6cd3996b4b45c418d3d96314f1b03d[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Mon Aug 31 15:47:55 2026 +0800

    IYZ: scalapack relevant changes

[33mcommit 8d0e65ff48d1bc9d8f34765fb69d9b70ee7ebac8[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Aug 30 22:30:54 2026 +0800

    IYZ: expose the dtrtrs triangular solver as _dtrtrs

[33mcommit 7f13b53988752d4d2f9d057fcdf6bd16e1fc1427[m
Merge: 1a69a56 354855e
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Thu Aug 27 16:19:56 2026 +0000

    !14 Add new feature to RIFull: 25d-pt2
    
    Merge pull request !14 from 钱洋/feature/25d-pt2

[33mcommit 354855ec23aae7cd6f740015b38e8e2281d15c33[m
Merge: cb82b0d 1a69a56
Author: 钱洋 <16210220043@fudan.edu.cn>
Date:   Thu Aug 27 12:10:51 2026 +0000

    Merge branch 'master' of gitee.com:restgroup/rest_tensors into feature/25d-pt2
    
    Signed-off-by: 钱洋 <16210220043@fudan.edu.cn>

[33mcommit 1a69a5645f8bc2d135ee398e6672ff3aede5e825[m
Merge: cf5220b 3d6158c
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Thu Aug 27 06:31:28 2026 +0000

    !12 add scalapack support
    
    Merge pull request !12 from 钱洋/scalapack

[33mcommit 29e6bc7ebb63fac4685e0e21ca20a77f4cc53277[m
Author: Delta_H <Delta_H@user.noreply.gitee.com>
Date:   Wed Aug 19 21:12:29 2026 +0800

    change-qr: rework QR-based pseudo-inverse (pass x explicitly, refactor dormqr calls)

[33mcommit cb82b0d255dca46a3f8293008d54a8bfd15467b5[m
Author: atlaschan1310 <491811899@qq.com>
Date:   Sat Aug 15 16:18:25 2026 +0800

    Add new feature to RIFull: 25d-pt2

[33mcommit cf5220b147b78713eaa552df06a942a7150ea409[m
Merge: 1e2a2b8 1a7a8c6
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Fri Aug 14 15:31:57 2026 +0000

    !13 Fix: 修复极为偶发的自洽场数值不稳定
    
    Merge pull request !13 from Zhenyu Zhu ajz34/fix/omp-thread-setter-safety

[33mcommit 1a7a8c692f937273f1013c147e9bd1bf81d7522b[m
Author: ajz34 <ajz34@outlook.com>
Date:   Thu Aug 13 11:38:28 2026 +0800

    enh: add global blas thread initializer

[33mcommit 93cdd246ba56b0f0b3853964e4acc788ef7ac292[m
Author: ajz34 <outlook.com>
Date:   Thu Aug 13 06:58:40 2026 +0800

    fix: omp_set_num_threads_wrapper use more concise language for code change description

[33mcommit cede332d981ceb52e65ae63099bed4c3d11ab1fd[m
Author: ajz34 <outlook.com>
Date:   Wed Aug 12 22:56:23 2026 +0800

    fix: omp_set_num_threads_wrapper should only call omp_set_num_threads
    
    openblas_set_num_threads / goto_set_num_threads (in the OpenMP build,
    blas_server_omp.c) call adjust_thread_buffers(), which does heap alloc/free
    on the GLOBAL shared blas_thread_buffer workspace. Calling them concurrently
    from sibling threads (as REST does via per-rayon-task omp_set_num_threads_
    wrapper(1) calls) races on those globals and corrupts in-flight BLAS results.
    This was the root cause of REST's rare (~1/17000) SCF convergence divergence.
    
    omp_set_num_threads (OpenMP runtime) only sets the thread-local nthrs-var ICV
    -> thread-safe. Since OpenBLAS is built/linked with OpenMP here, its omp
    parallel regions respect omp_set_num_threads, so BLAS threading (and thus
    oversubscription) is still controlled.
    
    Evidence (REST NH3 HF/RI, num_threads=4):
      - master wrapper (all 3 setters): 7/120000 SCF divergent
      - this wrapper (omp only):        0/120000 SCF divergent
      - num_threads=1 (no threads):     deterministic
      - minimal Rust+OpenBLAS reproducer: concurrent openblas_set_num_threads
        corrupts dgemm in seconds; concurrent omp_set_num_threads does not.
      - oversubscription unchanged (CPU 4-7 cores at num_threads=4).
    
    Co-Authored-By: Claude <noreply@anthropic.com>

[33mcommit 3d6158cacbebcf407035a5406a18836bbe4f5b25[m
Author: atlaschan1310 <491811899@qq.com>
Date:   Wed Aug 5 15:13:21 2026 +0800

    add scalapack support

[33mcommit 1e2a2b8e1a9f8c80baf95f366743b1931bfd5aa9[m[33m ([m[1;31morigin/master[m[33m, [m[1;31morigin/HEAD[m[33m)[m
Author: Shirong Wang <srwang20@fudan.edu.cn>
Date:   Thu Jun 4 08:44:24 2026 +0000

    !11 windows support
    
    * link gomp
    * proper cross
    * try link arg
    * import dll.a
    * import dll.a
    * use librestmatr
    * fix
    * for win

[33mcommit 0f6f10efd930b2d798e86bcce0b692102d3846d2[m
Merge: 16cec48 3f5e217
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Thu May 21 06:21:06 2026 +0000

    !10 zyli & zhlin: 修改基于截断QR分解的伪逆求解
    
    Merge pull request !10 from zyli_00/zyli

[33mcommit 3f5e217a46a382592839b6a0d42e4a1c99a0f3fc[m
Author: zyli00 <1432737608@qq.com>
Date:   Wed May 20 23:55:35 2026 +0800

    zyli & zhlin: 添加基于截断QR分解的伪逆求解

[33mcommit 16cec48b204dce656e651a1ab159811b9a16f593[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Wed May 13 14:54:41 2026 +0800

    IYZ: add a new version of pseudo-inversion based on QR factorizatn

[33mcommit 1bbfdd91fb11df93a9b65efa659aa2cffcf71e86[m
Merge: 04bb51b 2beca67
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Sun May 10 03:08:41 2026 +0000

    !9 注释 build.rs 中 rerun-if-changed=librestmatr.so
    
    Merge pull request !9 from Zhenyu Zhu ajz34/try-accelerate-rerun-if-changed

[33mcommit 2beca67c0b5d016eae3c1289c1300031b080c6b2[m
Author: ajz34 <ajz34@outlook.com>
Date:   Fri May 8 18:18:41 2026 +0800

    comment out rerun-if-changed on shared library binary

[33mcommit 04bb51bd6968f1d32929c2430f1aad021fe8f3e8[m
Merge: ba6c8b2 8f6fbf5
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Sat May 2 15:47:49 2026 +0000

    !8 fix warnings
    
    Merge pull request !8 from Shirong Wang/fixwarn

[33mcommit 8f6fbf5c1ea5f3fdafdb0eb7a3970054acb9b4b9[m
Author: Shirong_Wang <srwang20@fudan.edu.cn>
Date:   Fri May 1 21:47:30 2026 +0800

    address warnings

[33mcommit ba6c8b28a5a1112c715f4bb0272f6c12971fba57[m
Merge: 5c510d9 51d5c93
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Thu Apr 23 05:06:56 2026 +0000

    !7 Merge branch 'master' into _dsolve
    
    Merge pull request !7 from qrgao/master

[33mcommit 51d5c9361591477ab4fa77ac4aaa78313a0dbb9f[m
Merge: 0529b21 5c510d9
Author: qirui-gao-2025 <gaochenyuen@163.com>
Date:   Thu Apr 23 03:31:57 2026 +0000

    Merge branch 'master' into _dsolve
    dsolve is added as a linear equation solver.

[33mcommit 5c510d94f489d64761015c3c3c058b138eac390a[m
Author: Shirong Wang <srwang20@fudan.edu.cn>
Date:   Thu Apr 16 01:47:31 2026 +0000

    !6 for mac support
    
    * clean
    * link libomp
    * cfg linux for libc imports

[33mcommit 0529b21a3b3cad92898ed91d1cde1d0a5cb3927f[m
Author: qirui-gao-2025 <gaochenyuen@163.com>
Date:   Tue Apr 14 02:08:56 2026 +0000

    added linear equations solver _dsolve

[33mcommit 5ca3dba301041c2c2df4d5a8b0c5c36dddda5797[m
Merge: c5a844f e3e9c6a
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Tue Mar 24 15:33:46 2026 +0000

    !5 添加了返回值完整的LU分解lapack_dgetrf_full
    
    Merge pull request !5 from aI2o333/master

[33mcommit e3e9c6a8f110997829e4099c6bfe1958d1d8cd67[m
Author: aI2o333 <msy_bshh@163.com>
Date:   Sun Mar 22 23:49:01 2026 +0800

    为了使用LU分解修改了部分函数所有权

[33mcommit c5a844f532324d8ccfe68df694983fd946a543f1[m[33m ([m[1;33mtag: [m[1;33mv2025.02[m[33m)[m
Merge: d5e83d3 6cc846b
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Sat Dec 13 16:12:23 2025 +0000

    !4 allow using F90 or FC env var
    Merge pull request !4 from Shirong Wang/add-f90

[33mcommit 6cc846bd99f2d21bc8bb379a588f2ddcca2d0c46[m
Author: Shirong_Wang <srwang20@fudan.edu.cn>
Date:   Sat Dec 13 22:13:13 2025 +0800

    allow using F90 or FC

[33mcommit d5e83d3461da3de8aed119b910b3e44209c64ec1[m
Merge: fee9b6e 9a43b0a
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Sun Sep 21 15:23:00 2025 +0000

    !3 Fixed: Add complex MatrixFull
    Merge pull request !3 from DeltaH/feature/complex_matrix_full

[33mcommit 9a43b0a8b7a6f3627e48d9c5e2c4617935df89cb[m
Author: DeltaH <12386037+delta_h@user.noreply.gitee.com>
Date:   Fri Sep 19 09:41:58 2025 +0000

    Fixed: Add complex MatrixFull

[33mcommit 80fa4603c682a7a5e302bd6d7adba9f63dc877ca[m
Author: DeltaH <12386037+delta_h@user.noreply.gitee.com>
Date:   Thu Sep 18 16:48:29 2025 +0000

    Add ComplexMatrixFull

[33mcommit fee9b6e2de66ab45caf9c1bb6eac44d7edaf232b[m[33m ([m[1;33mtag: [m[1;33mv2025.01[m[33m)[m
Merge: d04bf46 1519fcb
Author: 张颖 <igor_zhangying@fudan.edu.cn>
Date:   Thu Sep 11 12:17:28 2025 +0000

    !1 added dgeev
    Merge pull request !1 from qrgao/master

[33mcommit 1519fcbb44112f3ce3aaff9c2aadf7919cc22132[m
Author: qirui-gao-2025 <gaochenyuen@163.com>
Date:   Wed Sep 10 10:19:32 2025 +0000

    added dgeev

[33mcommit d04bf461392573d5c8ae4742961de30f74c81464[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Mon Jul 21 00:05:11 2025 +0800

    IYZ:: fix a bug of using ilp64 for mkl incorrectly

[33mcommit 31765e61ae0829ee560b7442be3c500d1c48b665[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Jul 20 18:04:55 2025 +0800

    IYZ: a slight change to make openblas as the default feature

[33mcommit d80c1a5f3a6b6f0696ae0f847090d0e1b5102aab[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Jul 20 17:14:31 2025 +0800

    IYZ: leave feature=openblas as the the default feature

[33mcommit e1f9826915be0d9832d0322e24ab03113a7ac62e[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Jul 20 15:37:45 2025 +0800

    IYZ: add features of "openblas" and "intel-mkl"

[33mcommit 6de8a9fe84f473d48d623ed5bd42398ea322ada3[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Fri Mar 28 16:43:29 2025 +0800

    IYZ: fix a bug in contract_vxc_0_serial. NOTE that, use None (not none) for Option<f64>

[33mcommit 88bcd56918985c26482a7d868ae565dfec89fb63[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Tue Mar 18 16:45:47 2025 +0800

    IYZ: slightly change the trait of BasicMatrix

[33mcommit b226064e5f1937bedb7aa47816f423d9da076727[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Tue Jan 21 13:09:55 2025 +0800

    IYZ: fix a bug in _hamiltonian_fast_solver

[33mcommit 2a9fa9c2424f46fe63e8325e4975086f0fdb51df[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sat Jan 18 16:06:04 2025 +0800

    IYZ: add a fast algorithm to solve the hamiltonian problem
    '_hamiltonian_fast_solver'

[33mcommit 2381c07f3801e02957c832b8a96faf6a20ce2c2f[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Wed Oct 9 23:11:31 2024 +0800

    IYZ: fix some lifetime warning messages.

[33mcommit 69862164277843a3c3faccb596f62840e60ad6ae[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Fri Sep 27 16:26:52 2024 +0800

    IYZ

[33mcommit aa4c7befef1b0c5637ced5971e61fc84b6cde859[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Fri Sep 27 15:35:44 2024 +0800

    IYZ: fix a bug for the compilation of librestmatr.so

[33mcommit c7ddd2cccad72e431185951d5196ef839d6a390e[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Thu Sep 26 00:09:48 2024 +0800

    IYZ: add `push_column` to struct MatrixFull

[33mcommit 29e51669b91d9c992859014ad33a897f57c90fff[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Wed Aug 21 16:46:03 2024 +0800

    IYZ

[33mcommit e8514deddf0b756b6379619fe47d3b3a28a7755a[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Wed May 15 20:37:16 2024 +0800

    IYZ

[33mcommit 52e8cb576978a37baf325b471d1f44755784e91e[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Wed May 15 00:46:29 2024 +0800

    IYZ: fix a bug for matrixfull.iter_matrixupper_shift

[33mcommit a20cd73f19558ab177200da07069ded42cf217f8[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Tue May 14 00:23:36 2024 +0800

    IYZ

[33mcommit e566c27bb4a2186cfc51fae614e5ee3e565f3a80[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sat May 11 14:13:51 2024 +0800

    IYZ

[33mcommit 7a911e70170ea19db3c56dba9c219e34af964f38[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sat May 11 14:13:04 2024 +0800

    IYZ: unfinished job to include submatrix iteration functions for
    matrixupper and matrixfullslice and so on

[33mcommit 28081046952a53bbcb577f6734c1d42b54787d92[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Wed May 8 09:15:28 2024 +0800

    IYZ

[33mcommit 74b74080e815cbcf611b4d8de7eb991f241a5643[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Fri May 3 11:14:00 2024 +0800

    IYZ:

[33mcommit 41eb439fb4517c5a86ac6170280f7ab20e99e85a[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Thu May 2 22:15:01 2024 +0800

    IYZ: add the BasicMatUp trait

[33mcommit 0df70179d04c0d49ca3bf24d71d4722c070a2d2a[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Fri Nov 24 20:36:51 2023 +0800

    IYZ: some experimental test for numerical solution of inverse square
    root

[33mcommit 697e7a9c46ab81c7e63631d23aaca56c6f5b96a0[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Wed Nov 15 00:06:33 2023 +0800

    IYZ

[33mcommit 917ea4a49b3e59536ba7c86de4091fdb88cd4e63[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Thu Nov 2 00:15:05 2023 +0800

    IYZ: fix some bugs

[33mcommit 94ec150a160279e0f51f440ec291f502e3d7efbf[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Tue Oct 31 23:58:23 2023 +0800

    IYZ: add _dgemv for the product of matrix and vector

[33mcommit 1277e5cdb76acb4ab40c63615fae5c8d2516d2fb[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Tue Oct 31 00:39:17 2023 +0800

    IYZ: add dsyrk for C = alpha*A*A**T + beta*C and C = alpha*A**T*A + beta*C

[33mcommit 6314c50e4dc65e2ad2d0159a60af8174b8579ff5[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Fri Sep 22 14:53:21 2023 +0800

    IYZ

[33mcommit f2dd4c7c62a05c91446c83218d0efc0ec60ed705[m
Merge: 7d92e16 4f1992d
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Wed Sep 20 16:55:30 2023 +0800

    Merge branch 'zyli' of github.com:igor-1982/rest_tensors

[33mcommit 4f1992da1551736cebeae6ea1444818b6a23e2cc[m
Author: zyli <1432737608@qq.com>
Date:   Wed Sep 20 16:53:19 2023 +0800

    zyli: 1) change pinv to a faster version

[33mcommit 7d92e16f626a92057e8fa4b8760173fb6607cf2a[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Wed Sep 20 13:34:36 2023 +0800

    IYZ:: a simple fix to allow MatrixFull::new() to handle the generic
    variable "T" that is not bounded by the Copy trait

[33mcommit f3bc623cb38940255033703d1d0fa908e6802484[m
Author: tygao <tygao@fudan.edu.cn>
Date:   Tue Jul 25 21:44:06 2023 +0800

    test version: for int3c2e_ip function test only

[33mcommit 0ba627ad3f303622ecab3a5deea924b3d38a8c56[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Wed Jun 21 17:11:04 2023 +0800

    IYZ:: slight change

[33mcommit 0a7b2ebff8ecb2bc1b2cd20d4b8c57a90ae29be3[m
Merge: b4f8288 263c4dc
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Wed Jun 21 17:09:23 2023 +0800

    Merge branch 'tygao' of github.com:igor-1982/rest_tensors

[33mcommit b4f8288a625591291d897a48477d09bb0c947d1f[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Wed Jun 21 17:07:07 2023 +0800

    IYZ: add einsum mod

[33mcommit 263c4dcac2fda59672955e3f61903ff958b4e0f0[m
Merge: 1eb7e45 e4aecf1
Author: tygao <tygao@fudan.edu.cn>
Date:   Wed Jun 21 17:06:23 2023 +0800

    Merge branch 'master' of github.com:igor-1982/rest_tensors into tygao

[33mcommit 1eb7e45d77436206b8933c23fe6b8dcbc9056d06[m
Author: tygao <tygao@fudan.edu.cn>
Date:   Wed Jun 21 17:05:00 2023 +0800

    slight changes

[33mcommit e4aecf1225a233264a657d7bf562af9a767d0786[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Fri Jun 2 15:28:25 2023 +0800

    IYZ: add a new blas wrapper _dgemm_full using openblas directly

[33mcommit 5dd2ac1faac91bb5216ccfc0b4e9a412769081dd[m
Merge: 4a7c704 7013401
Author: tygao <tygao@fudan.edu.cn>
Date:   Mon May 8 21:17:03 2023 +0800

    Merge branch 'master' of github.com:igor-1982/rest_tensors into tygao

[33mcommit 4a7c704d27a76b6cf34f39dbbe07d147f6c2a968[m
Merge: c91d713 44fe9b1
Author: tygao <tygao@fudan.edu.cn>
Date:   Mon May 8 20:56:08 2023 +0800

    Merge branch 'master' of github.com:igor-1982/rest_tensors into tygao
    
    Conflicts:
            src/matrix/matrixfull.rs

[33mcommit c91d7130a794216ff8a068e56f7a95be8e8fa5c3[m
Author: tygao <tygao@fudan.edu.cn>
Date:   Mon May 8 20:48:09 2023 +0800

    fix formated_output_e function; add formated_output_e_with_threshold

[33mcommit 701340128f6c9e3664b72dbe022f76521debc2df[m
Merge: 560dfe7 44fe9b1
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun May 7 20:30:21 2023 +0800

    Merge remote-tracking branch 'github/master'

[33mcommit 560dfe7a571a510463c1e9a79cbd98198ca5f2b7[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun May 7 20:28:26 2023 +0800

    IYZ

[33mcommit 44fe9b1e6f858641d1a9a15916bf9fdad628a39a[m
Merge: 00503f2 1453ff5
Author: igor-1982 <34899511+igor-1982@users.noreply.github.com>
Date:   Thu May 4 20:02:47 2023 +0800

    Merge pull request #1 from hebrewsnabla/stab-dft
    
    davidson update

[33mcommit 1453ff5a9ef5a91cd08f88d9bc32779b0b4ed154[m
Author: jeanwsr <srwang20@fudan.edu.cn>
Date:   Thu Apr 20 21:37:33 2023 +0800

    davidson: support nroots > 1; set default nroots=3

[33mcommit a904e17ce50b55cb93e63b2d2199ef099bb12ec6[m
Author: jeanwsr <srwang20@fudan.edu.cn>
Date:   Tue Apr 18 22:25:58 2023 +0800

    davidson: misc adjust

[33mcommit 58a52a1ffb245d5b0c9587c327b559a7f5bcc73c[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Thu Apr 13 15:49:08 2023 +0800

    IYZ

[33mcommit 00503f2f006c17ea622f52d34b7deac814bc08a6[m
Merge: cc45f5f 0891787
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Fri Apr 7 15:07:49 2023 +0800

    Merge branch 'zyli'

[33mcommit 0891787cd93d2bc3f3fdcebecf29a164eb33d008[m
Author: zyli <1432737608@qq.com>
Date:   Fri Apr 7 15:04:41 2023 +0800

    zyli:   1) add pinv and pseudo_inverse

[33mcommit cc45f5f735de65864b33cd3c55c96d3b03ba6f77[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Fri Apr 7 14:55:33 2023 +0800

    IYZ

[33mcommit 092734a75c8be2dd96edaac10b33d46f9c9e3c90[m
Merge: b96b76a 367d17d
Author: tygao <tygao@fudan.edu.cn>
Date:   Fri Apr 7 12:55:23 2023 +0800

    Merge branch 'master' of github.com:igor-1982/rest_tensors

[33mcommit 367d17df3537dfa4b54834f2664a32100345a57e[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Tue Mar 28 14:40:44 2023 +0800

    IYZ: explicitly add "rayon" and "serial" to the _eisum methods

[33mcommit 7d28dd657acc5b8486ffd31b433aa5fc9c0f8953[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Mon Mar 27 14:07:00 2023 +0800

    IYZ

[33mcommit b96b76ac65d5da126e2066b50f45865b5d05d5d6[m
Merge: d48872e 00b6fca
Author: tygao <tygao@fudan.edu.cn>
Date:   Thu Mar 23 21:41:07 2023 +0800

    Merge branch 'master' of github.com:igor-1982/rest_tensors into main

[33mcommit 00b6fca38a4089eb5cc92c7390e805985027a9c4[m
Merge: b792927 6a045ac
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Thu Mar 23 16:32:42 2023 +0800

    Merge branch 'zyli'

[33mcommit 6a045aca2f14e4a490a704430be37f58b729c2aa[m
Author: zyli <1432737608@qq.com>
Date:   Thu Mar 23 16:15:06 2023 +0800

    zyli:   1)add lapack_dgesv

[33mcommit b79292719226c7226051ea8ac0de6527222342f4[m
Merge: e05daa3 931b669
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Thu Mar 23 15:38:23 2023 +0800

    Merge branch 'tygao2'

[33mcommit 931b669288e7c1abe97b6ef27cf178c019a95a84[m
Author: tygao <tygao@fudan.edu.cn>
Date:   Thu Mar 23 15:32:14 2023 +0800

    slight change

[33mcommit e05daa32c72ed930ba9fbb92aae72c6c1d8fdcf6[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Thu Mar 23 15:17:18 2023 +0800

    IYZ

[33mcommit d91d033ec8923738f810633405747404759bae5a[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Tue Mar 21 22:36:54 2023 +0800

    IYZ

[33mcommit c7744d91e85d770459e59059de3deaeb31ab5f5a[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Tue Mar 21 22:33:53 2023 +0800

    IYZ:: add _dinverse to the lapack library

[33mcommit 2e380bb5946eba2061f6735954ee5565387fccbd[m
Merge: fc6a44f 99a138e
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Mon Mar 20 17:07:32 2023 +0800

    Merge branch 'master' of /share/repo/REST/fdqc-tensors.mirror

[33mcommit fc6a44f4a51fb46b4daa186469e619b1ae1c1be1[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Mon Mar 20 17:07:12 2023 +0800

    IYZ

[33mcommit d48872e013929230fa88a0bec23dc673df62200f[m
Merge: be90564 99a138e
Author: igor-1982 <34899511+igor-1982@users.noreply.github.com>
Date:   Sun Mar 19 22:51:01 2023 +0800

    Merge pull request #1 from igor-1982/master
    
    Master

[33mcommit 99a138ecbd2eafd70a4e4c94444655b1324b81c3[m
Merge: d8fa79f be90564
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 22:48:08 2023 +0800

    Merge remote branch 'github/main'

[33mcommit d8fa79fcf4798c52bf51e6deb397aad765acf792[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 22:46:39 2023 +0800

    IYZ

[33mcommit 234f917b57840f548392950f5704668460b8d087[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 22:40:08 2023 +0800

    IYZ

[33mcommit be905644b210e43a5179c7e4feb584bc4d0dfaf0[m
Author: igor-1982 <34899511+igor-1982@users.noreply.github.com>
Date:   Sun Mar 19 22:39:16 2023 +0800

    Update README.md

[33mcommit bf3cf42f4376c9885f61b5030427e75c45f94a6b[m
Author: igor-1982 <34899511+igor-1982@users.noreply.github.com>
Date:   Sun Mar 19 22:37:09 2023 +0800

    Update README.md

[33mcommit 89e300a859ab2bfe3894e6cf091f1dff9822cdbd[m
Author: igor-1982 <34899511+igor-1982@users.noreply.github.com>
Date:   Sun Mar 19 22:34:39 2023 +0800

    Initial commit

[33mcommit 987672340fa3bb17799b19d6b80d46d9c0d115b2[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 22:05:24 2023 +0800

    IYZ

[33mcommit 4ca95d728ed544f208cde5abe5d4655c9ddabd48[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 21:55:55 2023 +0800

    IYZ

[33mcommit 37cba85346f8ec8dbb53f9e56bd265c43bdfc86d[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 21:43:30 2023 +0800

    IYZ

[33mcommit b40990d2668d3a9b53fe398ddcc68c9056d733a2[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 21:42:10 2023 +0800

    IYZ

[33mcommit e7677ee03b4fe8521cd4a913ea5c9e0c7037c9c4[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 21:41:31 2023 +0800

    IYZ

[33mcommit 7ed2f852b8efa791bb696a8a8b43c17c68cd81e9[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 21:40:35 2023 +0800

    IYZ

[33mcommit 285e7cdf7e5dfbc605dfa0c843830430c4bb53f1[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 21:29:25 2023 +0800

    IYZ

[33mcommit 1592ee3f5053d174476bfe6fb909ad24cd876452[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 21:27:12 2023 +0800

    IYZ

[33mcommit eef03fedcea2b2429a8c20889dd989778d1f5bfc[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 21:26:12 2023 +0800

    IYZ: slight change

[33mcommit c65ba84c61868b7cfa611143874bd60b0fd9a37c[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 20:23:28 2023 +0800

    IYZ: complete the information in Cargo.toml

[33mcommit dc7e8fdd21e01f0741c1cdf852bafdbaccc0d76e[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Sun Mar 19 20:12:27 2023 +0800

    IYZ: add more detailed usage for Matrix

[33mcommit 272f68855be6977aa23dc3e92a6469f9a3aff591[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Fri Mar 17 15:45:26 2023 +0800

    IYZ:: 1) re-organize API of matrixfull and relevant 2d-tensors
          2) add a detailed usage for matrixfull
          3) add several traits for matrixfull, including Index, Add and other math operators, and Iter for matrix

[33mcommit 34016c4d5b8681052814dea09b11fa1b437ef99d[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Tue Mar 7 15:01:22 2023 +0800

    IYZ:: update the API for MatrixFull and its variants again

[33mcommit 23b0549bf58d30885b85e86d6772a2f5703ab3ed[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Mon Mar 6 16:30:41 2023 +0800

    IYZ:: 1) update API mainly for the MatrixFull struct, and put all relevant codes to a new mod: "matrix"
          2) build a fortran library "src/external_libs/restmatr.f90" for some efficient lapack and blas functions
          3) update the build.rs script to automatically generate the library of librestmatr.so
          4) provide a detailed usage of MatrixFull in the rustdoc webpage

[33mcommit d8922235cf7ece5c32b5abf53e199679a6cf7dec[m
Merge: 69f1f58 7a48fca
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Wed Feb 22 16:43:15 2023 +0800

    Merge branch 'master' of /share/repo/REST/fdqc-tensors.mirror

[33mcommit 69f1f5806daf97923b059e43154da9e111cac236[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Wed Feb 22 16:42:03 2023 +0800

    IYZ:: 1) add benches; 2) introduce fortran code to have an efficienct ao2mo operation for RIFull

[33mcommit 7a48fca6270fec1d1ac34f8ab12283ed25f50b33[m
Merge: 5c5c0db 3eeffa5
Author: hebrewsnabla <srwang20@fudan.edu.cn>
Date:   Tue Feb 7 15:21:35 2023 +0800

    Merge branch 'master' of yogalab:~/REST/fdqc-tensors

[33mcommit 5c5c0dbe002270f745a6d6716f2eca34ef1ee009[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Sat Feb 4 20:22:39 2023 +0800

    IYZ:: change the get_slices_mut to the old version

[33mcommit 3eeffa5ed62850718b3bfdf45d68a9917ea0d4f3[m
Author: jeanwsr <srwang20@fudan.edu.cn>
Date:   Sat Feb 4 16:02:37 2023 +0800

    davison-patch

[33mcommit dd519da5629d5bef295d913348fa9d8291705412[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Sat Feb 4 13:44:57 2023 +0800

    IYZ: update according to some new functions in fdqc

[33mcommit 1d954011e167fe29e52fbf1bcc1c48ec8cf04ae2[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Tue Jan 17 15:11:25 2023 +0800

    IYZ:

[33mcommit 189dd7ac9165113f1a44b89259812340352d2c57[m
Author: jeanwsr <srwang20@fudan.edu.cn>
Date:   Tue Sep 20 15:48:38 2022 +0800

    add davidson solver (still not work)

[33mcommit 1217f1949e8a047fe6f14fba0a037cd1b6a354fe[m
Author: jeanwsr <srwang20@fudan.edu.cn>
Date:   Mon Aug 22 22:24:06 2022 +0800

    impl diag for MatrixFull; draft for Davidson solver

[33mcommit 5e38c4c9b3d2ecf0a69e312d6fd094139dba839b[m
Author: igor-1982 <ygxu1982@hotmail.com>
Date:   Mon Aug 15 01:33:08 2022 +0800

    IYZ:: add some einsum functionalities

[33mcommit 4e164f840eadd69601fec1577b584cecccfdc090[m
Author: igor-1982 <ygxu1982@hotmail.com>
Date:   Fri Aug 12 00:58:50 2022 +0800

    IYZ:: some new functionalities

[33mcommit cc82a492ef1f5ca2a2a563ac361bb782c0b02c61[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Thu Jun 23 23:23:52 2022 +0800

    IYZ:: slight adjustment for a better friendly installation

[33mcommit 61e5c45735ae34f74146624bf691896cdd9880db[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Thu Jun 23 22:55:53 2022 +0800

    IYZ:: replace blas-src by openblas-src directly for a static link to the blas library

[33mcommit 1172cd416d26a404e348741584e459ce311f0b7d[m
Author: Igor@xhpc1 <igor_zhangying@fudan.edu.cn>
Date:   Thu Jun 23 13:49:27 2022 +0800

    IYZ

[33mcommit 8435a90963aed1eeceb3032462bc91d0b8f81ce0[m
Author: igor-1982 <ygxu1982@hotmail.com>
Date:   Wed Jun 22 23:22:30 2022 +0800

    IYZ:: add RIFull for 3-Rank tensor

[33mcommit 58f5f008d5c55215143d276dbd33b08b85b842d7[m
Author: Igor Ying Zhang <igor_zhangying@fudan.edu.cn>
Date:   Mon Nov 8 14:57:31 2021 +0800

    IYZ: the first version of fdqc-tensors
         complete the basic tensor operations for the fdqc package
