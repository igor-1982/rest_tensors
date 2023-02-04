//extern crate nalgebra as na;
//
//use eigenvalues::algorithms::davidson::Davidson;
//use eigenvalues::utils::generate_diagonal_dominant;
//use eigenvalues::{DavidsonCorrection, SpectrumTarget};
//use nalgebra::{DMatrix, DMatrixSlice, DVector, DVectorSlice};
//
//pub fn davidson(Ax: Box<dyn FnMut>) -> (DVector<f64>, DMatrix<f64>) {
//    impl MatrixOperations for DMatrix<f64> {
//        fn matrix_vector_prod(&self, vs: DVectorSlice<f64>) -> DVector<f64> {
//            Ax(vs)
//        }
//    }
//    let arr = generate_diagonal_dominant(10, 0.005);
//    //let eig = sort_eigenpairs(nalgebra::linalg::SymmetricEigen::new(arr.clone()), true);
//    let spectrum_target = SpectrumTarget::Lowest;
//    let tolerance = 1.0e-4;
//
//    let dav = Davidson::new(
//        arr.clone(),
//        2,
//        DavidsonCorrection::DPR,
//        spectrum_target.clone(),
//        tolerance,
//    )
//    .unwrap();
//    (dav.eigenvalues, dav.eigenvectors)
//}
use std::default::Default;
use crate::MatrixFull;
use nalgebra::DVector;

#[derive(Debug)]
pub struct DavidsonParams { 
    pub tol: f64,
    pub maxcyc: usize,
    pub maxspace: usize,
    pub lindep: f64,
    pub nroots: usize
    }

impl Default for DavidsonParams {
    fn default() -> Self {
        DavidsonParams{
           tol:      1e-5, 
           maxcyc:   50,
           maxspace: 12,
           lindep:   1e-14,
           nroots:   1
        }
    }
}

pub fn davidson_solve(mut a_x: Box<dyn FnMut(&Vec<f64>) -> Vec<f64> + '_>,
                      x0_1d: &mut Vec<f64>,
                      hdiag: &mut Vec<f64>,
                      params: &DavidsonParams,
                      print_level: usize,
                      ) -> (Vec<bool>, Vec<f64>, Vec<Vec<f64>>) {
    let tol = params.tol;
    let maxcyc = params.maxcyc;
    let mut maxspace = params.maxspace;
    let lindep = params.lindep;
    let nroots = params.nroots;
    println!("Davidson solver parameters:\n  tol {:?} maxcyc {:?} maxspace {:?}", tol, maxcyc, maxspace);
    let tol_res = tol.sqrt();
    
    let precond = |dx: Vec<f64>, e: f64| -> Vec<f64> {
        let mut hdiagd = hdiag.clone();
        hdiagd.iter_mut().for_each(|h| *h -= e);
        hdiagd.iter_mut().for_each(|h| if h.abs()<1e-8 {*h = 1e-8});
        let mut x2 = dx.clone();
        x2.iter_mut().zip(hdiagd.iter()).for_each(|(x, h)| *x /= *h);
        x2
    };

    let mut x0:Vec<Vec<f64>> = vec![];
    x0.push(x0_1d.to_vec());
    maxspace = maxspace + (nroots-1) * 3;
    //let mut heff = MatrixFull::new([maxspace+nroots, maxspace+nroots], 0.0f64);
    let mut fresh_start = true;
    let mut xt:Vec<Vec<f64>> = vec![];
    let mut axt:Vec<Vec<f64>> = vec![];
    let mut xtlen = 0;
    let mut xs:Vec<Vec<f64>> = vec![];
    let mut ax:Vec<Vec<f64>> = vec![];
    let mut space = 0;
    let mut x0len = x0_1d.len();
    let mut e = vec![0.0f64;nroots];
    let mut v:MatrixFull<f64> = MatrixFull::empty();
    let mut conv = vec![false; nroots];
    //emin = None
    let mut norm_min:f64 = 1.0;
    let mut max_dx_last = 1e9;
    let mut heff = MatrixFull::empty();

    for icyc in 0..maxcyc {

        if fresh_start {
            xs = vec![];
            ax = vec![];
            space = 0;
            xt = _qr(&mut x0, lindep).0;
            xtlen = xt.len();
            println!("ov {:?} xt.len {:?}", x0len, xtlen);
            //println!("{:?}", xt);
            if xtlen == 0 {
                panic!("No linear independent basis found");
            }
            max_dx_last = 1e9;
            //heff = MatrixFull::new([space, space], 0.0f64);
        } else {
            if xt.len() > 1 {
                xt = _qr(&mut xt.clone(), lindep).0;
                xt = xt[0..40].to_vec();
            }
        }
        println!(">>> start cyc {:?}   fresh {:?}  ", icyc, fresh_start);
        //println!("    xs {:?} ", xs);
        //println!("    xt {:?} ", xt);
        //let mut axt = ax(xt)
        axt = vec![];
        for xi in xt {
            let mut axi = a_x(&mut xi.clone());
            println!("    axi {:?} ", axi);
            axt.push(axi.to_vec());
            xs.push(xi.clone().to_vec());
            ax.push(axi.to_vec());
        }
        //for xi in &mut xt {
        //    let mut axi = a_x(xi);
        //    axt.push(axi.to_vec());
        //    xs.push(xi.clone());
        //    ax.push(axi.to_vec());
        //}
        //let mut xt_new = xt.clone();
        //xt_new.iter().for_each(|xi| {
        //    println!("    xs {:?} ", xs);
        //   //let mut axi = a_x(&xi.clone());
        //    let mut axi = a_x(xi);
        //    println!("    xs {:?} ", xs);
        //    axt.push(axi.to_vec());
        //    xs.push(xi.clone());
        //    println!("    xs {:?} ", xs);
        //    ax.push(axi.to_vec());
        //});
        //println!("    xs {:?} ", xs);
        //let xslen = xs.len();
        let mut rnow = xtlen;
        let mut head = space;
        space = space+rnow;
        let mut elast = e.clone();
        let mut vlast = v.clone();
        let mut convlast = conv.clone();
        //println!(" space {:?}", space);
        //heff = 
        heff = fill_heff(&mut heff, &mut xs, &mut ax, //&xt, &axt, 
                         xtlen, fresh_start);
        let mut heff_upper = heff.clone().to_matrixupper();
        let (mut eigvec, mut eigval, n_found) = heff_upper.to_matrixupperslicemut().lapack_dspevx().unwrap();
        e = eigval[0..nroots].to_vec();
        v = MatrixFull::from_vec([space, nroots], 
                                         eigvec.get_slices(0..space, 0..nroots).map(|i| *i).collect()).unwrap();
        if print_level > 3 {
            println!("    heff {:?}", heff);
            println!("    eigval {:?}", eigval);
            println!("    eigvec[0] {:?}", v);
            println!("    xs {:?} ", xs);
            println!("    ax {:?} ", ax);
        }
        let mut x0 = _gen_x0(&mut v, &mut xs);
        let mut ax0 = _gen_x0(&mut v, &mut ax);
        if print_level > 3 {
            //println!("    xs {:?} ", xs);
            println!("    x0 {:?} ", x0);
        }
        (elast, convlast) = _sort_elast(elast, convlast, &mut vlast, &mut v, fresh_start);
        xt = vec![];
        let mut dx_norm:Vec<f64> = vec![];
        let mut de:Vec<f64> = vec![];
        for k in 0..nroots {
            let de_k = e[k] - elast[k];
            de.push(de_k);
            let mut xt_k = ax0[k].clone();
            xt_k.iter_mut().zip(x0[k].iter()).for_each(|(xt,x0)| *xt -= e[k]* *x0);
            xt.push(xt_k.clone());
            let xt_k_na = DVector::from(xt_k.clone());
            let dx_k_norm = xt_k_na.norm();
            //println!("{:?} {:?} {:?} ", ax0[k], e[k], x0);
            //println!("{:?}", dx_k_norm);
            dx_norm.push(dx_k_norm);
            let conv_k = de_k.abs() < tol && dx_k_norm < tol.sqrt();
            conv[k] = conv_k;
            //println!("{:?} {:?} {:?} {:?} {:?}", conv, de_k, dx_k_norm, de_k.abs() < tol, dx_k_norm < tol.sqrt());
            if conv[k] && !convlast[k] {
                println!(">   root {:?} converged  |r|= {:?}  e= {:?}  de= {:?}",
                              k, dx_k_norm, e[k], de_k);
            }
        }
        //println!("    xt {:?} ", xt);
        let mut ax0:Vec<Vec<f64>> = vec![];
        let all_conv = conv.iter().fold(true, |acc, x| acc && *x);
        let max_dx_norm = DVector::from(dx_norm.clone()).max();
        if all_conv {
            println!(">>> converged at step {:?}  |r|= {:?}  e= {:?}  de= {:?}",
                      icyc,  max_dx_norm, e, de);
            break;
        } else {
            if max_dx_norm > 1.0 && max_dx_norm/max_dx_last > 3.0 && space > nroots+2 {
                println!(">>> davidson step {:?}  |r|= {:?}  e= {:?}  de= {:?}  lindep= {:?}",
                      icyc,  max_dx_norm, e, de, norm_min);
                println!("Large |r| detected, restore previous x0");
                x0 = _gen_x0(&mut vlast, &mut xs);
                fresh_start = true;
                continue;
            }
        }

        let mut xt_new:Vec<Vec<f64>> = vec![];
        for k in 0..nroots {
            if dx_norm[k].powf(2.0) > lindep {
                xt[k] = precond(xt[k].clone(), e[0],// x0[k]
                );
                let xt_k_na = DVector::from(xt[k].clone());
                let norm = xt_k_na.norm();
                xt[k].iter_mut().for_each(|x| *x /= norm);
                xt_new.push(xt[k].clone());
            } else {
                println!("Drop eigvec {:?} with norm {:?}", k, dx_norm[k]);
            }
        }
        xt = xt_new.clone();
        for i in 0..space {
            let xsi_na = DVector::from(xs[i].clone());
            for k in 0..xt.len() {
                let mut xtk = &xt[k];
                let mut xtk_na = DVector::from(xtk.clone());
                xtk_na -= xsi_na.clone() * xsi_na.dot(&xtk_na);
                xt[k] = xtk_na.data.into();
            }
        }
        //println!("xt {:?} ", xt);
        let mut xt_new:Vec<Vec<f64>> = vec![];
        for k in 0..xt.len() {
            let xt_k_na = DVector::from(xt[k].clone());
            let norm = xt_k_na.norm();
            if norm.powf(2.0) > lindep {
                xt[k].iter_mut().for_each(|x| *x /= norm);
                xt_new.push(xt[k].clone());
                norm_min = norm_min.min(norm);
            } else {
                println!("Drop eigvec {:?} with norm {:?}", k, dx_norm[k]);
            }
        }
        xt = xt_new.clone();
        //println!("    xt {:?} ", xt);
        //println!("    xs {:?} ", xs);
        println!(">>> davidson step {:?}  |r|= {:?}  e= {:?}  de= {:?}  lindep= {:?}",
                      icyc,  max_dx_norm, e, de, norm_min);
        if xt.len() == 0 {
            println!("Linear dependency in trial subspace. |r| for each state {:?}",
                      dx_norm);
            break;
        }

        let max_dx_last = max_dx_norm;
        fresh_start = space + nroots > maxspace;
    }
    if x0.len() < //std::cmp::min(x0[0].len(), nroots)
                  x0[0].len().min(nroots) {
        println!("Not enough eigvec");
    }
    (conv, e, x0)
}

pub fn _qr(x:&mut Vec<Vec<f64>>, lindep:f64) -> (Vec<Vec<f64>>, MatrixFull<f64>)
                                        {
    let nvec = x.len();
    let vecsize = x[0].len();
    //println!("_qr \n nvec {:?} vecsize {:?}", nvec, vecsize);
    //let mut qs = MatrixFull::new([nvec,vecsize], 0.0f64);
    let mut qs = vec![vec![0.0f64;vecsize]; nvec];
    let mut rmat = MatrixFull::new([nvec,nvec], 0.0f64);

    let mut nv = 0;
    for i in 0..nvec {
        let mut xi = x[i].clone();
        rmat.iter_mut_j(nv).for_each(|r| *r = 0.0);
        rmat.data[nv*nvec+nv] = 1.0;
        //println!("{:?}", rmat.data);
        for j in 0..nv {
            //let mut qsj = qs[j];
            let mut prod:f64 = qs[j].iter().zip(xi.iter()).map(|(q,x)| q*x).sum();
            xi.iter_mut().zip( qs[j].iter()).for_each(|(x,q)| *x -= *q *prod);
            let mut rmat_clone = rmat.clone(); 
            rmat.iter_mut_j(nv).zip(rmat_clone.iter_j(j)).for_each(|(r, rj)| *r -= *rj *prod); 
        };
        //let mut innerprod:f64 = xi.iter().zip(xi.iter()).map(|(x1,x2)| x1*x2).sum();
        let mut xi_na = DVector::from(xi.clone());
        let mut norm = xi_na.norm();
        //let mut norm = innerprod.sqrt();
        //println!("{:?}", xi);
        //println!("{:?}", norm);
        if norm.powf(2.0) > lindep {
            qs[nv].iter_mut().zip(xi.iter()).for_each(|(q,x)| *q = *x / norm);
            rmat.get_slices_mut(0..nv+1,nv..nv+1).for_each(|r| *r /= norm);
            nv += 1;
        }
    };
    let mut rmat_inv = rmat.lapack_inverse().unwrap();
    //println!(" qs {:?}", qs[0]);
    
    (qs[0..nv].to_vec(), rmat_inv)
}

pub fn fill_heff(heff_old:&mut MatrixFull<f64>, 
                 xs:&mut Vec<Vec<f64>>, ax:&mut Vec<Vec<f64>>,
                 //xt:&Vec<Vec<f64>>, axt:&Vec<Vec<f64>>,
                 xtlen:usize,
                 fresh_start:bool
                 ) -> MatrixFull<f64> 
                   {
    let nrow = xtlen;
    let row1 = ax.len();
    let row0 = row1 - nrow;
    let space = row1;
    println!("    space {:?}  xt.len {:?} xs.len {:?}", space, nrow, row1);
    //        println!("    xs {:?} ", xs);
    //println!("xt[0] {:?} ", xt[0]);
    //println!("xs[0] {:?} \nax[0] {:?}", xs[0], ax[0]);
    //if row1 > 1 {
    //println!("xs[1] {:?} \nax[1] {:?}", xs[1], ax[1]);}
    let mut heff = if fresh_start {
        MatrixFull::new([space, space], 0.0f64)         
    } else {
        let space_old = heff_old.size[0];
        let mut heff = MatrixFull::new([space, space], 0.0f64);
        for i in 0..space_old {
            for j in 0..space_old {
                heff.data[i*space+j] = heff_old.data[i*space_old+j];
            }
        };
        heff
    };
    for i in row0..row1 {
        let xt_i_na = DVector::from(xs[i].clone());
        for j in row0..row1 {
            //let mut xt_i_axt_j:f64 = xt[i-row0].iter().zip(axt[j-row0].iter()
            //                                           ).map(|(x,a)| x*a).sum();
            let axt_j_na = DVector::from(ax[j].clone());
            let mut xt_i_axt_j = xt_i_na.dot(&axt_j_na);
            heff.data[i*space+j] = xt_i_axt_j;
            heff.data[j*space+i] = xt_i_axt_j;
        }
    };
    //        println!("    xs {:?} ", xs);
    for j in row0..row1 {
        let xt_j_na = DVector::from(xs[j].clone());
        //let mut all_neg = false;
        for i in 0..row0 {
            let ax_i_na = DVector::from(ax[i].clone());
            //let mut ax_i_xt_j:f64 = ax[i].iter().zip(xt[j-row0].iter()
            //                                           ).map(|(x,a)| x*a).sum();
            let mut ax_i_xt_j = ax_i_na.dot(&xt_j_na);
            heff.data[i*space+j] = ax_i_xt_j;
            heff.data[j*space+i] = ax_i_xt_j;
        }
        //if heff.data[0+j] < 0.0 {
        //    xs[j].iter_mut().for_each(|x| *x *= (-1.0));
        //    ax[j].iter_mut().for_each(|x| *x *= (-1.0));
        //    for i in 0..row0 {
        //        heff.data[i*space+j] *= (-1.0);
        //        heff.data[j*space+i] *= (-1.0);
        //    }
        //}
    };
    //        println!("    xs {:?} ", xs);
    heff
}

fn _gen_x0(v:&mut MatrixFull<f64>, x:&mut Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let space = v.size[0];
    let nroots = v.size[1];
    let xlen = x[0].len();
    let mut xmat = MatrixFull::from_vec([xlen, space], x.concat() ).unwrap();
    let mut x0mat = MatrixFull::new([xlen, nroots], 0.0f64);

    x0mat.lapack_dgemm(&mut xmat, v, 'N', 'N', 1.0, 0.0);
    let mut x0vecs:Vec<Vec<f64>> = vec![];
    x0mat.iter_columns_full().for_each(|x0| x0vecs.push(x0.to_vec()));

    x0vecs
}

fn _sort_elast(elast:Vec<f64>, convlast:Vec<bool>, 
               vlast:&mut MatrixFull<f64>, v:&mut MatrixFull<f64>, fresh_start:bool) -> (Vec<f64>, Vec<bool>) {
    if fresh_start {
        (elast, convlast)
    } else {
        //println!("v.size {:?}  vlast.size {:?}", v.size, vlast.size);
        let head = vlast.size[0];
        let nroots = vlast.size[1];
        let mut ovlp = MatrixFull::new([nroots, nroots], 0.0f64);
        let mut v_head = MatrixFull::from_vec([head, nroots], 
                                         v.get_slices(0..head, 0..nroots).map(|i| *i).collect()).unwrap();
        ovlp.lapack_dgemm(&mut v_head, vlast, 'T', 'N', 1.0, 0.0);
        let mut idx:Vec<usize> = vec![];
        ovlp.iter_columns_full().for_each(|x| { 
                                          let x_na = DVector::from(x.to_vec());
                                          idx.push(x_na.imax() ); } );

        //println!("{:?}", idx);
        let mut new_elast:Vec<f64> = vec![];
        let mut new_convlast:Vec<bool> = vec![];
        for i in idx {
            new_elast.push(elast[i]);
            new_convlast.push(convlast[i]);
        }
        (elast, convlast)
    }
}

