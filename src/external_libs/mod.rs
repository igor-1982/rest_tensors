mod ffi_restmatr;
use std::{ops::Range};

use crate::external_libs::ffi_restmatr::*;

pub fn ri_ao2mo_f(
    eigenvector:&[f64], 
    ri3fn:&[f64], 
    ri3mo:&mut [f64], 
    num_states: usize, 
    num_basis: usize, 
    num_auxbas: usize) 
{
    unsafe{ri_ao2mo_f_(
        eigenvector.as_ptr(), 
        ri3fn.as_ptr(), 
        ri3mo.as_mut_ptr(), 
        &(num_states as i32),
        &(num_basis as i32),
        &(num_auxbas as i32))
    }
}


/// # an efficient and general dgemm wrapped to a fortran source
///  matr_c[(range_row_c, range_column_c)] =
///      alpha * opa(matr_a[(range_row_a, range_column_a)])*opb(matr_b[(range_row_b, range_column_b)]) +
///      beta * matr_c[(range_row_c, range_column_c)]
/// Example
/// ```
///    use rest_tensors::MatrixFull;
///    use crate::rest_tensors::BasicMatrix;
///    use rest_tensors::external_libs::general_dgemm_f;
///    let matr_a = MatrixFull::from_vec([3,3], (1..10).map(|x| x as f64).collect::<Vec<f64>>()).unwrap();
///    //             | 1.0 | 4.0 | 7.0 |
///    // matr_a =    | 2.0 | 5.0 | 8.0 |
///    //             | 3.0 | 6.0 | 9.0 |
///    let matr_b = MatrixFull::from_vec([3,3], (6..15).map(|x| x as f64).collect::<Vec<f64>>()).unwrap();
///    //             | 6.0 | 9.0 |12.0 |
///    // matr_b =    | 7.0 |10.0 |13.0 |
///    //             | 8.0 |11.0 |14.0 |
///    let mut matr_c = MatrixFull::new([3,3], 2.0);
///    //             | 2.0 | 2.0 | 2.0 |
///    // matr_c =    | 2.0 | 2.0 | 2.0 |
///    //             | 2.0 | 2.0 | 2.0 |
/// 
///    let matr_c_size = matr_c.size.clone();
/// 
///    general_dgemm_f(
///         matr_a.data_ref().unwrap(), matr_a.size(), 1..3, 1..3, 'N',
///         matr_b.data_ref().unwrap(), matr_b.size(), 0..2, 0..2, 'N',
///         matr_c.data_ref_mut().unwrap(), &matr_c_size, 1..3, 0..2, 
///         1.0, 1.0
///    );
///    //             |  2.0 |  2.0 | 2.0 |
///    // matr_c =    | 88.0 |127.0 | 2.0 |
///    //             |101.0 |144.0 | 2.0 |
///    assert_eq!(matr_c.get_submatrix(1..3, 0..2).data(),vec![88.0,101.0,127.0,146.0])
/// ```
pub fn general_dgemm_f(
    matr_a: &[f64], size_a: &[usize], range_row_a: Range<usize>, range_column_a: Range<usize>, opa: char,
    matr_b: &[f64], size_b: &[usize], range_row_b: Range<usize>, range_column_b: Range<usize>, opb: char,
    matr_c: &mut [f64], size_c: &[usize], range_row_c: Range<usize>, range_column_c: Range<usize>,
    alpha: f64, beta: f64
) {
    unsafe{general_dgemm_f_(
        matr_a.as_ptr(),&(size_a[0] as i32),&(size_a[1] as i32),
        &(range_row_a.start as i32), &(range_row_a.len() as i32),
        &(range_column_a.start as i32), &(range_column_a.len() as i32),
        &(opa as std::ffi::c_char), 

        matr_b.as_ptr(),&(size_b[0] as i32),&(size_b[1] as i32),
        &(range_row_b.start as i32), &(range_row_b.len() as i32),
        &(range_column_b.start as i32), &(range_column_b.len() as i32),
        &(opb as std::ffi::c_char), 

        matr_c.as_mut_ptr(),&(size_c[0] as i32),&(size_c[1] as i32),
        &(range_row_c.start as i32), &(range_row_c.len() as i32),
        &(range_column_c.start as i32), &(range_column_c.len() as i32),

        &alpha, &beta,
    )
    }

}
