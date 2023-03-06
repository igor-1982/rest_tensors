use std::ffi::{c_double, c_int, c_char};

//#[link(name="restmatr")]
extern "C" {
    pub fn ri_ao2mo_f_(eigenvector: *const c_double, 
        ri3fn: *const c_double, 
        ri3mo: *mut c_double, 
        num_states: *const c_int,
        num_basis: *const c_int,
        num_auxbas: *const c_int,
    );
    
    pub fn general_dgemm_f_(
        matr_a: *const c_double, rows_a: *const c_int, columns_a: *const c_int, 
        start_row_a: *const c_int, len_row_a: *const c_int, 
        start_column_a: *const c_int, len_column_a: *const c_int, opa: *const c_char,
        matr_b: *const c_double, rows_b: *const c_int, columns_b: *const c_int, 
        start_row_b: *const c_int, len_row_b: *const c_int,
        start_column_b: *const c_int, len_column_b: *const c_int, opb: *const c_char,
        matr_c: *mut c_double, rows_c: *const c_int, columns_c: *const c_int, 
        start_row_c: *const c_int, len_row_c: *const c_int,
        start_column_c: *const c_int, len_column_c: *const c_int,
        alpha: *const c_double, beta: *const c_double
    );
}
