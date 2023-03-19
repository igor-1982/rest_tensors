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
    pub fn special_dgemm_f_01_(
        ten3_a: *const c_double, x_a: *const c_int, y_a: *const c_int, z_a: *const c_int,
        start_x_a: *const c_int, len_x_a: *const c_int, i_y: *const c_int,
        start_z_a: *const c_int, len_z_a: *const c_int,
        matr_b: *const c_double, rows_b: *const c_int, columns_b: *const c_int, 
        start_row_b: *const c_int, len_row_b: *const c_int,
        start_column_b: *const c_int, len_column_b: *const c_int,
        alpha: *const c_double, beta: *const c_double
    );

    pub fn copy_mm_(
        x_len: *const c_int, y_len: *const c_int,
        f_matr: *const c_double, f_x_len: *const c_int, f_y_len: *const c_int, f_x_start: *const c_int, f_y_start: *const c_int,
        t_matr: *mut c_double, t_x_len: *const c_int, t_y_len: *const c_int, t_x_start: *const c_int, t_y_start: *const c_int,
    );

    pub fn copy_mr_(
        x_len: *const c_int, y_len: *const c_int,
        f_matr: *const c_double, f_x_len: *const c_int, f_y_len: *const c_int, f_x_start: *const c_int, f_y_start: *const c_int,
        t_ri: *mut c_double, t_x_len: *const c_int, t_y_len: *const c_int, t_z_len: *const c_int, t_x_start: *const c_int, t_y_start: *const c_int,
        t_x3: *const c_int, t_mod: *const c_int
    );

    pub fn copy_rm_(
        x_len: *const c_int, y_len: *const c_int,
        f_ri: *const c_double, f_x_len: *const c_int, f_y_len: *const c_int, f_z_len: *const c_int, f_x_start: *const c_int, f_y_start: *const c_int,
        f_x3: *const c_int, f_mod: *const c_int,
        t_matr: *mut c_double, t_x_len: *const c_int, t_y_len: *const c_int, t_x_start: *const c_int, t_y_start: *const c_int
    );

    pub fn copy_rr_(
        x_len: *const c_int, y_len: *const c_int, z_len: *const c_int,
        f_ri: *const c_double, f_x_len: *const c_int, f_y_len: *const c_int, f_z_len: *const c_int,
        f_x_start: *const c_int, f_y_start: *const c_int, f_z_start: *const c_int,
        t_ri: *mut c_double,   t_x_len: *const c_int, t_y_len: *const c_int, t_z_len: *const c_int, 
        t_x_start: *const c_int, t_y_start: *const c_int, t_z_start: *const c_int
    );
}
