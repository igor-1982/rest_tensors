#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, improper_ctypes)]

mod sys {
    include!(concat!(env!("OUT_DIR"), "/pexsi_bindings.rs"));
}

pub use sys::*;
