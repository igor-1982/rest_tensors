#!/usr/bin/env bash

#ifort restmatr.f90  -shared -fpic -o librestmatr.so -O2 -L${REST_EXT_DIR} -lopenblas
# Manual compile example. Use $REST_EXT_DIR (REST external library dir) so the
# script is machine-independent; normally `cargo build` in rest_tensors compiles
# restmatr automatically with the same setting.
gfortran restmatr.f90  -shared -fpic -o librestmatr.so -O2 -L"${REST_EXT_DIR:?set REST_EXT_DIR to the REST external library dir}" -m64 -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl


#ifort -O2 restmatr.f90 -L${REST_EXT_DIR} -lopenblas


