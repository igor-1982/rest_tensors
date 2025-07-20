#!/usr/bin/env bash

#ifort restmatr.f90  -shared -fpic -o librestmatr.so -O2 -L/share/apps/rust/OpenBLAS-0.3.17 -lopenblas
gfortran restmatr.f90  -shared -fpic -o librestmatr.so -O2 -L/home/igor/Documents/Package-Pool/rest_workspace/lib -m64 -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl


#ifort -O2 restmatr.f90 -L/share/apps/rust/OpenBLAS-0.3.17 -lopenblas


