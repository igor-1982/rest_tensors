#!/usr/bin/env bash

ifort restmatr.f90  -shared -fpic -o librestmatr.so -O2 -L/share/apps/rust/OpenBLAS-0.3.17 -lopenblas

#ifort -O2 restmatr.f90 -L/share/apps/rust/OpenBLAS-0.3.17 -lopenblas


