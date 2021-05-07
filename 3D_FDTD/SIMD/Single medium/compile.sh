#!/bin/bash
icc -o no_vec fdtd_base_no_vec.c -O3 -qopenmp -no-vec
icc -o auto_vec fdtd_base_auto.c -O3 -xskylake-avx512 -qopt-zmm-usage=high -qopenmp
icc -o gather fdtd_gather.c -O3 -xskylake-avx512 -qopt-zmm-usage=high -qopenmp
icc -o alignr fdtd_alignr.c -O3 -xskylake-avx512 -qopt-zmm-usage=high -qopenmp
icc -o switch fdtd_switch.c -O3 -xskylake-avx512 -qopt-zmm-usage=high -qopenmp
icc -o alignr_switch fdtd_alignr_switch.c -O3 -xskylake-avx512 -qopt-zmm-usage=high -qopenmp
