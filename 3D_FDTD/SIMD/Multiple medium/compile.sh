#!/bin/bash
icc -o auto_vec_mul fdtd_base_auto_mul.c -O3 -xskylake-avx512 -qopt-zmm-usage=high -qopenmp
icc -o alignr_mul fdtd_alignr_mul.c -O3 -xskylake-avx512 -qopt-zmm-usage=high -qopenmp
icc -o alignr_switch_mul fdtd_alignr_switch_mul.c -O3 -xskylake-avx512 -qopt-zmm-usage=high -qopenmp
