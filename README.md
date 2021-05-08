A Finite Differential Time Domain (FDTD) method program for electro-magnetic field simulation, written in C.


`2D_FDTD_TE` contains files for 2D FDTD simulation, and `3D_FDTD/Sequential` are for 3D FDTD.
Each source file has a different perfect electric conductor (PEC) model.


Files in `3D_FDTD/SIMD` are for AVX-512 SIMD optimizations of 3D FDTD simulation.
You may want to use `Intel C Compiler (icc)` for best performance.

# COMPILE
Compile the source files in these folders just like other normal C programs, no additional options required except `3D_FDTD/SIMD`.

```
icc 2d_fdtd.c -o 2d_fdtd -O3
```

In `3D_FDTD/SIMD`, you may want to use the shell script in the folder to compile and execute.

# USAGE
## 2D_FDTD_TE and 3D_FDTD/Sequential
Execute the executables and redirect the output to `result.dat` so you can plot the simulation results in gnuplot.
1. `$ ./2d_fdtd > result.dat`
2. `$ gnuplot`
3. `(in gnuplot) >load "plot_2d.gp"`

## 3D_FDTD/SIMD
Use the shell scripts to and execute. Enter a dimension size and max time step to simulate.
Please note that the given dimension size must be dividable by 8.
1. `$ sh execute.sh ($size) ($time step)`

# References
- 橋本 修, "FDTD時間領域差分法入門"
- 宇野 亨, "FDTD法による電磁界およびアンテナ解析"
- 平野拓一, "[FDTD法]"(http://www.takuichi.net/em_analysis/fdtd/fdtd.html)