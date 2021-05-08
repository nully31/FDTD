// 3D FDTD Program for auto SIMD-vectorization by OpenMP (you may need `Intel C Compiler (icc)` in order to auto vectorize).
// The model for this simulation is the vacuum space covered by perfect electric conductor (PEC).
// Please compile with `openmp` options.
// Enter a size of each dimension that can be divided by 8 for command line arguments when execute.

#define ex(i, j, k) ex[ny*nz*(i) + nz*(j) + (k)]
#define ey(i, j, k) ey[ny*nz*(i) + nz*(j) + (k)]
#define ez(i, j, k) ez[ny*nz*(i) + nz*(j) + (k)]
#define hx(i, j, k) hx[ny*nz*(i) + nz*(j) + (k)]
#define hy(i, j, k) hy[ny*nz*(i) + nz*(j) + (k)]
#define hz(i, j, k) hz[ny*nz*(i) + nz*(j) + (k)]
#define media_id(i, j, k) media_id[ny*nz*(i) + nz*(j) + (k)]

#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

/*----------constants----------*/
const double pi = 3.141592653589793;
const double c = 2.998e8;
const double epsilon0 = 8.854e-12;
const double mu0 = 1.25663e-6;
const double sig0 = 1.0e-8;
const double z0 = 376.734309182110; // impedance

/*----------source freqency----------*/
const double freq = 8e9;

/*----------time step----------*/
int ntime;
double time, dt;

/*----------field----------*/
int nx, ny, nz;
double dx, dy, dz;
double * restrict ex, * restrict ey, * restrict ez;
double * restrict hx, * restrict hy, * restrict hz;

/*----------media----------*/
// int mmedia = 10;
const int nmedia = 2;
double * restrict eps;
double * restrict mu;
double * restrict sig;

int * restrict media_id;

/*----------media update constant----------*/
double cex0, cey0, cez0;
double cexry0, cexrz0,
	ceyrz0, ceyrx0,
	cezrx0, cezry0;
double chxry0, chxrz0,
	chyrz0, chyrx0,
	chzrx0, chzry0;
double * restrict cex, * restrict cey, * restrict cez;
double * restrict cexry, * restrict cexrz,
	* restrict ceyrz, *restrict ceyrx,
	* restrict cezrx, * restrict cezry;
double * restrict chxry, * restrict chxrz,
	* restrict chyrz, * restrict chyrx,
	* restrict chzrx, * restrict chzry;

void init();
void modeling();
void electric_field();
void electric_boundary_condition();
void ecur_source();
double ecur(double t);
void magnetic_field();
void output();
void free_all();

int main(int argc, char *argv[]) {
	if (argc != 5) {
		fprintf(stderr, "Enter the size of each dimension and timesteps\n");
		return -1;
	}
	nx = atoi(argv[1]) + 2;
	ny = atoi(argv[2]) + 2;
	nz = atoi(argv[3]) + 2;
	ntime = atoi(argv[4]);
	/*----------main routine----------*/
	init();
	modeling();

	/*----------FDTD routine----------*/
	time = 0.0; 

	double dtime = 0.0, dtime_e, dtime_m;
	for (int n = 0; n < ntime; n++) {
		dtime_e = - omp_get_wtime();
		electric_field();
		dtime_e += omp_get_wtime();
		electric_boundary_condition();
		ecur_source();
		time = time + dt / 2.0;

		dtime_m = - omp_get_wtime();
		magnetic_field();
		dtime_m += omp_get_wtime();
		time = time + dt / 2.0;

		dtime += (dtime_e + dtime_m);
		/*----------output----------*/
		//if (n % 10 == 0) {
		//	output();
		//}
	}
	printf("%lf,\t", dtime);
	output();
	free_all();
	return 0;
}


void init() {
	/*----------initialize vectors/matrices----------*/
	int i, j, k;
	ex = malloc((nx + 1) * (ny + 1) * (nz + 1) * sizeof(double));
	ey = malloc((nx + 1) * (ny + 1) * (nz + 1) * sizeof(double));
	ez = malloc((nx + 1) * (ny + 1) * (nz + 1) * sizeof(double));
	hx = malloc((nx + 1) * (ny + 1) * (nz + 1) * sizeof(double));
	hy = malloc((nx + 1) * (ny + 1) * (nz + 1) * sizeof(double));
	hz = malloc((nx + 1) * (ny + 1) * (nz + 1) * sizeof(double));
	media_id = malloc((nx + 1) * (ny + 1) * (nz + 1) * sizeof(int));

	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			for (k = 0; k < nz; k++) {
				ex(i, j, k) = 0.0;
				ey(i, j, k) = 0.0;
				ez(i, j, k) = 0.0;
				hx(i, j, k) = 0.0;
				hy(i, j, k) = 0.0;
				hz(i, j, k) = 0.0;
			}
		}
	}

	eps = malloc(nmedia * sizeof(double));
	mu = malloc(nmedia * sizeof(double));
	sig = malloc(nmedia * sizeof(double));

	cex = malloc(nmedia * sizeof(double));
	cey = malloc(nmedia * sizeof(double));
	cez = malloc(nmedia * sizeof(double));
	cexry = malloc(nmedia * sizeof(double));
	cexrz = malloc(nmedia * sizeof(double));
	ceyrz = malloc(nmedia * sizeof(double));
	ceyrx = malloc(nmedia * sizeof(double));
	cezrx = malloc(nmedia * sizeof(double));
	cezry = malloc(nmedia * sizeof(double));
	chxry = malloc(nmedia * sizeof(double));
	chxrz = malloc(nmedia * sizeof(double));
	chyrz = malloc(nmedia * sizeof(double));
	chyrx = malloc(nmedia * sizeof(double));
	chzrx = malloc(nmedia * sizeof(double));
	chzry = malloc(nmedia * sizeof(double));

	/*----------cell size----------*/
	/* Courant stability condition
	   vdt < 1/sqrt((1/dx)^2+(1/dy)^2+(1/dz)^2)
	   vdt < dx/sqrt(3) when dx=dy=dz
	*/
	dx = 29.1e-3 / 10.0;
	dy = dx;
	dz = dx;
	dt = 1.0 / freq / 100;

	/*----------media settings----------*/

	// 0:vacuum
	eps[0] = epsilon0;
	mu[0] = mu0;
	sig[0] = sig0;

	// 1:PEC, PMC (in other routine)

	/*----------media constants----------*/
	// 0: vacuum
	cex[0] = 1.0;
	cey[0] = 1.0;
	cez[0] = 1.0;
	
	cexry[0] = (dt / epsilon0) / dy;
	cexrz[0] = (dt / epsilon0) / dz;
	ceyrz[0] = (dt / epsilon0) / dz;
	ceyrx[0] = (dt / epsilon0) / dx;
	cezrx[0] = (dt / epsilon0) / dx;
	cezry[0] = (dt / epsilon0) / dy;

	chxry[0] = (dt / mu0) / dy;
	chxrz[0] = (dt / mu0) / dz;
	chyrz[0] = (dt / mu0) / dz;
	chyrx[0] = (dt / mu0) / dx;
	chzrx[0] = (dt / mu0) / dx;
	chzry[0] = (dt / mu0) / dy;
	
	// 1: PEC
	cex[1] = 0.0;
	cey[1] = 0.0;
	cez[1] = 0.0;
	
	cexry[1] = 0.0;
	cexrz[1] = 0.0;
	ceyrz[1] = 0.0;
	ceyrx[1] = 0.0;
	cezrx[1] = 0.0;
	cezry[1] = 0.0;

	chxry[1] = 0.0;
	chxrz[1] = 0.0;
	chyrz[1] = 0.0;
	chyrx[1] = 0.0;
	chzrx[1] = 0.0;
	chzry[1] = 0.0;

	// 2 or more: general media
	for (int id = 2; id < nmedia; id++) {
		cex[id] = (1.0 - (sig[id] * dt) / (2.0 * eps[id]))
			/ (1.0 + ((sig[id] * dt) / (2.0 * eps[id])));
		cey[id] = (1.0 - (sig[id] * dt) / (2.0 * eps[id]))
			/ (1.0 + ((sig[id] * dt) / (2.0 * eps[id])));
		cez[id] = (1.0 - (sig[id] * dt) / (2.0 * eps[id]))
			/ (1.0 + ((sig[id] * dt) / (2.0 * eps[id])));

		cexry[id] = (dt / eps[id]) / (1.0 + ((sig[id] * dt) / (2.0 * eps[id]))) / dy;
		cexrz[id] = (dt / eps[id]) / (1.0 + ((sig[id] * dt) / (2.0 * eps[id]))) / dz;
		ceyrz[id] = (dt / eps[id]) / (1.0 + ((sig[id] * dt) / (2.0 * eps[id]))) / dz;
		ceyrx[id] = (dt / eps[id]) / (1.0 + ((sig[id] * dt) / (2.0 * eps[id]))) / dx;
		cezrx[id] = (dt / eps[id]) / (1.0 + ((sig[id] * dt) / (2.0 * eps[id]))) / dx;
		cezry[id] = (dt / eps[id]) / (1.0 + ((sig[id] * dt) / (2.0 * eps[id]))) / dy;

		chxry[id] = (dt / mu[id]) / dy;
		chxrz[id] = (dt / mu[id]) / dz;
		chyrz[id] = (dt / mu[id]) / dz;
		chyrx[id] = (dt / mu[id]) / dx;
		chzrx[id] = (dt / mu[id]) / dx;
		chzry[id] = (dt / mu[id]) / dy;
	}
}


void modeling() {
	/*----------modeling----------*/
	int i, j, k;

	// init with vacuum(=0)
	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			for (k = 0; k < nz; k++) {
				media_id(i, j, k) = 0;
			}
		}
	}

	// init with PEC(=1)
	// up and down
	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			media_id(i, j, 0) = 1;
			media_id(i, j, nz-1) = 1;
		}
	}
	// left and right
	for (j = 0; j < ny; j++) {
		for (k = 0; k < nz; k++) {
			media_id(0, j, k) = 1;
			media_id(nx-1, j, k) = 1;
		}
	}
	// front
	for (i = 0; i < nx; i++) {
		for (k = 0; k < nz; k++) {
			media_id(i, 0, k) = 1;
		}
	}
}


void electric_field() {
        /*----------update electric field----------*/
	int i, j, k;
	int id;
	for (i = 1; i < nx-1; i++) {
		for (j = 1; j < ny-1; j++) {
			#pragma omp simd
			for (k = 1; k < nz-1; k++) {
				id = media_id(i, j, k);

				ex(i, j, k) = cex[id] * ex(i, j, k)
						+ cexry[id] * (hz(i, j, k) - hz(i, j-1, k))
					- cexrz[id] * (hy(i, j, k) - hy(i, j, k-1));
				ey(i, j, k) = cey[id] * ey(i, j, k)
					+ ceyrz[id] * (hx(i, j, k) - hx(i, j, k-1))
					- ceyrx[id] * (hz(i, j, k) - hz(i-1, j, k));
				ez(i, j, k) = cez[id] * ez(i, j, k)
					+ cezrx[id] * (hy(i, j, k) - hy(i-1, j, k))
					- cezry[id] * (hx(i, j, k) - hx(i, j-1, k));
			}
		}
	}
}


void electric_boundary_condition() {
	/*----------electric boundary condition----------*/
	int i, j, k;
	// set e values touching left wall to zero
	for (j = 0; j < ny; j++) {
		for (k = 0; k < nz; k++) {
			ey(1, j, k) = 0.0;
			ez(1, j, k) = 0.0;
		}
	}

	// set e values touching bottom wall to 0
	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			ex(i, j, 1) = 0.0;
			ey(i, j, 1) = 0.0;
		}
	}
}


void ecur_source() {
	/*-----------electronic source----------*/
	int i, j, k;
	// plane wave
	i = 2;
	for (j = 1; j < ny-1; j++) {
		for (k = 1; k < nz-1; k++) {
			ez(i, j, k) = ecur(time - dt / 2.0);
			hx(i, j, k) = ez(i, j, k) / z0;
		}
	}
}


double ecur(double t) {
	double ecur;
	double tp, t1;

	tp = 1.0/freq;
	t1 = 4 * tp;
	if (t < t1) {
		ecur = 0.5 * (1.0 - cos(pi * t / t1)) * sin(2.0 * pi * freq * t);
	} else {
		ecur = sin(2.0 * pi * freq * t);
	}

	return ecur;
}


void magnetic_field() {
	/*----------update magnetic field----------*/
	int i, j, k;
	int id;
	for (i = 1; i < nx-1; i++) {
		for (j = 1; j < ny-1; j++) {
			#pragma omp simd
			for (k = 1; k < nz-1; k++) {
				id = media_id(i, j, k);

				hx(i, j, k) = hx(i, j, k)
					- chxry[id] * (ez(i, j+1, k) - ez(i, j, k))
					+ chxrz[id] * (ey(i, j, k+1) - ey(i, j, k));
				hy(i, j, k) = hy(i, j, k)
					- chyrz[id] * (ex(i, j, k+1) - ex(i, j, k))
					+ chyrx[id] * (ez(i+1, j, k) - ez(i, j, k));
				hz(i, j, k) = hz(i, j, k)
					- chzrx[id] * (ey(i+1, j, k) - ey(i, j, k))
					+ chzry[id] * (ex(i, j+1, k) - ex(i, j, k));
			}
		}
	}
}

void output() {
	int i = 4, j = 1;
	// 	for (int i = 0; i < nx; i++) {
	// 		for (int j = 0; j < ny; j++) {
				printf("%d %d %.12f\n", i, j, ez(i, j, nz/2));
	//		}
	//	}
	//	printf("\n\n");
}

void free_all() {
	free(ex);
	free(ey);
	free(ez);
	free(hx);
	free(hy);
	free(hz);
	free(media_id);
	free(eps);
	free(mu);
	free(sig);
	free(cex);
	free(cey);
	free(cez);
	free(cexry);
	free(cexrz);
	free(ceyrz);
	free(ceyrx);
	free(cezrx);
	free(cezry);
	free(chxry);
	free(chxrz);
	free(chyrz);
	free(chyrx);
	free(chzrx);
	free(chzry);
}
