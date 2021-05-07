// 3D FDTD Program for SIMD-vectorization by intrinsics.
// The model for this simulation is the vacuum space covered by perfect magnetic conductor (PMC).
// Arrays are zero-padded to exclude boundaries from the start of 64 byte alignments.
// This program uses `broadcast` instruction instead of `gather` instruction to load media constants,
// if the media is the same between all the SIMD lanes.
// And also uses `alinq` instruction to reduce a few loads.
// Please compile with `openmp` and `avx512` options.
// Enter a size of each dimension that can be divided by 8 for command line arguments when execute.

#define pad 8 //padding for vector registers

#define ex(i, j, k) ex[ny*nz*(i) + nz*(j) + (k)]
#define ey(i, j, k) ey[ny*nz*(i) + nz*(j) + (k)]
#define ez(i, j, k) ez[ny*nz*(i) + nz*(j) + (k)]
#define hx(i, j, k) hx[ny*nz*(i) + nz*(j) + (k)]
#define hy(i, j, k) hy[ny*nz*(i) + nz*(j) + (k)]
#define hz(i, j, k) hz[ny*nz*(i) + nz*(j) + (k)]
#define media_id(i, j, k) media_id[ny*nz*(i) + nz*(j) + (k)]
#define media_row(i, j, k) media_row[ny*nz*(i) + nz*(j) + (k)]

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
int * restrict media_row;
/*----------media update constant----------*/
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
	nz = atoi(argv[3]);
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
	ex = _mm_malloc((nx) * (ny) * (nz + pad * 2) * sizeof(double), 64);
	ey = _mm_malloc((nx) * (ny) * (nz + pad * 2) * sizeof(double), 64);
	ez = _mm_malloc((nx) * (ny) * (nz + pad * 2) * sizeof(double), 64);
	hx = _mm_malloc((nx) * (ny) * (nz + pad * 2) * sizeof(double), 64);
	hy = _mm_malloc((nx) * (ny) * (nz + pad * 2) * sizeof(double), 64);
	hz = _mm_malloc((nx) * (ny) * (nz + pad * 2) * sizeof(double), 64);
	media_id = _mm_malloc((nx) * (ny) * (nz + pad * 2) * sizeof(int), 32);
	media_row = _mm_malloc((nx) * (ny) * (nz + pad * 2) * sizeof(int), 32);

	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			for (k = 0; k < nz+pad*2; k++) {
				ex(i, j, k) = 0.0;
				ey(i, j, k) = 0.0;
				ez(i, j, k) = 0.0;
				hx(i, j, k) = 0.0;
				hy(i, j, k) = 0.0;
				hz(i, j, k) = 0.0;
			}
		}
	}

	eps = _mm_malloc(nmedia * sizeof(double), 64);
	mu = _mm_malloc(nmedia * sizeof(double), 64);
	sig = _mm_malloc(nmedia * sizeof(double), 64);

	cex = _mm_malloc(nmedia * sizeof(double), 64);
	cey = _mm_malloc(nmedia * sizeof(double), 64);
	cez = _mm_malloc(nmedia * sizeof(double), 64);
	cexry = _mm_malloc(nmedia * sizeof(double), 64);
	cexrz = _mm_malloc(nmedia * sizeof(double), 64);
	ceyrz = _mm_malloc(nmedia * sizeof(double), 64);
	ceyrx = _mm_malloc(nmedia * sizeof(double), 64);
	cezrx = _mm_malloc(nmedia * sizeof(double), 64);
	cezry = _mm_malloc(nmedia * sizeof(double), 64);
	chxry = _mm_malloc(nmedia * sizeof(double), 64);
	chxrz = _mm_malloc(nmedia * sizeof(double), 64);
	chyrz = _mm_malloc(nmedia * sizeof(double), 64);
	chyrx = _mm_malloc(nmedia * sizeof(double), 64);
	chzrx = _mm_malloc(nmedia * sizeof(double), 64);
	chzry = _mm_malloc(nmedia * sizeof(double), 64);

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
	int rest = 0;

	// init with vacuum(=0)
	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			for (k = 0; k < nz+pad*2; k++) {
				media_id(i, j, k) = 0;

				if (k == pad) rest = nz;
				media_row(i, j, k) = rest;
				if (rest > 0) rest--;
				else rest = 0;
			}
		}
	}

	// init with PEC(=1)
	// up and down
	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			media_id(i, j, pad-1) = 1;
			media_id(i, j, nz+pad-1) = 1;
		}
	}
	// left and right
	for (j = 0; j < ny; j++) {
		for (k = pad-1; k < nz+pad; k++) {
			media_id(0, j, k) = 1;
			media_id(nx-1, j, k) = 1;
		}
	}
	// front
	for (i = 0; i < nx; i++) {
		for (k = pad-1; k < nz+pad; k++) {
			media_id(i, 0, k) = 1;
		}
	}
}


void electric_field() {
        /*----------update electric field----------*/
	int i, j, k;
	int id;

	__m128d xcex, xcey, xcez;
	__m128d xcexry, xcexrz,
		xceyrz, xceyrx,
		xcezrx, xcezry;
	__m512d zcex, zcey, zcez;
	__m512d zcexry, zcexrz,
		zceyrz, zceyrx,
		zcezrx, zcezry;

	for (i = 1; i < nx-1; i++) {
		for (j = 1; j < ny-1; j++) {
			for (k = pad; k < nz+pad; k += 8) {
				//id = media_id(i, j, k);
				if (media_row(i, j, k) < 8) {
					__m256i yid = _mm256_load_si256((__m256i *)&media_id(i, j, k));
					zcex = _mm512_i32gather_pd(yid, &cex[0], 8);
					zcey = _mm512_i32gather_pd(yid, &cey[0], 8);
					zcez = _mm512_i32gather_pd(yid, &cez[0], 8);
					zcexry = _mm512_i32gather_pd(yid, &cexry[0], 8);
					zcexrz = _mm512_i32gather_pd(yid, &cexrz[0], 8);
					zceyrz = _mm512_i32gather_pd(yid, &ceyrz[0], 8);
					zceyrx = _mm512_i32gather_pd(yid, &ceyrx[0], 8);
					zcezrx = _mm512_i32gather_pd(yid, &cezrx[0], 8);
					zcezry = _mm512_i32gather_pd(yid, &cezry[0], 8);
				} else {
					id = media_id(i, j, k);
					xcex = _mm_load_sd(&cex[id]);
					zcex = _mm512_broadcastsd_pd(xcex);
					xcexry = _mm_load_sd(&cexry[id]);
					zcexry = _mm512_broadcastsd_pd(xcexry);
					xcexrz = _mm_load_sd(&cexrz[id]);
					zcexrz = _mm512_broadcastsd_pd(xcexrz);

					xcey = _mm_load_sd(&cey[id]);
					zcey = _mm512_broadcastsd_pd(xcey);
					xceyrz = _mm_load_sd(&ceyrz[id]);
					zceyrz = _mm512_broadcastsd_pd(xceyrz);
					xceyrx = _mm_load_sd(&ceyrx[id]);
					zceyrx = _mm512_broadcastsd_pd(xceyrx);

					xcez = _mm_load_sd(&cez[id]);
					zcez = _mm512_broadcastsd_pd(xcez);
					xcezrx = _mm_load_sd(&cezrx[id]);
					zcezrx = _mm512_broadcastsd_pd(xcezrx);
					xcezry = _mm_load_sd(&cezry[id]);
					zcezry = _mm512_broadcastsd_pd(xcezry);
				}

				__m512d hx0 = _mm512_load_pd(&hx(i, j, k));
				__m512d hy0 = _mm512_load_pd(&hy(i, j, k));
				__m512d hz0 = _mm512_load_pd(&hz(i, j, k));
				//ex(i, j, k) = cex0 * ex(i, j, k)
				//	+ cexry0 * (hz(i, j, k) - hz(i, j-1, k))
				//	- cexrz0 * (hy(i, j, k) - hy(i, j, k-1));
				//a = a*b + ((c*d - ce) - (f*g - fh))
				__m512d cde = zcexry * _mm512_load_pd(&hz(i, j-1, k));
				cde = _mm512_fmsub_pd(zcexry, hz0, cde);

				__m512d hyk = _mm512_load_pd(&hy(i, j, k-8));
				hyk = (__m512d)_mm512_alignr_epi64((__m512i)hyk, (__m512i)hy0, 0);
				__m512d fgh = zcexrz * hyk; //_mm512_loadu_pd(&hy(i, j, k-1));
				fgh = _mm512_fmsub_pd(zcexrz, hy0, fgh);
				__m512d ab = _mm512_fmadd_pd(zcex, _mm512_load_pd(&ex(i, j, k)), cde - fgh);
				_mm512_store_pd(&ex(i, j, k), ab);

				//ey(i, j, k) = cey0 * ey(i, j, k)
				//	+ ceyrz0 * (hx(i, j, k) - hx(i, j, k-1))
				//	- ceyrx0 * (hz(i, j, k) - hz(i-1, j, k));
				__m512d hxk = _mm512_load_pd(&hx(i, j, k-8));
				hxk = (__m512d)_mm512_alignr_epi64((__m512i)hxk, (__m512i)hx0, 0);
				cde = zceyrz * hxk; //_mm512_loadu_pd(&hx(i, j, k-1);
				cde = _mm512_fmsub_pd(zceyrz, hx0, cde);
				fgh = zceyrx * _mm512_load_pd(&hz(i-1, j, k));
				fgh = _mm512_fmsub_pd(zceyrx, hz0, fgh);
				ab = _mm512_fmadd_pd(zcey, _mm512_load_pd(&ey(i, j, k)), cde - fgh);
				_mm512_store_pd(&ey(i, j, k), ab);

				//ez(i, j, k) = cez0 * ez(i, j, k)
				//	+ cezrx0 * (hy(i, j, k) - hy(i-1, j, k))
				//	- cezry0 * (hx(i, j, k) - hx(i, j-1, k));
				cde = zcezrx * _mm512_load_pd(&hy(i-1, j, k));
				cde = _mm512_fmsub_pd(zcezrx, hy0, cde);
				fgh = zcezry * _mm512_load_pd(&hx(i, j-1, k));
				fgh = _mm512_fmsub_pd(zcezry, hx0, fgh);
				ab = _mm512_fmadd_pd(zcez, _mm512_load_pd(&ez(i, j, k)), cde - fgh);
				_mm512_store_pd(&ez(i, j, k), ab);
			}
		}
	}
}


void electric_boundary_condition() {
	/*----------electric boundary condition----------*/
	int i, j, k;
	// set e values touching left wall to zero
	for (j = 0; j < ny; j++) {
		for (k = pad-1; k < nz+pad+1; k++) {
			ey(1, j, k) = 0.0;
			ez(1, j, k) = 0.0;
		}
	}

	// set e values touching bottom wall to 0
	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			ex(i, j, pad) = 0.0;
			ey(i, j, pad) = 0.0;
		}
	}
}


void ecur_source() {
	/*-----------electronic source----------*/
	int i, j, k;
	// plane wave
	i = 2;
	for (j = 1; j < ny-1; j++) {
		for (k = pad; k < nz+pad; k++) {
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

	__m128d xchxry, xchxrz,
		xchyrz, xchyrx,
		xchzrx, xchzry;
	__m512d zchxry, zchxrz,
		zchyrz, zchyrx,
		zchzrx, zchzry;

	for (i = 1; i < nx-1; i++) {
		for (j = 1; j < ny-1; j++) {
			for (k = pad; k < nz+pad; k += 8) {
				//id = media_id(i, j, k);
				if (media_row(i, j, k) < 8) {
					__m256i yid = _mm256_load_si256((__m256i *)&media_id(i, j, k));

					zchxry = _mm512_i32gather_pd(yid, &chxry[0], 8);
					zchxrz = _mm512_i32gather_pd(yid, &chxrz[0], 8);
					zchyrz = _mm512_i32gather_pd(yid, &chyrz[0], 8);
					zchyrx = _mm512_i32gather_pd(yid, &chyrx[0], 8);
					zchzrx = _mm512_i32gather_pd(yid, &chzrx[0], 8);
					zchzry = _mm512_i32gather_pd(yid, &chzry[0], 8);
				} else {
					id = media_id(i, j, k);
					xchxry = _mm_load_sd(&chxry[id]);
					zchxry = _mm512_broadcastsd_pd(xchxry);
					xchxrz = _mm_load_sd(&chxrz[id]);
					zchxrz = _mm512_broadcastsd_pd(xchxrz);

					xchyrz = _mm_load_sd(&chyrz[id]);
					zchyrz = _mm512_broadcastsd_pd(xchyrz);
					xchyrx = _mm_load_sd(&chyrx[id]);
					zchyrx = _mm512_broadcastsd_pd(xchyrx);

					xchzrx = _mm_load_sd(&chzrx[id]);
					zchzrx = _mm512_broadcastsd_pd(xchzrx);
					xchzry = _mm_load_sd(&chzry[id]);
					zchzry = _mm512_broadcastsd_pd(xchzry);
				}

				__m512d ex0 = _mm512_load_pd(&ex(i, j, k));
				__m512d ey0 = _mm512_load_pd(&ey(i, j, k));
				__m512d ez0 = _mm512_load_pd(&ez(i, j, k));

				//hx(i, j, k) = hx(i, j, k)
				//	- chxry0 * (ez(i, j+1, k) - ez(i, j, k))
				//	+ chxrz0 * (ey(i, j, k+1) - ey(i, j, k));
				//a = a - (b*c - bd) + (e*f - eg)
				__m512d bcd = _mm512_fmsub_pd(zchxry, _mm512_load_pd(&ez(i, j+1, k)), zchxry * ez0);
				__m512d eyk = _mm512_load_pd(&ey(i, j, k+8));
				eyk = (__m512d)_mm512_alignr_epi64((__m512i)ey0, (__m512i)eyk, 6);
				__m512d efg = _mm512_fmsub_pd(zchxrz, eyk /*_mm512_loadu_pd(&ey(i, j, k+1))*/, zchxrz * ey0);
				_mm512_store_pd(&hx(i, j, k), _mm512_load_pd(&hx(i, j, k)) - bcd + efg);

				//hy(i, j, k) = hy(i, j, k)
				//	- chyrz0 * (ex(i, j, k+1) - ex(i, j, k))
				//	+ chyrx0 * (ez(i+1, j, k) - ez(i, j, k));
				__m512d exk = _mm512_load_pd(&ex(i, j, k+8));
				exk = (__m512d)_mm512_alignr_epi64((__m512i)ex0, (__m512i)exk, 6);
				bcd = _mm512_fmsub_pd(zchyrz, exk /*_mm512_loadu_pd(&ex(i, j, k+1))*/, zchyrz * ex0);
				efg = _mm512_fmsub_pd(zchyrx, _mm512_load_pd(&ez(i+1, j, k)), zchyrx * ez0);
				_mm512_store_pd(&hy(i, j, k), _mm512_load_pd(&hy(i, j, k)) - bcd + efg);

				//hz(i, j, k) = hz(i, j, k)
				//	- chzrx0 * (ey(i+1, j, k) - ey(i, j, k))
				//	+ chzry0 * (ex(i, j+1, k) - ex(i, j, k));
				bcd = _mm512_fmsub_pd(zchzrx, _mm512_load_pd(&ey(i+1, j, k)), zchzrx * ey0);
				efg = _mm512_fmsub_pd(zchzry, _mm512_load_pd(&ex(i, j+1, k)), zchzry * ex0);
				_mm512_store_pd(&hz(i, j, k), _mm512_load_pd(&hz(i, j, k)) - bcd + efg);
			}
		}
	}
}

void output() {
	int i = 4, j = 1;
	int k = (nz+pad*2)/2+0;
	//	for (int i = 0; i < nx; i++) {
	//		for (int j = 0; j < ny; j++) {
				printf("%d %d %.12f\n", i, j, ez(i, j, k));
	//		}
	//	}
	//	printf("\n\n");
}

void free_all() {
	_mm_free(ex);
	_mm_free(ey);
	_mm_free(ez);
	_mm_free(hx);
	_mm_free(hy);
	_mm_free(hz);
	_mm_free(media_id);
	_mm_free(media_row);
	_mm_free(eps);
	_mm_free(mu);
	_mm_free(sig);
	_mm_free(cex);
	_mm_free(cey);
	_mm_free(cez);
	_mm_free(cexry);
	_mm_free(cexrz);
	_mm_free(ceyrz);
	_mm_free(ceyrx);
	_mm_free(cezrx);
	_mm_free(cezry);
	_mm_free(chxry);
	_mm_free(chxrz);
	_mm_free(chyrz);
	_mm_free(chyrx);
	_mm_free(chzrx);
	_mm_free(chzry);
}
