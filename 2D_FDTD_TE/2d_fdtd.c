#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

const int it = 10000; // time steps
const int iy = 41, iz = 41; // dimension
const double freq = 1.0e9; // frequency (Hx)

int main(int argc, char *argv[]) {
	/*----------initialize parameters----------*/
	int mt = 2, mdy = 41, mdz = 41;
	double cvel = 2.99792458e8;
	double ey[mt][mdy][mdz], ez[mt][mdy][mdz];
	double hx[mt][mdy][mdz];
	double sigmay[mdy][mdz], sigmaz[mdy][mdz],
		epsiry[mdy][mdz], epsirz[mdy][mdz], amux[mdy][mdz];

	double alamda = cvel / freq; // wave length
	double dy = alamda / 20.0; // cell size
	double dz = alamda / 20.0; // cell size
	double dt = 1.0e-12; // delta t
	int icyc = 1.0 / freq / dt; // steps per lambda
	double pi = 3.141592654; // pi
	double epsir0 = 8.8541878e-12; // epsiron 0
	double amu0 = 1.0e-7 * 4.0 * pi; // mu 0
	double sigma0 = 1.0e-8; // sigma 0

	// init mu, eps, sigma
	int jy, jz;
	for (jy = 0; jy < mdy; jy++) {
		for (jz = 0; jz < mdz; jz++) {
			amux[jy][jz] = amu0;
			epsiry[jy][jz] = epsir0;
			epsirz[jy][jz] = epsir0;
			sigmay[jy][jz] = sigma0;
			sigmaz[jy][jz] = sigma0;
		}
	}

	// init e, h
	
	for (jy = 0; jy < iy; jy++) {
		for (jz = 0; jz < iz; jz++) {
			ey[0][jy][jz] = 0.0;
			ez[0][jy][jz] = 0.0;
			hx[0][jy][jz] = 0.0;
		}
	}


	/*----------fdtd routine----------*/
	int ky0 = 21, kz0 = 21;
	int ii = 0, jt = 0, kt = 0;

L50:
	kt = kt + 1;

	// source
	double ax = sin( (fmod(kt-1, icyc) + 1) * 2.0 * pi / (double)icyc);
	hx[jt][ky0][kz0] = 1.0 / 376.7 * ax;

	// compute hx
	double dey, dez;
	for (jy = 0; jy < iy-1; jy++) {
		for (jz = 0; jz < iz-1; jz++) {
			dey = (ey[jt][jy][jz+1] - ey[jt][jy][jz]) / dz;
			dez = (ez[jt][jy+1][jz] - ez[jt][jy][jz]) / dy;
			hx[jt+1][jy][jz] = hx[jt][jy][jz] + dt / amux[jy][jz] * (dey - dez);
		}
	}

	// compute ez
	double term1, term2, dhx, dhy;
	for (jy = 1; jy < iy-1; jy++) {
		for (jz = 1; jz < iz-1; jz++) {
			term1 = (1.0 - sigmaz[jy][jz] * dt / (2.0 * epsirz[jy][jz]))
				/ (1.0 + sigmaz[jy][jz] * dt / (2.0 * epsirz[jy][jz]));
			term2 = 1.0 / (1.0 + sigmaz[jy][jz] * dt / (2.0 * epsirz[jy][jz]));
			dhy = 0.0;
			dhx = (hx[jt+1][jy][jz] - hx[jt+1][jy-1][jz]) / dy;
			ez[jt+1][jy][jz] = term1 * ez[jt][jy][jz] + dt
				/ epsirz[jy][jz] * term2 * (dhy - dhx);
		}
	}

	// compute ey
	double dhz;
	for (jy = 1; jy < iy-1; jy++) {
		for (jz = 1; jz < iz-1; jz++) {
			term1 = (1.0 - sigmay[jy][jz] * dt / (2.0 * epsiry[jy][jz]))
				/ (1.0 + sigmay[jy][jz] * dt / (2.0 * epsiry[jy][jz]));
			term2 = 1.0 / (1.0 + sigmay[jy][jz] * dt / (2.0 * epsiry[jy][jz]));
			dhx = (hx[jt+1][jy][jz] - hx[jt+1][jy][jz-1]) / dz;
			dhz = 0.0;
			ey[jt+1][jy][jz] = term1 * ey[jt][jy][jz] + dt
				/ epsiry[jy][jz] * term2 * (dhx - dhz);
		}
	}
	
	// apply Perfect Electronic Conductor
	for (jy = 0; jy < iy; jy++) {
		ey[jt+1][jy][0] = 0.0;
		ey[jt+1][jy][iz-1] = 0.0;
	}
	for (jz = 0; jz < iz; jz++) {
		ez[jt+1][0][jz] = 0.0;
		ez[jt+1][iy-1][jz] = 0.0;
	}

	if (kt > it) goto L900;

	// output
	if (kt % 10 == 0) {
	for (jy = 0; jy < iy; jy++) {
		for (jz = 0; jz < iz; jz++) {
			printf("%d %d %.13f\n", jy, jz, hx[jt][jy][jz]);
		}
	}
	printf("\n\n");}

	// exchange e and h
	for (jy = 0; jy < iy; jy++) {
		for (jz = 0; jz < iz; jz++) {
			hx[jt][jy][jz] = hx[jt+1][jy][jz];
			ey[jt][jy][jz] = ey[jt+1][jy][jz];
			ez[jt][jy][jz] = ez[jt+1][jy][jz];
		}
	}

	goto L50;

L900:
	return 0;
}



