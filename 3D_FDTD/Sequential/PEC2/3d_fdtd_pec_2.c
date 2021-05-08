#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>

double ecur(double t, double pi, double freq) {
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

int main(int argc, char *argv[]) {
	/*----------constants----------*/
	const double pi = 3.141592653589793;
	const double c = 2.998e8;
	const double epsilon0 = 8.854e-12;
	const double mu0 = 4.0e-7 * pi;
	const double sig0 = 1.0e-8;
	const double z0 = 376.734309182110; // impedance

	/*----------source freqency----------*/
	double freq;

	/*----------time step----------*/
	int ntime;
	double time, dt;

	/*----------field----------*/
	int mx = 100, my = 100, mz = 3;
	int nx, ny, nz;
	double dx, dy, dz;
	double ex[mx+1][my+1][mz+1], ey[mx+1][my+1][mz+1], ez[mx+1][my+1][mz+1];
	double hx[mx+1][my+1][mz+1], hy[mx+1][my+1][mz+1], hz[mx+1][my+1][mz+1];

	/*----------media----------*/
	int mmedia = 10;
	int nmedia;
	double eps[mmedia];
	double mu[mmedia];
	double sig[mmedia];

	int media_id[mx][my][mz];

	/*----------media update constant----------*/
	double cex0, cey0, cez0;
	double cexry0, cexrz0,
		ceyrz0, ceyrx0,
		cezrx0, cezry0;
	double chxry0, chxrz0,
		chyrz0, chyrx0,
		chzrx0, chzry0;
	double cex[mmedia], cey[mmedia], cez[mmedia];
	double cexry[mmedia], cexrz[mmedia],
		ceyrz[mmedia], ceyrx[mmedia],
		cezrx[mmedia], cezry[mmedia];
	double chxry[mmedia], chxrz[mmedia],
		chyrz[mmedia], chyrx[mmedia],
		chzrx[mmedia], chzry[mmedia];

	/*----------main routine----------*/

	/*----------initialize----------*/
	/*----------source frequency----------*/
	freq = 8e9;

	/*----------cell count----------*/
	nx = 100;
	ny = 100;
	nz = 3;

	/*----------cell size----------*/
	/* Courant stability condition
	   vdt < 1/sqrt((1/dx)^2+(1/dy)^2+(1/dz)^2)
	   vdt < dx/sqrt(3) when dx=dy=dz
	*/
	dx = 29.1e-3 / 10.0;
	dy = dx;
	dz = dx;
	dt = 1.0 / freq / 100;

	ntime = 5000; // step count

	/*----------media settings----------*/
	nmedia = 2;

	// 0:vacuum
	eps[0] = epsilon0;
	mu[0] = mu0;
	sig[0] = sig0;

	// 1:PEC, PMC (in other routine)

	/*----------media constants----------*/
	// 0: vacuum
	cex0 = 1.0;
	cey0 = 1.0;
	cez0 = 1.0;

	cexry0 = (dt / epsilon0) / dy;
	cexrz0 = (dt / epsilon0) / dz;
	ceyrz0 = (dt / epsilon0) / dz;
	ceyrx0 = (dt / epsilon0) / dx;
	cezrx0 = (dt / epsilon0) / dx;
	cezry0 = (dt / epsilon0) / dy;

	chxry0 = (dt / mu0) / dy;
	chxrz0 = (dt / mu0) / dz;
	chyrz0 = (dt / mu0) / dz;
	chyrx0 = (dt / mu0) / dx;
	chzrx0 = (dt / mu0) / dx;
	chzry0 = (dt / mu0) / dy;

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

	/*----------modeling----------*/
	int i, j, k;

	// init with vacuum(=0)
	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			for (k = 0; k < nz; k++) {
				media_id[i][j][k] = 0;
			}
		}
	}

	// Perfect Conductor walls
	i = nx / 2;
	for (j = 0; j < ny / 2; j++) {
		for (k = 0; k < nz; k++) {
			media_id[i][j][k] = 1;
		}
	}
	
	// init with PEC(=1)
	// up and down
	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			media_id[i][j][0] = 1;
			media_id[i][j][nz-1] = 1;
		}
	}
	// left and right
	for (j = 0; j < ny; j++) {
		for (k = 0; k < nz; k++) {
			media_id[0][j][k] = 1;
			media_id[nx-1][j][k] = 1;
		}
	}
	// front
	for (i = 0; i < nx; i++) {
		for (k = 0; k < nz; k++) {
			media_id[i][0][k] = 1;
		}
	}
	
	/*----------FDTD routine----------*/
	int id;
	time = 0.0;

	for (int n = 0; n < ntime; n++) {
		/*----------update electric field----------*/
		//Ex
		for (i = 0; i < nx; i++) {
			for (j = 1; j < ny; j++) {
				for (k = 1; k < nz; k++) {
					id = media_id[i][j][k];

					if (id == 0) {
						// free space
						ex[i][j][k] = cex0 * ex[i][j][k]
							+ cexry0 * (hz[i][j][k] - hz[i][j-1][k])
							- cexrz0 * (hy[i][j][k] - hy[i][j][k-1]);
					} else if (id == 1) {
						// perfect conductor
						ex[i][j][k] = 0.0;
					} else {
						// arbitrary media
						ex[i][j][k] = cex[id] * ex[i][j][k]
							+ cexry[id] * (hz[i][j][k] - hz[i][j-1][k])
							- cexrz[id] * (hy[i][j][k] - hy[i][j][k-1]);
					}
				}
			}
		}

		// Ey
		for (i = 1; i < nx; i++) {
			for (j = 0; j < ny; j++) {
				for (k = 1; k < nz; k++) {
					if (id == 0) {
						// free space
						ey[i][j][k] = cey0 * ey[i][j][k]
							+ ceyrz0 * (hx[i][j][k] - hx[i][j][k-1])
							- ceyrx0 * (hz[i][j][k] - hz[i-1][j][k]);
					} else if (id == 1) {
						// perfect conductor
						ey[i][j][k] = 0.0;
					} else {
						// arbitrary media
						ey[i][j][k] = cey[id] * ey[i][j][k]
							+ ceyrz[id] * (hx[i][j][k] - hx[i][j][k-1])
							- ceyrx[id] * (hz[i][j][k] - hz[i-1][j][k]);
					}
				}
			}
		}

		// Ez
		for (i = 1; i < nx; i++) {
			for (j = 1; j < ny; j++) {
				for (k = 0; k < nz; k++) {
					id = media_id[i][j][k];
					
					if (id == 0) {
						// free space
						ez[i][j][k] = cez0 * ez[i][j][k]
							+ cezrx0 * (hy[i][j][k] - hy[i-1][j][k])
							- cezry0 * (hx[i][j][k] - hx[i][j-1][k]);
					} else if (id == 1) {
						// perfect conductor
						ez[i][j][k] = 0.0;
					} else {
						// arbitrary media
						ez[i][j][k] = cez[id] * ez[i][j][k]
							+ cezrx[id] * (hy[i][j][k] - hy[i-1][j][k])
							- cezry[id] * (hx[i][j][k] - hx[i][j-1][k]);
					}
				}
			}
		}

		/*----------electric boundary condition----------*/
		// set e values touching left wall to zero 
		for (j = 0; j < ny; j++) {
			for (k = 0; k < nz; k++) {
				ey[1][j][k] = 0.0;
				ez[1][j][k] = 0.0;
			}
		}
		
		// set e values touching bottom wall to 0
		for (i = 0; i < nx; i++) {
			for (j = 0; j < ny; j++) {
				ex[i][j][1] = 0.0;
				ey[i][j][1] = 0.0;
			}
		}

		/*-----------electronic source----------*/
		// plane wave
		i = 2;
		for (j = 0; j < ny-1; j++) {
			for (k = 0; k < nz-1; k++) {
				ez[i][j][k] = ecur(time - dt / 2.0, pi, freq);
				hx[i][j][k] = ez[i][j][k] / z0;
			}
		}

		time = time + dt / 2.0;
		/*----------update magnetic field----------*/
		// Hx
		for (i = 1; i < nx; i++) {
			for (j = 0; j < ny; j++) {
				for (k = 0; k < nz; k++) {
					id = media_id[i][j][k];

					if (id == 0) {
						// free space
						hx[i][j][k] = hx[i][j][k]
							- chxry0 * (ez[i][j+1][k] - ez[i][j][k])
							+ chxrz0 * (ey[i][j][k+1] - ey[i][j][k]);
					} else if (id == 1) {
						// perfect conductor
						hx[i][j][k] = 0.0;
					} else {
						// arbitrary media
						hx[i][j][k] = hx[i][j][k]
							- chxry[id] * (ez[i][j+1][k] - ez[i][j][k])
							+ chxry[id] * (ey[i][j][k+1] - ey[i][j][k]);
					}
				}
			}
		}

		// Hy
		for (i = 0; i < nx; i++) {
			for (j = 1; j < ny; j++) {
				for (k = 0; k < nz; k++) {
					id = media_id[i][j][k];

					if (id == 0) {
						// free space
						hy[i][j][k] = hy[i][j][k]
							- chyrz0 * (ex[i][j][k+1] - ex[i][j][k])
							+ chyrx0 * (ez[i+1][j][k] - ez[i][j][k]);
					} else if (id == 1) {
						// perfect conductor
						hy[i][j][k] = 0.0;
					} else {
						// arbitrary media
						hy[i][j][k] = hy[i][j][k]
							- chyrz[id] * (ex[i][j][k+1] - ex[i][j][k])
							+ chyrx[id] * (ez[i+1][j][k] - ez[i][j][k]);
					}
				}
			}
		}

		// Hz
		for (i = 0; i < nx; i++) {
			for (j = 0; j < ny; j++) {
				for (k = 1; k < nz; k++) {
					id = media_id[i][j][k];

					if (id == 0) {
						// free space
						hz[i][j][k] = hz[i][j][k]
							- chzrx0 * (ey[i+1][j][k] - ey[i][j][k])
							+ chzry0 * (ex[i][j+1][k] - ex[i][j][k]);
					} else if (id == 1) {
						// perfect conductor
						hz[i][j][k] = 0.0;
					} else {
						// arbitrary media
						hz[i][j][k] = hz[i][j][k]
							- chzrx[id] * (ey[i+1][j][k] - ey[i][j][k])
							+ chzry[id] * (ex[i][j+1][k] - ex[i][j][k]);
					}
				}
			}
		}
		time = time + dt / 2.0;

		/*----------output----------*/
		if (n % 10 == 0) {
		for (i = 0; i < nx; i++) {
			for (j = 0; j < ny; j++) {
				printf("%d %d %.12f\n", i, j, ez[i][j][(int)(nz / 2)]);
			}
		}
		printf("\n\n");
		}
		
	}
	return 0;
}
	
