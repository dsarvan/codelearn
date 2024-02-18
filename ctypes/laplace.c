/* File: laplace.c
 * Name: D.Saravanan
 * Date: 18/02/2024
 * Program to solve the laplace equation
*/

#include <stdio.h>
#include <math.h>

double time_step(double *u, int nx, int ny, double dx, double dy) {

	double tmp, err, diff, dx2, dy2, dnr_inv;

	err = 0.0;
	dx2 = dx*dx;
	dy2 = dy*dy;
	dnr_inv = 0.5/(dx2*dy2);

	for (int i = 1; i < nx - 1; ++i) {
		for (int j = 1; j < ny - 1; ++j) {
			tmp = u[i*nx + j];
			u[i*nx + j] = ((u[(i-1)*nx + j] + u[(i+1)*nx + j]) * dy2 +
						   (u[i*nx + j - 1] + u[i*nx + j + 1]) * dx2) * dnr_inv;

			diff = u[i*nx + j] - tmp;
			err += diff * diff;
			}
	}

	return sqrt(err);
}

double solve_equation(double *u, int nx, int ny, double dx, double dy) {
	/* solve the laplace equation */

	int iter = 0;
	double err = 1;

	while(iter < 10000 && err > 1e-6) {
		err = time_step(u, nx, ny, dx, dy);
		iter++;
	}

	return err;
}
