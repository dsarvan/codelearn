/* File: program.c
 * Name: D.Saravanan
 * Date: 11/02/2024
 * Program source of the shared library program.so
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int compute(int nval) {
	static int counter = 0;
	counter = counter + nval;
	return counter;
}


double multiply(int xval, double yval) {
	double rval = xval * yval;
	return rval;
}


double mean(long *arr, long n) {
	double sum = 0;
	for (long i = 0; i < n; sum = sum + arr[i], i++);
	double mean = sum / (double)n;
	return mean;
}


double *matrixC(int nrow, int ncol) {
	double *C = (double *) calloc(nrow*ncol, sizeof(*C));
	for (size_t i = 0; i < nrow*ncol; C[i] = 1.0 * i, i++);
	for (size_t i = 0; i < nrow*ncol; printf("%lf\n", C[i]), i++);
	return C;
}


static int const t0 = 40;
static int const ke = 200;
static int const kc = ke/2;
static int const nsteps = 100;
static double const spread = 12;

int simulation(double *ex, double *hy) {

	/* FDTD loop */
	for (int ts = 1; ts <= nsteps; ts++) {

		/* calculate the Ex field */
		for (int k = 1; k < ke; k++)
			ex[k] = ex[k] + 0.5 * (hy[k-1] - hy[k]);

		/* put a Gaussian pulse in the middle */
		ex[kc] = exp(-0.5 * pow(((t0 - ts)/spread), 2));

		/* calculate the Hy field */
		for (int k = 0; k < ke-1; k++)
			hy[k] = hy[k] + 0.5 * (ex[k] - ex[k+1]);
	}

	return 0;
}
