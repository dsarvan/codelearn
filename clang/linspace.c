/* File: linspace.c
 * Name: D.Saravanan
 * Date: 12/03/2023
 * Program to evaluate a num linearly spaced intervals
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double linspace(int N, double a, double b) {

	double h = (b - a)/N;

	double *x = (double *) calloc(N+1, sizeof(*x));
	double *f = (double *) calloc(N+1, sizeof(*f));

	for (size_t n = 0; n <= N; n++) {
		x[n] = a + (b - a) * n / N;
		f[n] = sin(x[n]);
	}

	free(x);

	double sum(double f[], int nval) {
		double sval = 0.0;
		for (size_t i = 1; i <= nval; i++)
			sval += f[i];
		return sval;
	}

	double trap;
	trap = (h/2) * (f[0] + 2 * sum(f, N-1) + f[N]);

	free(f);

	return trap;
}


double trapezoid(int N, double a, double b) {

	double h = (b - a)/N;

	double *x = (double *) calloc(N+1, sizeof(*x));
	x[0] = a; x[N] = b;
	for (size_t i = 1; i < N; i++)
		x[i] = x[i-1] + (x[N] - x[0])/N;

	double *f = (double *) calloc(N+1, sizeof(*f));
	for (size_t i = 0; i <= N; i++)
		f[i] = sin(x[i]);

	free(x);
	
	double sum(double f[], int nval) {
		double sval = 0.0;
		for (size_t i = 1; i <= nval; i++)
			sval += f[i];
		return sval;
	}

	double trap;
	trap = (h/2) * (f[0] + 2 * sum(f, N-1) + f[N]);

	free(f);

	return trap;
}


int main() {

	int n = 100000000;	/* number of trapezoids */
	double a = 0.0;		/* left endpoint */
	double b = 10.0;	/* right endpoint */

	printf("%f\n", linspace(n, a, b));
	printf("%f\n", trapezoid(n, a, b));

	return 0;
}
