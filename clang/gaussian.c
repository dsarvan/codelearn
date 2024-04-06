/* File: gaussian.c
 * Name: D.Saravanan
 * Date: 17/03/2024
 * Program to generate isolated Gaussian pulse
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * Return evenly spaced numbers over a specified interval.
 *
 * :param double x1: the starting value of the sequence
 * :param double x2: the end value of the sequence
 * :param int n: number of samples to generate
 *
 * :return: n equally spaced samples in the close interval
 */
double *linspace(double x1, double x2, int n) {
	double *x = (double *) calloc(n, sizeof(*x));
	double st = (x2 - x1)/(double)(n - 1);
	for (int i = 0; i < n; x[i] = x1 + (double) i * st, i++);
	return x;
}

/**
 * Generate isolated Gaussian pulse
 *
 * :param int fs: sampling frequency in Hz
 * :param double sigma: pulse width in seconds
 */
double *gaussian(int fs, double sigma) {
	double *t = (double *) calloc(fs, sizeof(*t));
	double *g = (double *) calloc(fs, sizeof(*g));

	t = linspace(-0.5, 0.5, fs);
	double sval = 1 / (sqrt(2*M_PI) * sigma);

	for (int i = 0; i < fs; g[i] = sval * exp(-0.5 * pow(t[i] / sigma, 2)), i++);
	return g;
}

int main() {

	int fs = 80;
	double sigma = 0.1;

	double *g = (double *) calloc(fs, sizeof(*g));
	g = gaussian(fs, sigma);

	for (int i = 0; i < fs; printf("%lf\n", g[i]), i++);
	return 0;
}
