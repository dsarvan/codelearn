/* File: fdtdd.c
 * Name: D.Saravanan
 * Date: 05/12/2022
 * FDTD simulation of a pulse in free space after 100 time steps.
 * The pulse originated in the center and travels outward.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define t0 40.0
#define nsteps 100
#define spread 12

int main(void) {

	int k, ts;
	int ke = 200;
	int kc = ke/2;

	double *ex = (double *) calloc(ke, sizeof(*ex));
	double *hy = (double *) calloc(ke, sizeof(*hy));

	for (ts = 1; ts <= nsteps; ts++) {

		for (k = 1; k < ke; k++) {
			ex[k] = ex[k] + 0.5 * (hy[k-1] - hy[k]);
		}

		ex[kc] = exp(-0.5 * pow(((t0 - ts)/spread), 2));

		for (k = 0; k < ke-1; k++) {
			hy[k] = hy[k] + 0.5 * (ex[k] - ex[k+1]);
		}
	}

	for (k = 0; k < ke; k++) printf("%d %e %e\n", k, ex[k], hy[k]);
	free(ex); free(hy);
	return 0;
}
