/* File: fdtdt.c
 * Name: D.Saravanan
 * Date: 11/11/2022
 * FDTD simulation of a pulse in free space after 100 time steps.
 * The pulse originated in the center and travels outward. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <util/xmalloc.h>
#include <math.h>

#define t0 40.0
#define ke 200
#define nsteps 100
#define spread 12

int main(void) {

	int k, ts;
	int kc = ke/2;

	double *ex = (double *) xcalloc(ke, sizeof(*ex));
	double *hy = (double *) xcalloc(ke, sizeof(*hy));

	/* FDTD loop */
	for (ts = 1; ts <= nsteps; ts++) {

		/* calculate the Ex field */
		for (k = 1; k < ke; k++) {
			ex[k] = ex[k] + 0.5 * (hy[k-1] - hy[k]);
		}

		/* put a Gaussian pulse in the middle */
		ex[kc] = exp(-0.5 * pow(((t0 - ts)/spread), 2));

		/* calculate the Hy field */
		for (k = 0; k < ke-1; k++) {
			hy[k] = hy[k] + 0.5 * (ex[k] - ex[k+1]);
		}
	}

	free(ex); free(hy);
	return 0;
}
