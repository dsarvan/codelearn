/* File: wave.c
 * Name: D.Saravanan
 * Date: 23/11/2022
 * Program for wave simulation 
*/

#include <stdio.h>
#include "timestep.h"

#define N 10000000
static double H[N], U[N], V[N], dx[N], dy[N];
static int celltype[N];

int main(int argc, char *argv[]) {
	
	double mindt;
	double g = 9.80, sigma = 0.95;

	/* Initializes arrays and data */
	for (int ic = 0; ic < N; ic++) {
		H[ic] = 10.0;
		U[ic] = 0.0;
		V[ic] = 0.0;
		dx[ic] = 0.5;
		dy[ic] = 0.5;
		celltype[ic] = REAL_CELL;
	}

	H[N/2] = 20.0;

	/* Calls the timestep calculation */
	mindt = timestep(N, g, sigma, celltype, H, U, V, dx, dy);

	printf("Minimum dt is %lf\n", mindt);
}
