/* File: fdtdt.c
 * Name: D.Saravanan
 * Date: 11/11/2022
 * FDTD simulation of a pulse in free space after 100 time steps.
 * The pulse originated in the center and travels outward. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define t0 40.0
#define nsteps 100
#define spread 12

void plotResults(int ke, double *ex, double *hy);

void plotResults(int ke, double *ex, double *hy) {

	FILE *gnuplot = popen("gnuplot -persist", "w");

	fprintf(gnuplot, "set terminal pngcairo font 'Times,11'\n");
	fprintf(gnuplot, "unset key; set output 'fdtdt.png'\n");
	fprintf(gnuplot, "set style line 1 lc rgb '#000000'; set multiplot " 
	"title 'FDTD simulation of a pulse in free space after 100 time steps' "
	"layout 2,1 margins 0.09,0.9,0.1,0.93 spacing 0.1\n");

	fprintf(gnuplot, "set xrange [0:200]; set yrange [-1.2:1.2]\n");
	fprintf(gnuplot, "set tics scale 0.5 font ',10'; set xtics 20\n");

	fprintf(gnuplot, "set xlabel ''; set ylabel 'E_{x}'\n");
	fprintf(gnuplot, "plot '-' w lines lc rgb '#000000'\n");
	for (int k = 0; k < ke; k++) fprintf(gnuplot, "%d %e\n", k, ex[k]);
	fprintf(gnuplot, "e\n");

	fprintf(gnuplot, "set xlabel 'FDTD cells'; set ylabel 'H_{y}'\n");
	fprintf(gnuplot, "plot '-' w lines lc rgb '#000000'\n");
	for (int k = 0; k < ke; k++) fprintf(gnuplot, "%d %e\n", k, hy[k]);
	fprintf(gnuplot, "e\n");

	fprintf(gnuplot, "unset multiplot\n");
	fflush(gnuplot);

	pclose(gnuplot);
}

int main(void) {

	int k, ts;
	int ke = 200;
	int kc = ke/2;

	double *ex = (double *) calloc(ke, sizeof(*ex));
	double *hy = (double *) calloc(ke, sizeof(*hy));

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

	plotResults(ke, ex, hy);
	free(ex); free(hy);
	return 0;
}
