// File: fdtd.c
// Name: D.Saravanan
// Date: 11/11/2022

/* FDTD simulation of a pulse in free space after 100 time steps.
   The pulse originated in the center and travels outward. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int const t0 = 40;
static int const ke = 200;
static int const kc = ke/2;
static int const nsteps = 100;
static double const spread = 12;

void plotResults(int ke, double *ex, double *hy);

int main() {

	double *ex = (double *)malloc(ke * sizeof(double));
	double *hy = (double *)malloc(ke * sizeof(double));

	for (int ts = 1; ts <= nsteps; ts++) {

		for (int k = 1; k < ke; k++)
			ex[k] = ex[k] + 0.5 * (hy[k-1] - hy[k]);

		ex[kc] = exp(-0.5 * pow(((t0 - ts)/spread), 2));

		for (int k = 0; k < ke-1; k++)
			hy[k] = hy[k] + 0.5 * (ex[k] - ex[k+1]);
	}

	plotResults(ke, ex, hy);
	free(ex); free(hy);
	return 0;
}

void plotResults(int ke, double *ex, double *hy) {

	FILE *gnuplot = popen("gnuplot -persist", "w");

	fprintf(gnuplot, "set terminal pngcairo font 'Times,11'\n");
	fprintf(gnuplot, "unset key; set output 'fdtd.png'\n");
	fprintf(gnuplot, "set style line 1 lc rgb '#000000'; set multiplot " 
	"title 'FDTD simulation of a pulse in free space after 100 time steps' "
	"layout 2,1 margins 0.09,0.9,0.1,0.93 spacing 0.1\n");

	fprintf(gnuplot, "set xrange [0:200]; set yrange [-1.2:1.2]\n");
	fprintf(gnuplot, "set tics scale 0.5 font ',10'; set xtics 20\n");

	fprintf(gnuplot, "set xlabel ''; set ylabel 'E_{x}'\n");
	fprintf(gnuplot, "plot '-' w lines lc rgb '#000000'\n");
	for (int k = 0; k < ke; k++)
		fprintf(gnuplot, "%d %e\n", k, ex[k]);
	fprintf(gnuplot, "e\n");

	fprintf(gnuplot, "set xlabel 'FDTD cells'; set ylabel 'H_{y}'\n");
	fprintf(gnuplot, "plot '-' w lines lc rgb '#000000'\n");
	for (int k = 0; k < ke; k++) 
		fprintf(gnuplot, "%d %e\n", k, hy[k]);
	fprintf(gnuplot, "e\n");

	fprintf(gnuplot, "unset multiplot\n");
	fflush(gnuplot);

	pclose(gnuplot);
}
