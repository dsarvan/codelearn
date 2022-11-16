/* File: parallelwave.c
 * Name: D.Saravanan
 * Date: 25/10/2022
 * Program to compute sine wave with mpi
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#define PI M_PI

int main(int argc, char **argv) {

	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	unsigned N = 10000000;
	double *xval = (double *) malloc(N * sizeof(double));

	xval[0] = -2*PI; xval[N-1] = 2*PI;
	for (size_t i = 0; i < N-1; i++)
		xval[i+1] = xval[i] + (xval[N-1] - xval[0])/(N-1);

	int workloads[size];
	for (size_t i = 0; i < size; i++)
		workloads[i] = N/size;

	for (size_t i = 0; i < N%size; i++)
		workloads[i] += 1;

	size_t sload = 0;
	for (size_t n = 0; n < rank; n++)
		sload += workloads[n];
	size_t eload = sload + workloads[rank];

	size_t n = eload - sload;
	double *wave = (double *) malloc(n * sizeof(double));
	for (size_t i = 0; i < n; i++)
		wave[i] = sin(xval[i+sload]);

	double *rwave = (double *) malloc(N * sizeof(double));
	MPI_Gather(wave, n, MPI_DOUBLE, rwave, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(wave);

	if (rank == 0) {

		FILE *gnuplot = popen("gnuplot -persist", "w");

		fprintf(gnuplot, "set terminal pngcairo font 'Palatino,12'\n");
		fprintf(gnuplot, "set output 'parallelwave.png'\n");
		fprintf(gnuplot, "set style line 1 linecolor rgb '#e50000'\n");
		fprintf(gnuplot, "set linetype 1 linewidth 1\n");
		fprintf(gnuplot, "set title 'Sine wave computed with mpi'\n");
		fprintf(gnuplot, "set yrange [-1.5:1.5]; set grid; set key off\n");
		fprintf(gnuplot, "set autoscale xfixmin; set autoscale xfixmax\n");
		fprintf(gnuplot, "set xlabel '{/OpenSymbol x}' font ',11'\n");
		fprintf(gnuplot, "set ylabel '{/OpenSymbol f(x)}' font ',11'\n");
		fprintf(gnuplot, "set tics scale 0.5 font ',10'\n");
		fprintf(gnuplot, "set xtics pi format '%.0Ppi'\n");
		fprintf(gnuplot, "plot '-' with lines ls 1\n");
		for (int i = 0; i < N; i++)
			fprintf(gnuplot, "%e %e\n", xval[i], rwave[i]);

		pclose(gnuplot);
	}

	free(xval);
	free(rwave);

	MPI_Finalize();
	return 0;
}
