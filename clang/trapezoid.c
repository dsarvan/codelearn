/* File: trapezoid.c
 * Name: D.Saravanan
 * Date: 08/03/2023
 * Program for trapezoid rule
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


double trapezoid(int N, int rank, int size, double a, double b) {
    /* numerical integration */

    int wload[size];
    for (size_t i = 0; i < size; i++)
	wload[i] = N / size;

    for (size_t i = 0; i < N % size; i++)
	wload[i] += 1;

    /* trapezoid base length */
    double h = (b - a) / N;

    double la = a + rank * wload[rank] * h;
    double lb = la + wload[rank] * h;

    double *x = (double *) calloc(wload[rank] + 1, sizeof(*x));
    x[0] = la;
    x[wload[rank]] = lb;
    for (size_t i = 1; i < wload[rank]; i++)
	x[i] = x[i - 1] + (x[wload[rank]] - x[0]) / (wload[rank]);

    double *f = (double *) calloc(wload[rank] + 1, sizeof(*f));
    for (size_t i = 0; i <= wload[rank]; i++)
	f[i] = sin(x[i]);

    free(x);

    double sum(double f[], int nval) {
	double sval = 0.0;
	for (size_t i = 1; i <= nval; i++)
	    sval += f[i];
	return sval;
    }

    double ltrap, trap;
    ltrap = (h / 2) * (f[0] + 2 * sum(f, wload[rank] - 1) + f[wload[rank]]);
    MPI_Reduce(&ltrap, &trap, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    free(f);

    if (rank == 0) {
	return trap;
    }
}


int main(int argc, char **argv) {

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* integral parameters */
    int n = 100000000;		/* number of trapezoids */
    double a = 0.0;		/* left endpoint */
    double b = M_PI;		/* right endpoint */

    double integral = trapezoid(n, rank, size, a, b);
    if (rank == 0) {
	printf("%f\n", integral);
    }

    MPI_Finalize();
    return 0;
}
