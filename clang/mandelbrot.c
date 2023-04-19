/* File: mandelbrot.c
 * Name: D.Saravanan
 * Date: 17/04/2023
 * Program for the Mandelbrot set
*/

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

int **mandelbrot(int rmin, int rmax, int imin, int imax) {
    /* an algorithm to generate an image of the Mandelbrot set */

    int width = 512;
    int height = 512;
    int max_iters = 256;
    double upper_bound = 2.5;

    double *rval = (double *) calloc(width, sizeof(double));
    rval[0] = rmin;
    rval[width - 1] = rmax;
    for (size_t i = 0; i < width - 1; i++)
	rval[i + 1] = rval[i] + (rval[width - 1] - rval[0]) / (width - 1);

    double *ival = (double *) calloc(height, sizeof(double));
    ival[0] = imin;
    ival[height - 1] = imax;
    for (size_t i = 0; i < height - 1; i++)
	ival[i + 1] =
	    ival[i] + (ival[height - 1] - ival[0]) / (height - 1);

    /* we will represent members as 0, non-members as 1 */
    int **mandelbrot_graph = NULL;
    mandelbrot_graph = (int **) calloc(height, sizeof(int *));
    for (int i = 0; i < height; i++)
	mandelbrot_graph[i] = (int *) calloc(width, sizeof(int));

    for (int x = 0; x < width; x++) {
	for (int y = 0; y < height; y++) {
	    double complex c = rval[x] + ival[y] * I;
	    double complex z = 0. * I;

	    for (int n = 0; n < max_iters; n++) {
		z = (z * z) + c;

		if (abs(z) > upper_bound) {
		    mandelbrot_graph[y][x] = 1;
		    break;
		}
	    }
	}
    }

    free(rval);
    free(ival);

    return mandelbrot_graph;

    for (int i = 0; i < height; i++)
	free(mandelbrot_graph[i]);
    free(mandelbrot_graph);
}


int main() {

    int **mandel = mandelbrot(-2, 2, -2, 2);

    FILE *gnuplot = popen("gnuplot -persist", "w");
    fprintf(gnuplot, "set terminal pngcairo font 'Times,12'\n");
    fprintf(gnuplot, "set output 'mandelbrot.png'\n");
    fprintf(gnuplot, "set xrange[-2:2]; set yrange[-2:2]\n");
    fprintf(gnuplot, "set palette defined (0 'blue', 1 'white')\n");
    fprintf(gnuplot, "set cbrange [0:1]; set autoscale cbfix\n");
    fprintf(gnuplot, "plot '-' matrix with image notitle\n");
    for (int i = 0; i < 512; i++) {
	for (int j = 0; j < 512; j++) {
	    fprintf(gnuplot, "%e\n", mandel[i][j]);
	}
    }
    fprintf(gnuplot, "e\n");
    pclose(gnuplot);

    return 0;
}
