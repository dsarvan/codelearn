/* File: plotting.c
 * Name: D.Saravanan
 * Date: 12/11/2022
 * Plotting with gnuplot in C
*/

#include <stdio.h>

int main() {

	FILE *gnuplot = popen("gnuplot -persist", "w");
	
	if (gnuplot) {
		fprintf(gnuplot, "unset border\n");
		fprintf(gnuplot, "set samples 200\n");
		fprintf(gnuplot, "set tics font ',10'\n");
		fprintf(gnuplot, "set autoscale xfixmin\n");
		fprintf(gnuplot, "set autoscale yfixmin\n");
		fprintf(gnuplot, "set terminal pngcairo\n");
		fprintf(gnuplot, "set output 'plotting.png'\n");
		fprintf(gnuplot, "set xlabel 'x' font ',11'\n");
		fprintf(gnuplot, "set ylabel 'exp(-x^2)' font ',11'\n");
		fprintf(gnuplot, "plot exp(-x**2) smooth freq w boxes\n");
	}

	pclose(gnuplot);
	return 0;
}
