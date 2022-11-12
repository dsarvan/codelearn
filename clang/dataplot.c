/* File: dataplot.c
 * Name: D.Saravanan
 * Date: 12/11/2022
 * Plotting data points with gnuplot in C
*/

#include <stdio.h>

int main() {
	
	int x[] = {2015, 2016, 2017, 2018, 2019, 2020};
	int y[] = {344, 543, 433, 232, 212, 343};

	FILE *gnuplot = popen("gnuplot -persistent", "w");

	fprintf(gnuplot, "set style line 1 linecolor rgb '#0060ad'\n");
	fprintf(gnuplot, "set linetype 1 linewidth 1\n");
	fprintf(gnuplot, "set title 'Data Plot' font ',11'\n");
	fprintf(gnuplot, "set autoscale xfixmin\n");
	fprintf(gnuplot, "set autoscale yfixmin\n");
	fprintf(gnuplot, "set xlabel 'x' font ',11'\n");
	fprintf(gnuplot, "set ylabel 'y' font ',11'\n");
	fprintf(gnuplot, "set tics font ',10'\n");
	fprintf(gnuplot, "set terminal pngcairo\n");
	fprintf(gnuplot, "set output 'dataplot.png'\n");
	fprintf(gnuplot, "plot '-' with linespoints ls 1\n");
	for (int i = 0; i < 6; i++) {
		fprintf(gnuplot, "%d %d\n", x[i], y[i]);
	}

	pclose(gnuplot);
	return 0;
}
