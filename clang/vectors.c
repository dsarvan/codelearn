/* File: vectors.c
 * Name: D.Saravanan
 * Date: 22/11/2022
 * Vector addition
*/

#include <stdio.h>
#include <time.h>
#include "timer.h"

#define N 80000000
static double a[N], b[N], c[N];

int main(int argc, char *argv[]) {

	struct timespec tstart;
	double time_sum = 0.0;

	for (int i=0; i<N; i++) {
		a[i] = 1.0;
		b[i] = 2.0;
	}

	cpu_timer_start(&tstart);
	for (int i=0; i<N; i++)
		c[i] = a[i] + b[i];
	time_sum += cpu_timer_stop(tstart);

	printf("Runtime is %lf ms\n", time_sum);
	return 0;
}
