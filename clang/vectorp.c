/* File: vectorp.c
 * Name: D.Saravanan
 * Date: 22/11/2022
 * Vector add with a simple loop-level OpenMP pragma and vectorization
*/

#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "timer.h"

#define N 80000000
static double a[N], b[N], c[N];

int main(int argc, char *argv[]) {
	
	#pragma omp parallel
		if (omp_get_thread_num() == 0)
			printf("Running with %d thread(s)\n", omp_get_num_threads());

	struct timespec tstart;
	double time_sum = 0.0;

	/* Initialization pragma so first touch
	gets memory in the proper location */
	#pragma omp parallel for simd
	for (int i=0; i<N; i++) {
		a[i] = 1.0;
		b[i] = 2.0;
	}

	cpu_timer_start(&tstart);
	/* OpenMP for pragma to distribute work
	for vector add loop across threads */
	#pragma omp parallel for simd
	for (int i=0; i<N; i++)
		c[i] = a[i] + b[i];
	time_sum += cpu_timer_stop(tstart);

	printf("Runtime is %lf ms\n", time_sum);
	return 0;
}
