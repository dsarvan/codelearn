/* File: openmpp.c
 * Name: D.Saravanan
 * Date: 20/11/2022
 * A parallel OpenMP program that implements multiple threads
*/

#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {

	/* Variables are defined before the parallel region, thus these are
	   shared variables in the heap. Each thread writes to these, and the
	   final value is determined by which one writes last. */
	int nthreads, thread_id;

	printf("Goodbye slow serial world and Hello OpenMP!\n");

	/* Adding a parallel region */
	#pragma omp parallel /* Spawn threads */
	{
		nthreads = omp_get_num_threads();
		thread_id = omp_get_thread_num();
		printf("I have %d thread(s) and my thread id is %d\n", nthreads, thread_id);
	} /* Implied Barrier */

	return 0;
}
