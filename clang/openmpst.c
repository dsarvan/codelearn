/* File: openmpst.c
 * Name: D.Saravanan
 * Date: 20/11/2022
 * A parallel OpenMP program that implements multiple threads
 * Adding a single pragma to print output
*/

#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {

	printf("Goodbye slow serial world and Hello OpenMP!\n");

	/* Adding a parallel region */
	#pragma omp parallel /* Spawn threads */
	{
		/* Variables defined in a parallel region are private */
		int nthreads = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		#pragma omp single
		{
			/* Places output statements into an OpenMP single pragma block */
			printf("I have %d thread(s) and my thread id is %d\n", nthreads, thread_id);
		} /* Impiled Barrier */
	} /* Implied Barrier */

	return 0;
}
