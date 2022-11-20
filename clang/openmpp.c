/* File: openmpp.c
 * Name: D.Saravanan
 * Date: 20/11/2022
 * A parallel OpenMP program that implements multiple threads
*/

#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {

	int nthreads, thread_id;

	printf("Goodbye slow serial world and Hello OpenMP!\n");
	#pragma omp parallel 
	{
		int nthreads = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		printf("I have %d thread(s) and my thread id is %d\n", nthreads, thread_id);
	}

	return 0;
}
