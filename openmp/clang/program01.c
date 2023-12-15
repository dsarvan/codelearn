/* File: program01.c
 * Name: D.Saravanan
 * Date: 20/11/2023
 * Program that implements single thread
*/

#include <stdio.h>
#include <omp.h> /* includes OpenMP header file for the OpenMP function calls */

int main(int argc, char *argv[]) {

	int nthreads = omp_get_num_threads(); /* number of threads */
	int thread_id = omp_get_thread_num(); /* thread ID */

	printf("I have %d thread(s) and my thread id is %d\n", nthreads, thread_id);

	return 0;
}
