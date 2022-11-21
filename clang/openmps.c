/* File: openmps.c
 * Name: D.Saravanan
 * Date: 20/11/2022
 * A serial OpenMP program that implements single thread
*/

#include <stdio.h>
#include <omp.h> /* Includes OpenMP header file for the OpenMP function calls */

int main(int argc, char *argv[]) {

	int nthreads, thread_id;

	nthreads = omp_get_num_threads(); /* number of threads */
	thread_id = omp_get_thread_num(); /* thread ID */
	printf("Goodbye slow serial world and Hello OpenMP!\n");
	printf("I have %d thread(s) and my thread id is %d\n", nthreads, thread_id);

	return 0;
}
