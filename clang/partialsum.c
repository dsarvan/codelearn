/* File: partialsum.c
 * Name: D.Saravanan
 * Date: 28/11/2022
 * Program to compute partial sum with OpenMP
*/

#include <stdio.h>
#include <omp.h>

int main(void) {
	int partial_sum, total_sum;

	#pragma omp parallel private(partial_sum) shared(total_sum)
	{
		partial_sum = 0;
		total_sum = 0;

		#pragma omp for
		for(int i = 1; i <= 80000000; i++) {
			partial_sum += i;
		}

		#pragma omp critical
		{
			total_sum += partial_sum;
		}
	}

	printf("Total sum: %d\n", total_sum);
	return 0;
}
