/* File: serprog04.c
 * Name: D.Saravanan
 * Date: 03/12/2023
 * Program to initialize an array, assign values and compute sum
*/

#include <stdio.h>
#include <stdlib.h>

int main() {

    size_t N = 200000000; /* array size */
    double *data = (double *) calloc(N, sizeof(*data));

    printf("Starting serial array example ...\n");
    printf("Using an array of %d floats requires %ld bytes\n", N, sizeof(*data));

    /* initializes array */
    printf("Initializing array ...\n");
    for (size_t i = 0; i < N; data[i] = i * 1.0, i++);

    /* assign values and compute sum */
    printf("Performing computation on array elements ...\n");
	double sum = 0;
    for (size_t i = 0; i < N; i++) {
		data[i] = data[i] + data[i];
		sum += data[i];
	}

    /* print sample results */
    printf("Sample results:\n");
    for (size_t i = 10; i < N; i *= 10)
		printf("\t data[%d] = %e\n", i, data[i]);

	printf("\nFinal sum = %e\n", sum);

    free(data);

    return 0;
}
