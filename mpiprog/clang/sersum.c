/* File: sersum.c
 * Name: D.Saravanan
 * Date: 03/12/2023
 * Program to initialize an array, assign values and compute sum
*/

#include <stdio.h>
#include <stdlib.h>

int main() {

    int N = 200000000; /* array size */
    double *data = (double *) calloc(N, sizeof(*data));	/* declare array */

    /* initialize array */
    for (size_t i = 0; i < N; data[i] = i * 1.0, i++);

    /* assign values and compute sum */
	double sum = 0;
    for (size_t i = 0; i < N; i++) {
		data[i] = data[i] + data[i];
		sum += data[i];
	}

    free(data);
	printf("Final sum = %e\n", sum);

    return 0;
}
