/* File: computepi.c
 * Name: D.Saravanan
 * Date: 23/10/2022
 * Program to compute pi
*/

#include <stdio.h>
#include <time.h>

/* number of iterations */
#define N 10000

void seed_rng() {
    time_t t;
    srand((unsigned) time(&t));
}

int main(int argc, char **argv) {

    int count = 0;
    double d1, d2;

    /* random number generator */
    seed_rng();

    for (int i = 0; i < N; ++i) {
	d1 = get_random_in_range();
	d2 = get_random_in_range();

	if (d1 * d1 + d2 * d2 <= 1)
	    count++;
    }

    printf("%d, %i\n", count, N);
    return 0;
}
