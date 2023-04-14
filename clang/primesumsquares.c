/* File: primesumsquares.c
 * Name: D.Saravanan
 * Date: 13/04/2023
 * Program to check an odd prime number has sum of two squares
*/

#include <stdio.h>
#include <math.h>

int sumsquares(int nval) {
    /* An odd prime number p, the sum of
       two squares if and only if it leaves
       the remainder 1 on division by 4. */
    if (nval % 4 == 1)
	printf("%d\n", nval);

    return 0;
}


int prime(int number) {
    /* function to check prime */
    double sqrt_number = sqrt(number);
    for (int i = 2; i <= (int) sqrt_number; i++) {
	if (number % i == 0)
	    return 0;
    }
    return sumsquares(number);
}

int main() {
    int N = 1000;
    for (size_t n = 2; n <= N; n++)
	prime(n);

    return 0;
}
