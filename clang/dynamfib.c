/* File: dynamfib.c
 * Name: D.Saravanan
 * Date: 22/11/2022
 * Program to compute nth fibonacci using dynamic programming
*/

#include <stdio.h>
#include <gmp.h>


mpz_t *fibonacci(mpz_t * result, int n) {
    /* compute nth fibonacci */

    mpz_t fib[2];

    mpz_init_set_ui(fib[0], 0);
    mpz_init_set_ui(fib[1], 1);

    for (size_t i = 0; i < n; i++) {
	mpz_add(fib[0], fib[0], fib[1]);
	mpz_swap(fib[0], fib[1]);
    }

    mpz_set(*result, fib[0]);
    return result;
}


int main() {

    int N = 10000;

    mpz_t result;
    mpz_init(result);

    gmp_printf("%Zd\n", fibonacci(&result, N));

    return 0;
}
