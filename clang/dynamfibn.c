/* File: dynamfibn.c
 * Name: D.Saravanan
 * Date: 22/11/2022
 * Program to compute nth fibonacci using dynamic programming
*/

#include <stdio.h>
#include <gmp.h>

void fibonacci(mpz_t result, int n) {
    /* compute nth fibonacci */

    mpz_t fib[3];

    mpz_init_set_ui(fib[0], 0);
    mpz_init_set_ui(fib[1], 1);
    mpz_init_set_ui(fib[2], 1);

    for (size_t i = 1; i < n; i++) {
	mpz_add(fib[2], fib[0], fib[1]);
	mpz_set(fib[0], fib[1]);
	mpz_set(fib[1], fib[2]);
    }

    mpz_set(result, (n == 0) ? fib[0] : (n == 1) ? fib[1] : fib[2]);
}

int main() {
    int N = 10000;

    mpz_t result;
    mpz_init(result);

    fibonacci(result, N);

    gmp_printf("%Zd\n", result);
    mpz_clear(result);

    return 0;
}
