/* File: doublfib.c
 * Name: D.Saravanan
 * Date: 25/11/2022
 * Program to compute nth fibonacci using doubling method
*/

#include <stdio.h>
#include <gmp.h>


mpz_t *fibonacci(mpz_t * result, int n) {
    /* compute nth fibonacci */

    mpz_t fib[4];

    mpz_init_set_ui(fib[0], 0);
    mpz_init_set_ui(fib[1], 1);
    mpz_init_set_ui(fib[2], 1);
    mpz_init_set_ui(fib[3], 2);

    int nval = 0;
    for (size_t i = n; i > 0; nval++, i >>= 1);

    mpz_t xval, yval;
    mpz_init_set_ui(xval, 0);
    mpz_init_set_ui(yval, 0);

    for (int m = nval - 1; m >= 0; m--) {

	mpz_mul_ui(xval, fib[1], 2);
	mpz_sub(yval, xval, fib[0]);
	mpz_mul(fib[2], fib[0], yval);

	mpz_pow_ui(xval, fib[0], 2);
	mpz_pow_ui(yval, fib[1], 2);
	mpz_add(fib[3], xval, yval);

	if ((n >> m) & 1) {
	    mpz_set(fib[0], fib[3]);
	    mpz_add(fib[1], fib[2], fib[3]);
	} else {
	    mpz_set(fib[0], fib[2]);
	    mpz_set(fib[1], fib[3]);
	}
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
