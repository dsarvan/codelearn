/* File: fib.c
 * Name: D.Saravanan
 * Date: 14/07/2021
 * Fibonacci number calculation
*/

int fib(int num);

int fib(int num) {

	if (num <= 0) return -1;
	else if (num == 1) return 0;
	else if (num == 2 || num == 3) return 1;
	else return fib(num-2) + fib(num-1);
}
