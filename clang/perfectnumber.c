/* File: perfectnumber.c
 * Name: D.Saravanan
 * Date: 18/03/2023
 * Program to find a perfect number
*/


#include <stdio.h>

int main() {

    int n = 10000;

    for (int x = 2; x <= n; x++) {
	int sum = 0;
	for (int n = 1; n <= x / 2; n++) {
	    if (x % n == 0)
		sum += n;
	}
	if (x == sum)
	    printf("%d\n", x);
    }

    return 0;
}
