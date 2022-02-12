/* File: range.c
 * Name: D.Saravanan
 * Date: 02/12/2022
 * Program to print the range of values for certain data types
*/

#include <stdio.h>
#include <limits.h> /* integer specifications */
#include <float.h>  /* floating-point specifications */

/* range limits of certain types */
int main(void) {

	printf("Integer range: \t%d\t%d\n", INT_MIN, INT_MAX);
	printf("Long range: \t%ld\t%ld\n", LONG_MIN, LONG_MAX);
	printf("Float range: \t%e\t%e\n", FLT_MIN, FLT_MAX);
	printf("Double range: \t%e\t%e\n", DBL_MIN, DBL_MAX);
	printf("Long double range: \t%e\t%e\n", LDBL_MIN, LDBL_MAX);
	printf("Float-Double epsilon: \t%e\t%e\n", FLT_EPSILON, DBL_EPSILON);

	return 0;
}
