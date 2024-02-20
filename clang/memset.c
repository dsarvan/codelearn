/* File: memset.c
 * Name: D.Saravanan
 * Date: 20/02/2024
 * Program to fill block of memory
*/

#include <stdio.h>
#include <string.h>

int main() {

	/* void *memset(void *ptr, int value, size_t num);

	   Sets the first num bytes of the block of memory pointed
	   by ptr to the specified value (interpreted as an unsigned char).

	  Parameters:

		ptr
			Pointer to the block of memory to fill.

		value
			Value to be set. The value is passed as an int, but the
			function fills the block of memory using the unsigned char
			conversion of this value.

		num
			Number of bytes to be set to the value.
			size_t is an unsigned integral type.

	  Return Value:

		ptr is returned.

	*/

	char str[] = "almost every programmer should know memset!";
	memset(str, '-', 6);
	puts(str);
	return 0;
}
