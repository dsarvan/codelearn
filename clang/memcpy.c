/* File: memcpy.c
 * Name: D.Saravanan
 * Date: 20/02/2024
 * Program to copy block of memory
*/

#include <stdio.h>
#include <string.h>

struct {
	char name[40];
	int age;
} person, person_copy;

int main() {

	/* void *memcpy(void *destination, const void *source, size_t num);

	   Copies the values of num bytes from the location pointed to by
	   source directly to the memory block pointed to by destination.

	   The underlying type of the objects pointed to by both the source
	   and destination pointers are irrelevant for this function. The
	   result is a binary copy of the data.

	   The function does not check for any terminating null character
	   in source - it always copies exactly num bytes.

	   To avoid overflows, the size of the arrays pointed to by both the
	   destination and source parameters, shall be at least num bytes, and
	   should not overlap (for overlapping memory blocks, memmove is a safer
	   approach).

	   Parameters:

		destination
			Pointer to the destination array where the content is to be
			copied, type-casted to a pointer of type void*.

		source
			Pointer to the source of data to be copied, type-casted to a
			pointer of type const void*.

		num
			Number of bytes to copy.
			size_t is an unsigned integral type.

	   Return Value:

		destination is returned.

	*/

	char myname[] = "Pierre de Fermat";

	/* using memcpy to copy string */
	memcpy(person.name, myname, strlen(myname)+1);
	person.age = 46;

	/* using memcpy to copy structure */
	memcpy(&person_copy, &person, sizeof(person));

	printf("person_copy: %s, %d\n", person_copy.name, person_copy.age);

	return 0;
}
