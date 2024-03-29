/* File: arrayex.c
 * Name: D.Saravanan
 * Date: 29/03/2023
 * Program for dynamic sized arrays
*/

#include <stdio.h>
#include <stdlib.h>

int main() {
    int n;
    printf("Size of the array: ");
    scanf("%d", &n);

    /* Creating enough space for 'n' inegets. */
    int *arr = calloc(n, sizeof(int));

    if (arr == NULL) {
	printf("Unable to allocate memory\n");
	return -1;
    }
    printf("Enter the elements (space/newline separated): ");
    for (int i = 0; i < n; i++)
	scanf("%d", arr + i);	/* (arr + i) points to ith element */

    printf("Given array: ");
    for (int i = 0; i < n; i++)
	printf("%d ", *(arr + i));   /* Dereferencing the pointer */
    printf("\n");

    printf("Removing first element i.e., arr[0] = %d.\n", arr[0]);
    for (int i = 1; i < n; i++)
	arr[i - 1] = arr[i];
    arr = realloc(arr, (n - 1) * sizeof(int));

    printf("Modified Array: ");
    for (int i = 0; i < n; i++)
	printf("%d ", arr[i]);
    printf("\n");

    free(arr);
    return 0;
}
