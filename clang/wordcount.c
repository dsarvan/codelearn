/* File: wordcount.c
 * Name: D.Saravanan
 * Date: 19/03/2023
 * Program to count lines, words, and characters
*/

#include <stdio.h>

int main() {

    /* count lines, words, and characters in input */

    int state = 0;
    int c, nl, nw, nc;
    nl = nw = nc = 0;

    while ((c = getchar()) != EOF) {
	++nc;
	if (c == '\n')
	    ++nl;
	if (c == ' ' || c == '\n' || c == '\t')
	    state = 0;
	else if (state == 0) {
	    state = 1;
	    ++nw;
	}
    }

    printf("%d %d %d\n", nl, nw, nc);

    return 0;
}
