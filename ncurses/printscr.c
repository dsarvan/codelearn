/* File: printscr.c
 * Name: D.Saravanan
 * Date: 05/12/2020
 * Program to print text in terminal screen
*/

#include <ncurses.h>

int main(void) {

    initscr();
    addstr("Institute of Mathematical Sciences");
    refresh();
    getch();

    endwin();
    return 0;
}
