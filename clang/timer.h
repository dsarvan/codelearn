#ifndef TIMER_H
#define TIMER_H
#include <time.h>

/* page no.: 620 */
void cpu_timer_start(struct timespec *tstart_cpu);
double cpu_timer_stop(struct timespec tstart_cpu);
#endif
