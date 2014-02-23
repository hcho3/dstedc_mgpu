#include <stdlib.h>
#include <sys/time.h>
#include "timer.h"

void get_time(struct timeval *timer)
{
    gettimeofday(timer, NULL);
}

double get_elapsed_ms(struct timeval time1, struct timeval time2)
{
    int sec, usec;
    sec  = time2.tv_sec  - time1.tv_sec;
    usec = time2.tv_usec - time1.tv_usec;
    return (1000.*(double)(sec) + (double)(usec) * 0.001);
}
