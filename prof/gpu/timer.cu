#include <sys/time.h>
#include "timer.h"

void get_time(timeval *timer)
{
    gettimeofday(timer, NULL);
}

double get_elapsed_ms(timeval time1, timeval time2)
{
    int sec, usec;
    sec  = time2.tv_sec  - time1.tv_sec;
    usec = time2.tv_usec - time1.tv_usec;
    return (1000.*(double)(sec) + (double)(usec) * 0.001);
}
