#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "dstedc.h"

long max_matsiz_host(void)
{
    long num_pages = sysconf( _SC_PHYS_PAGES );
    long page_size = sysconf( _SC_PAGESIZE ); 
    long host_mem = num_pages * page_size;
    long max_matsiz_host;

    max_matsiz_host
        = (long)floor((-7.0 + sqrt(49.0 + 0.8 * host_mem)) / 4.0);

    return max_matsiz_host;
}

double *allocate_work(long N)
{
    long maxN = max_matsiz_host();

    //printf("maxN = %ld\n", maxN);

    if (N > maxN) {
        printf("The input matrix is too big!\n"
               "The main memory will hold only %ldx%ld\n", maxN, maxN);
        exit(1);
    }

    return (double *)malloc((2*N + 2*N*N) * sizeof(double));
}

long *allocate_iwork(long N)
{
    return (long *)malloc((5*N) * sizeof(long));
}

void free_work(double *WORK)
{
    free(WORK);
}

void free_iwork(long *IWORK)
{
    free(IWORK);
}
