#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "dstedc.h"
#include "safety.h"

long max_matsiz_gpu(long NGPU)
{
    long gpu_mem;
    long max_matsiz_gpu;

    cudaDeviceProp cdp;
    long i;
    long min = -5L;

    for (i = 0; i < NGPU; i++) {
        safe_cudaGetDeviceProperties(&cdp, i);
        if ((long)cdp.totalGlobalMem < min || min < 0L)
            min = cdp.totalGlobalMem;
    }

    gpu_mem = min;
    max_matsiz_gpu = (long)floor(sqrt(2.4*gpu_mem/8.0) / 3.0);

    return max_matsiz_gpu;
}

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
    double *WORK;
    long maxN = max_matsiz_host();

    if (N > maxN) {
        printf("The input matrix is too big!\n"
               "The main memory will hold only %ldx%ld\n", maxN, maxN);
        exit(1);
    }

    safe_cudaHostAlloc((void **)&WORK,
        (2*N + 2*N*N) * sizeof(double), cudaHostAllocPortable);

    return WORK;
}

double **allocate_work_dev(long NGPU, long N)
{
    double **WORK_dev;
    long i;
    long maxN = max_matsiz_gpu(NGPU);

    if (N > maxN)
        N = maxN;

    WORK_dev = (double **)malloc(NGPU * sizeof(double *));
    for (i = 0; i < NGPU; i++) {
        safe_cudaSetDevice(i);
        safe_cudaMalloc((void **)&WORK_dev[i],
            (3*N*N) * sizeof(double));
    }

    return WORK_dev;
}

long *allocate_iwork(long N)
{
    long *IWORK;
    safe_cudaHostAlloc((void **)&IWORK, (5*N) * sizeof(long),
        cudaHostAllocPortable);

    return IWORK;
}

void free_work(double *WORK)
{
    safe_cudaFreeHost(WORK);
}

void free_work_dev(double **WORK_dev, long NGPU)
{
    long i;

    for (i = 0; i < NGPU; i++) {
        safe_cudaSetDevice(i);
        safe_cudaFree(WORK_dev[i]);
    }

    free(WORK_dev);
}

void free_iwork(long *IWORK)
{
    safe_cudaFreeHost(IWORK);
}
