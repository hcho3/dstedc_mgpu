#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "dstedc.h"

#ifdef DEBUG
#define check( x ) _check( (x), __LINE__ )
#else
#define check( x ) (x)
#endif

static void _check(cudaError_t cs, long line)
{
    const char *errstr;

    if (cs != cudaSuccess) {
        errstr = cudaGetErrorString(cs);
        printf("CUDA error %s at %ld.\n", errstr, line);
        exit(1);
    }
}

long max_matsiz_gpu(long NGPU)
{
    long gpu_mem;
    long max_matsiz_gpu;

    cudaDeviceProp cdp;
    long i;
    long min = -5L;

    for (i = 0; i < NGPU; i++) {
        cudaGetDeviceProperties(&cdp, i);
        if (cdp.totalGlobalMem < min || min < 0L)
            min = cdp.totalGlobalMem;
    }

    gpu_mem = min;
    max_matsiz_gpu = (long)floor(sqrt(2.4*gpu_mem/8.0) / 3.0);

    return max_matsiz_gpu;
}

long max_matsiz_host(long NGPU)
{
    long num_pages = sysconf( _SC_PHYS_PAGES );
    long page_size = sysconf( _SC_PAGESIZE ); 
    long host_mem = num_pages * page_size;
    long max_matsiz_host;

    max_matsiz_host
        = (long)floor(
            (sqrt(52.0*NGPU*NGPU+(0.4*host_mem-12)*NGPU+0.4*host_mem)-8.0*NGPU)
            / (2.0*(NGPU+1))); 

    return max_matsiz_host;
}

double **allocate_work(long NGPU, long N)
{
    double **WORK;
    long i;
    long maxN = max_matsiz_host(NGPU);

    if (N > maxN) {
        printf("The input matrix is too big!\n"
               "The main memory will hold only %ldx%ld\n", maxN, maxN);
        exit(1);
    }

    WORK = (double **)malloc(NGPU * sizeof(double *));

    cudaSetDevice(0);
    check(cudaMallocHost((void **)&WORK[0],
        (2*N + 2*N*N) * sizeof(double)));
    for (i = 1; i < NGPU; i++) {
        cudaSetDevice(i);
        check(cudaMallocHost((void **)&WORK[i],
            (2*N + N*N) * sizeof(double)));
    }

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
        cudaSetDevice(i);
        check(cudaMalloc((void **)&WORK_dev[i],
            (3*N*N) * sizeof(double)));
    }

    return WORK_dev;
}

long **allocate_iwork(long NGPU, long N)
{
    long **IWORK;
    long i;

    IWORK = (long **)malloc(NGPU * sizeof(long *));
    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        check(cudaMallocHost((void **)&IWORK[i], (3 + 5*N) * sizeof(long)));
    }

    return IWORK;
}

void free_work(double **WORK, long NGPU)
{
    long i;

    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaFreeHost(WORK[i]);
    }

    free(WORK);
}

void free_work_dev(double **WORK_dev, long NGPU)
{
    long i;

    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaFree(WORK_dev[i]);
    }

    free(WORK_dev);
}

void free_iwork(long **IWORK, long NGPU)
{
    long i;

    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaFreeHost(IWORK[i]);
    }

    free(IWORK);
}
