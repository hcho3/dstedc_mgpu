#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "dstedc.h"

long max_matsiz_gpu(int NGPU)
{
    long gpu_mem;
    long max_matsiz_gpu;

    cudaDeviceProp cdp;
    int i;
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

long max_matsiz_host(int NGPU)
{
    long num_pages = sysconf( _SC_PHYS_PAGES );
    long page_size = sysconf( _SC_PAGESIZE ); 
    long host_mem = num_pages * page_size;
    long max_matsiz_host;

    max_matsiz_host = (long)floor(sqrt(13.0 + 0.8*host_mem/8.0/NGPU) - 4.0); 

    return max_matsiz_host;
}

double **allocate_work(int NGPU, int N)
{
    double **WORK;
    int i;
    int maxN = (int)max_matsiz_host(NGPU);

    if (N > maxN) {
        printf("The input matrix is too big!\n"
               "The main memory will hold only %dx%d\n", maxN, maxN);
        exit(1);
    }

    WORK = (double **)malloc(NGPU * sizeof(double *));
    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaMallocHost((void **)&WORK[i], (2*N + N*N) * sizeof(double));
    }

    return WORK;
}

double **allocate_work_dev(int NGPU, int N)
{
    double **WORK_dev;
    int i;
    int maxN = (int)max_matsiz_gpu(NGPU);

    if (N > maxN)
        N = maxN;

    WORK_dev = (double **)malloc(NGPU * sizeof(double *));
    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaMalloc((void **)&WORK_dev[i], (3*N*N) * sizeof(double));
    }

    return WORK_dev;
}

int **allocate_iwork(int NGPU, int N)
{
    int **IWORK;
    int i;

    IWORK = (int **)malloc(NGPU * sizeof(int *));
    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaMallocHost((void **)&IWORK[i], (3 + 5*N) * sizeof(int));
    }

    return IWORK;
}

void free_work(double **WORK, int NGPU)
{
    int i;

    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaFreeHost(WORK[i]);
    }

    free(WORK);
}

void free_work_dev(double **WORK_dev, int NGPU)
{
    int i;

    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaFree(WORK_dev[i]);
    }

    free(WORK_dev);
}

void free_iwork(int **IWORK, int NGPU)
{
    int i;

    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaFreeHost(IWORK[i]);
    }

    free(IWORK);
}
