#include <stdlib.h>
#include "dstedc.h"

double **allocate_work(int NGPU, int N)
{
    double **WORK;
    int i;

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
