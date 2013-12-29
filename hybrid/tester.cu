#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include "dstedc.h"

extern double *read_mat(const char *filename, const char *varname,
    size_t *dims);
extern void write_mat(const char *filename, const char *varname,
    double *array, size_t *dims);

double GetTimerValue(timeval time_1, timeval time_2);

int main(int argc, char **argv)
{
    const char *fin  = argv[1];
    const char *fout1 = argv[2];
    const char *fout2 = argv[3];
	timeval timer1, timer2;
    
    size_t D_dims[2], E_dims[2], Q_dims[2];
    double *D = read_mat(fin, "D", D_dims);
    double *E = read_mat(fin, "E", E_dims);
    double *Q;
    int N = (D_dims[0] > D_dims[1]) ? (int)D_dims[0] : (int)D_dims[1];
    double *WORK, *WORK_dev;
    int *IWORK;

    Q_dims[0] = Q_dims[1] = (size_t)N;
    //Q = (double *)malloc(N * N * sizeof(double));
    //WORK = (double *)malloc((2 * N + 2 * N * N) * sizeof(double));
    //IWORK = (int *)malloc((3 + 5 * N) * sizeof(int));
    Q = (double *)malloc(N * N * sizeof(double));
    cudaMallocHost((void **)&WORK, (2 * N + 2 * N * N) * sizeof(double));
    cudaMallocHost((void **)&IWORK, (3 + 5 * N) * sizeof(int));
    cudaMalloc((void **)&WORK_dev, (2 * N + 3 * N * N) * sizeof(double));

	gettimeofday(&timer1, NULL);
    dlaed0(N, D, E, Q, N, WORK, WORK_dev, IWORK);
	gettimeofday(&timer2, NULL);
	printf("Time: %.3lf s\n", GetTimerValue(timer1, timer2) / 1000.0 );

    write_mat(fout1, "D", D, D_dims);
    write_mat(fout2, "Q", Q, Q_dims);

    free(Q);
    cudaFreeHost(WORK);
    cudaFreeHost(IWORK);
    cudaFree(WORK_dev);
    free(D);
    free(E);

    return 0;
}

double GetTimerValue(timeval time_1, timeval time_2)
{
    int sec, usec;
    sec  = time_2.tv_sec  - time_1.tv_sec;
    usec = time_2.tv_usec - time_1.tv_usec;
    return (1000.*(double)(sec) + (double)(usec) * 0.001);
}
