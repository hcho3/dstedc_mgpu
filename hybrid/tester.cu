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
    int NGPU, MAX_NGPU;
    const char *fin  = argv[2];
    const char *fout1 = argv[3];
    const char *fout2 = argv[4];
	timeval timer1, timer2;
    
    size_t D_dims[2], E_dims[2], Q_dims[2];
    double *D;
    double *E;
    double *Q;
    int N;
    double **WORK;
    double **WORK_dev;
    int **IWORK;

    int i;

    cudaGetDeviceCount(&MAX_NGPU);

    if (argc < 5 || sscanf(argv[1], "%d", &NGPU) < 1 ||
        NGPU <= 0 || NGPU > MAX_NGPU) {
        printf("Usage: %s [# of GPUs] [input.mat] [D.mat] [Q.mat]\n", argv[0]);
        printf("The number of GPUs must be between 1 and %d.\n", MAX_NGPU);
        printf("[input.mat]: name of mat file containing the diagonal and "
               "tridiagonal of the input matrix.\n");
        printf("[D.mat]: name of mat file that will contain the eigenvalues"
               " on exit.\n");
        printf("[Q.mat]: name of mat file that will contain the eigenvectors"
               " on exit.\n");
        return 1;
    }

    printf("NGPU = %d\n", NGPU);
    D = read_mat(fin, "D", D_dims);
    E = read_mat(fin, "E", E_dims);
    N = (D_dims[0] > D_dims[1]) ? (int)D_dims[0] : (int)D_dims[1];

    Q_dims[0] = Q_dims[1] = (size_t)N;
    Q = (double *)malloc(N * N * sizeof(double));

    WORK = (double **)malloc(NGPU * sizeof(double *));
    WORK_dev = (double **)malloc(NGPU * sizeof(double *));
    IWORK = (int **)malloc(NGPU * sizeof(int *));
    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaMallocHost((void **)&WORK[i], (2*N + N*N) * sizeof(double));
        cudaMallocHost((void **)&IWORK[i], (3 + 5 * N) * sizeof(int));
        cudaMalloc((void **)&WORK_dev[i], (3*N*N) * sizeof(double));
    }

	gettimeofday(&timer1, NULL);
    dlaed0_m(NGPU, N, D, E, Q, N, WORK, WORK_dev, IWORK);
	gettimeofday(&timer2, NULL);
	printf("Time: %.3lf s\n", GetTimerValue(timer1, timer2) / 1000.0 );

    write_mat(fout1, "D", D, D_dims);
    write_mat(fout2, "Q", Q, Q_dims);

    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaFree(WORK_dev[i]);
        cudaFreeHost(WORK[i]);
        cudaFreeHost(IWORK[i]);
    }

    free(Q);
    free(D);
    free(E);
    free(WORK);
    free(WORK_dev);
    free(IWORK);

    return 0;
}

double GetTimerValue(timeval time_1, timeval time_2)
{
    int sec, usec;
    sec  = time_2.tv_sec  - time_1.tv_sec;
    usec = time_2.tv_usec - time_1.tv_usec;
    return (1000.*(double)(sec) + (double)(usec) * 0.001);
}
