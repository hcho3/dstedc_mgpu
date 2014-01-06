#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include "dstedc.h"

#define MAX_NGPU 8

extern double *read_mat(const char *filename, const char *varname,
    size_t *dims);
extern void write_mat(const char *filename, const char *varname,
    double *array, size_t *dims);

int get_ngpu(void);
double GetTimerValue(timeval time_1, timeval time_2);

int main(int argc, char **argv)
{
    const char *fin  = argv[1];
    const char *fout1 = argv[2];
    const char *fout2 = argv[3];
	timeval timer1, timer2;
    int ngpu;
    
    size_t D_dims[2], E_dims[2], Q_dims[2];
    double *D = read_mat(fin, "D", D_dims);
    double *E = read_mat(fin, "E", E_dims);
    double *Q;
    int N = (D_dims[0] > D_dims[1]) ? (int)D_dims[0] : (int)D_dims[1];
    double *WORK[MAX_NGPU];
    double *WORK_dev[MAX_NGPU];
    int *IWORK[MAX_NGPU];

    int i;

    ngpu = get_ngpu();
    printf("NGPUS = %d\n", ngpu);

    Q_dims[0] = Q_dims[1] = (size_t)N;
    Q = (double *)malloc(N * N * sizeof(double));

    for (i = 0; i < ngpu; i++) {
        cudaSetDevice(i);
        cudaMallocHost((void **)&WORK[i], (2*N + N*N) * sizeof(double));
        cudaMallocHost((void **)&IWORK[i], (3 + 5 * N) * sizeof(int));
        cudaMalloc((void **)&WORK_dev[i], (2*N + 3*N*N) * sizeof(double));
    }

	gettimeofday(&timer1, NULL);
    dlaed0_m(ngpu, N, D, E, Q, N, WORK, WORK_dev, IWORK);
	gettimeofday(&timer2, NULL);
	printf("Time: %.3lf s\n", GetTimerValue(timer1, timer2) / 1000.0 );

    write_mat(fout1, "D", D, D_dims);
    write_mat(fout2, "Q", Q, Q_dims);

    for (i = 0; i < ngpu; i++) {
        cudaSetDevice(i);
        cudaFree(WORK_dev[i]);
        cudaFreeHost(WORK[i]);
        cudaFreeHost(IWORK[i]);
    }

    free(Q);
    free(D);
    free(E);

    return 0;
}

int get_ngpu(void)
{
    const char *res;
    int ngpu;

    if ( (res = getenv("NUM_GPUS")) == NULL) {
        printf("NUM_GPUS was not set and thus defaults to 1.\n");
        return 1;
    } else {
        if (sscanf(res, "%d", &ngpu) < 1 || ngpu < 1 || ngpu > MAX_NGPU) {
            printf("NUM_GPUS must be a positive integer between 1 and %d.\n",
                MAX_NGPU);
            exit(1);
        }
        return ngpu;
    }
}

double GetTimerValue(timeval time_1, timeval time_2)
{
    int sec, usec;
    sec  = time_2.tv_sec  - time_1.tv_sec;
    usec = time_2.tv_usec - time_1.tv_usec;
    return (1000.*(double)(sec) + (double)(usec) * 0.001);
}
