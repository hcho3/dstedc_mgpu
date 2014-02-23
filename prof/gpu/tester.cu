#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dstedc.h"
#include "safety.h"
#include "timer.h"

int main(int argc, char **argv)
{
    int temp;
    long NGPU, MAX_NGPU;
    const char *Din  = argv[2];
    const char *Ein  = argv[3];
    const char *Dout = argv[4];
    const char *Qout = argv[5];
	timeval timer1, timer2;
    
    long D_dims[2], E_dims[2], Q_dims[2];
    double *D;
    double *E;
    double *Q;
    long N;
    double *WORK;
    double **WORK_dev;
    long *IWORK;

    safe_cudaGetDeviceCount(&temp);
    MAX_NGPU = (long)temp;

    if (argc < 6 || sscanf(argv[1], "%ld", &NGPU) < 1 ||
        NGPU <= 0 || NGPU > MAX_NGPU) {
        printf("Usage: %s [# of GPUs] [Din.bin] [E.bin] [Dout.bin] [Q.bin]\n",
            argv[0]);
        printf("The number of GPUs must be between 1 and %ld.\n", MAX_NGPU);
        printf("[Din.bin]: file containing the diagonal of the input "
               "tridiagonal matrix.\n");
        printf("[Ein.bin]: file containing the subdiagonal of the input "
               "tridiagonal matrix.\n");
        printf("[Dout.bin]: file that will contain the eigenvalues of the "
               "input matrix on exit.\n");
        printf("[Q.bin]: file that will contain the eigenvectors of the "
               "input matrix on exit.\n");
        return 1;
    }

    printf("NGPU = %ld\n", NGPU);
    D = read_mat(Din, D_dims);
    E = read_mat(Ein, E_dims);
    N = (D_dims[0] > D_dims[1]) ? D_dims[0] : D_dims[1];

    Q_dims[0] = Q_dims[1] = N;
    Q = (double *)malloc(N * N * sizeof(double));

    WORK = allocate_work(N);
    WORK_dev = allocate_work_dev(NGPU, N);
    IWORK = allocate_iwork(N);

    get_time(&timer1);
    dlaed0_m(NGPU, N, D, E, Q, N, WORK, WORK_dev, IWORK);
    get_time(&timer2);
    printf("Time: %.3lf s\n", get_elapsed_ms(timer1, timer2) / 1000.0 );

    write_mat(Dout, D, D_dims);
    write_mat(Qout, Q, Q_dims);

    free(Q);
    free(D);
    free(E);
    free_work(WORK);
    free_work_dev(WORK_dev, NGPU);
    free_iwork(IWORK);

    return 0;
}
