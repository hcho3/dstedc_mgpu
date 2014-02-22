#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "dstedc.h"
#include "timer.h"

int main(int argc, char **argv)
{
    long NGRP, NCORE, MAX_NCORE;
    // NGRP: # of compute groups
    // NCORE: total # of cores to use; compute groups will get fair shares.
    // MAX_NCORE: maximum # of cores
    const char *Din  = argv[3];
    const char *Ein  = argv[4];
    const char *Dout = argv[5];
    const char *Qout = argv[6];
	struct timeval timer1, timer2;
    
    long D_dims[2], E_dims[2], Q_dims[2];
    double *D;
    double *E;
    double *Q;
    long N;
    double *WORK;
    long *IWORK;

    MAX_NCORE = omp_get_num_procs();
    omp_set_nested(1);

    if (argc < 7 ||
        sscanf(argv[1], "%ld", &NGRP) < 1 || NGRP <= 0 ||
        sscanf(argv[2], "%ld", &NCORE) < 1 || NCORE <= 0 ||
        NCORE > MAX_NCORE || NGRP > NCORE) {
        printf("Usage: %s "
               "[# of compute groups] [total # of cores to use] "
               "[Din.bin] [E.bin] [Dout.bin] [Q.bin]\n", argv[0]);
        printf("This system has %ld CPU cores that can run in parallel.\n",
               MAX_NCORE);
        printf("Each compute group gets an equal share of available codes.\n");
        printf("So the number of compute groups should not exceed that of "
               "cores assigned.\n");
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

    printf("NGRP = %ld\n", NGRP);
    printf("NCORE = %ld\n", NCORE);
    printf("MAX_NCORE = %ld\n", MAX_NCORE);
    D = read_mat(Din, D_dims);
    E = read_mat(Ein, E_dims);
    N = (D_dims[0] > D_dims[1]) ? D_dims[0] : D_dims[1];

    Q_dims[0] = Q_dims[1] = N;
    Q = (double *)malloc(N * N * sizeof(double));

    WORK = allocate_work(N);
    IWORK = allocate_iwork(N);

    get_time(&timer1);
    dlaed0_m(NGRP, NCORE, N, D, E, Q, N, WORK, IWORK);
    get_time(&timer2);
    printf("Time: %.3lf s\n", get_elapsed_ms(timer1, timer2) / 1000.0 );

    write_mat(Dout, D, D_dims);
    write_mat(Qout, Q, Q_dims);

    free(Q);
    free(D);
    free(E);
    free_work(WORK);
    free_iwork(IWORK);

    return 0;
}
