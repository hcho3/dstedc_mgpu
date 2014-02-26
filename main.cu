#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "dstedc.h"
#include "safety.h"
#include "timer.h"

int main(int argc, char **argv)
{
    int temp;
    long NGPU, MAX_NGPU;
    long NCPUW, MAX_NCPU;
    const char *Din  = argv[3];
    const char *Ein  = argv[4];
    const char *Dout = argv[5];
    const char *Qout = argv[6];
    cfg_ent cfg;
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

    MAX_NCPU = omp_get_num_procs();
    omp_set_nested(1);

    if (argc < 7 || sscanf(argv[1], "%ld", &NGPU) < 1 ||
        sscanf(argv[2], "%ld", &NCPUW) < 1 ||
        NGPU <= 0 || NCPUW <= 0 || NGPU > MAX_NGPU ||
        NGPU+NCPUW > MAX_NCPU) {
        printf("Usage: %s [# of GPUs] [# of CPU workers] [Din.bin] [E.bin] "
               "[Dout.bin] [Q.bin]\n", argv[0]);
        printf("This system has %ld GPU devices and %ld CPU cores.\n",
            MAX_NGPU, MAX_NCPU);
        printf("Important: It takes k CPU cores to control k GPU devices.\n"
               "           For instance, selecting 4 GPUs requires 4 "
               "CPU cores in additional to the worker CPU cores.\n");
        printf("           Thus, it must be that\n");
        printf("               [# of CPUs] + [# of CPU workers] "
               "<= [# of CPU cores available] = %ld\n", MAX_NCPU);
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

    printf("# of GPUs = %ld\n", NGPU);
    printf("# of CPU workers = %ld\n", NCPUW);
    D = read_mat(Din, D_dims);
    E = read_mat(Ein, E_dims);
    cfg = load_cfg("params.cfg");
    N = (D_dims[0] > D_dims[1]) ? D_dims[0] : D_dims[1];

    Q_dims[0] = Q_dims[1] = N;
    Q = (double *)malloc(N * N * sizeof(double));

    WORK = allocate_work(N);
    WORK_dev = allocate_work_dev(NGPU, N);
    IWORK = allocate_iwork(N);

    get_time(&timer1);
    dlaed0_m(NGPU, NCPUW, N, D, E, Q, N, WORK, WORK_dev, IWORK, cfg);
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
