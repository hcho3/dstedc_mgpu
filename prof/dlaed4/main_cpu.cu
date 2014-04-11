#include <stdio.h>
#include <omp.h>
#include "safety.h"
#include "timer.h"

double *read_mat(const char *filename, long *dims);
void write_mat(const char *filename, double *array, long *dims);
void dlaed4_cpu(long K, long I, double *D, double *Z, double RHO, double *tau,
    double *orig);
long max_num_block(void);

int main(int argc, char **argv)
{
    const char *fRHO    = argv[2];
    const char *fW      = argv[3];
    const char *fDLAMDA = argv[4];

    double *W, *DLAMDA;
    double *tau, *orig, *rho;
    double RHO;
    double tmp;
    long dims[2];
    long K;
    long i;
    long NCPUW, maxNCPUW;
    timeval timer1, timer2;

    maxNCPUW = omp_get_num_procs();

    if (argc < 5 ||
        sscanf(argv[1], "%ld", &NCPUW) < 1 || NCPUW <= 0 || NCPUW > maxNCPUW) {
        fprintf(stderr, "Usage: %s [# of CPU cores] [RHO.bin] "
                        "[W.bin] [DLAMDA.bin]\n", argv[0]);
        exit(1);
    }

    rho = read_mat(fRHO, dims);
    RHO = rho[0];
    W = read_mat(fW, dims);
    DLAMDA = read_mat(fDLAMDA, dims);
    K = dims[0];
    tau = (double *)malloc(K * sizeof(double));
    orig = (double *)malloc(K * sizeof(double));

    omp_set_num_threads(NCPUW);

    get_time(&timer1);
    #pragma omp parallel for default(none) \
        private(i) firstprivate(K, RHO) shared(DLAMDA, W, tau, orig)
    for (i = 0; i < K; i++)
        dlaed4_cpu(K, i, DLAMDA, W, RHO, &tau[i], &orig[i]);
    get_time(&timer2);
    tmp = get_elapsed_ms(timer1, timer2) / 1000.0;
    printf("CPU:%ld:%.20lf\n", NCPUW, tmp);
    fprintf(stderr, "# eigenvalues = %ld, %ld CPU cores: %.3lf s\n", K,
        NCPUW, tmp);

    free(W);
    free(DLAMDA);
    free(rho);
    free(tau);
    free(orig);

    return 0;
}
