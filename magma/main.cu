#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/time.h>
#define HAVE_CUBLAS 1
#include <magma.h>
#include "../timer.h"

extern double *read_mat(const char *filename, long *dims);
extern void write_mat(const char *filename, double *array, long *dims);

extern "C"{
    magma_int_t magma_dlaex0_m(magma_int_t nrgpu, magma_int_t n, double* d,
        double* e, double* q, magma_int_t ldq,
        double* work, magma_int_t* iwork,
        char range, double vl, double vu,
        magma_int_t il, magma_int_t iu, magma_int_t* info);
}

int main(int argc, char **argv)
{
    int NGPU, MAX_NGPU;
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
    int *IWORK;
    magma_int_t info;

    cudaGetDeviceCount(&MAX_NGPU);

    if (argc < 6 || sscanf(argv[1], "%d", &NGPU) < 1 ||
        NGPU <= 0 || NGPU > MAX_NGPU) {
        printf("Usage: %s [# of GPUs] [Din.bin] [E.bin] [Dout.bin] [Q.bin]\n",
            argv[0]);
        printf("The number of GPUs must be between 1 and %d.\n", MAX_NGPU);
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

    printf("# of GPUs = %d\n", NGPU);
    D = read_mat(Din, D_dims);
    E = read_mat(Ein, E_dims);
    N = (D_dims[0] > D_dims[1]) ? D_dims[0] : D_dims[1];

    Q_dims[0] = Q_dims[1] = N;
    Q = (double *)malloc(N * N * sizeof(double));

    WORK = (double *)malloc((2 * N + 2 * N * N) * sizeof(double));
    IWORK = (int *)malloc((3 + 5 * N) * sizeof(int));

    if (!Q || !WORK || !IWORK) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    get_time(&timer1);
    magma_dlaex0_m(NGPU, N, D, E, Q, N, WORK, IWORK, 'A', -1, -1, -1, -1, &info);
    get_time(&timer2);
    printf("info = %d\n", info);
    printf("Time: %.3lf s\n", get_elapsed_ms(timer1, timer2) / 1000.0 );

    write_mat(Dout, D, D_dims);
    write_mat(Qout, Q, Q_dims);

    free(Q);
    free(WORK);
    free(IWORK);
    free(D);
    free(E);

    return 0;
}
