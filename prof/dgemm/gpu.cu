#include <stdio.h>
#include <cublas_v2.h>
#include "timer.h"

double *read_mat(const char *filename, long *dims);
void write_mat(const char *filename, double *array, long *dims);

int main(int argc, char **argv)
{
    const char *fA = argv[1];
    const char *fB = argv[2];
    const char *fC = argv[3];
    int quiet = 0;
    cublasHandle_t handle;

    double *A, *B, *C;
    double *dA, *dB, *dC;
    long A_dims[2], B_dims[2], C_dims[2];
    long M, N, K;
    double one = 1.0, zero = 0.0;
    double tmp;
    timeval timer1, timer2;

    if (argc < 4) {
        fprintf(stderr, "Usage: %s [A.bin] [B.bin] [C.bin]\n"
               "[A.bin]: file where matrix A is saved\n"
               "[B.bin]: file where matrix B is saved\n"
               "[C.bin]: file where the product A*B will go\n", argv[0]);
        exit(1);
    }
    if (argc == 5 && strcmp(argv[4], "quiet") == 0)
        quiet = 1;

    A = read_mat(fA, A_dims);
    B = read_mat(fB, B_dims);
    
    if (A_dims[1] != B_dims[0]) {
        fprintf(stderr, "A and B have incompatible dimensions.\n");
        free(A);
        free(B);
        exit(1);
    }

    M = A_dims[0];
    N = B_dims[1];
    K = B_dims[0];
    /* multiply M-by-K matrix by K-by-N matrix and get M-by-N matrix out. */

    C = (double *)malloc(M * N * sizeof(double));
    C_dims[0] = M;
    C_dims[1] = N;

    get_time(&timer1);
    cudaMalloc((void **)&dA, M * K * sizeof(double));
    cudaMalloc((void **)&dB, K * N * sizeof(double));
    cudaMalloc((void **)&dC, M * N * sizeof(double));

    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasSetMatrix(M, K, sizeof(double), A, K, dA, K);
    cublasSetMatrix(K, N, sizeof(double), B, N, dB, N);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, 
        dA, K, dB, N, &zero, dC, N);
    cublasGetMatrix(M, N, sizeof(double), dC, N, C, N);
    get_time(&timer2);
    tmp = get_elapsed_ms(timer1, timer2);
    printf("%ld:%.20lf\n", M, tmp / 1000.0);
    fprintf(stderr, "Multiplying %ld-by-%ld matrix by %ld-by-%ld matrix: "
        "%.3lf s\n", M, K, K, N, tmp / 1000.0);

    if (!quiet)
        write_mat(fC, C, C_dims);

    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
