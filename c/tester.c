#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "dstedc.h"

extern double *read_mat(const char *filename, const char *varname,
    size_t *dims);
extern void write_mat(const char *filename, const char *varname,
    double *array, size_t *dims);

int main(int argc, char **argv)
{
    const char *fin  = argv[1];
    const char *fout1 = argv[2];
    const char *fout2 = argv[3];
    
    size_t D_dims[2], E_dims[2], Q_dims[2];
    double *D = read_mat(fin, "D", D_dims);
    double *E = read_mat(fin, "E", E_dims);
    double *Q;
    int N = (D_dims[0] > D_dims[1]) ? (int)D_dims[0] : (int)D_dims[1];
    double *WORK;
    int *IWORK;

    Q_dims[0] = Q_dims[1] = (size_t)N;
    Q = (double *)malloc(N * N * sizeof(double));
    WORK = (double *)malloc((2 * N + 2 * N * N) * sizeof(double));
    IWORK = (int *)malloc((3 + 5 * N) * sizeof(int));

    dlaed0(N, D, E, Q, N, WORK, IWORK);

    write_mat(fout1, "D", D, D_dims);
    write_mat(fout2, "Q", Q, Q_dims);

    return 0;
}
