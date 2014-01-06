#include <stdio.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <omp.h>
#include "dstedc.h"

#define SMLSIZ 25

void dlaed0_m(int NGPU, int N, double *D, double *E, double *Q, int LDQ,
    double **WORK, double **WORK_dev, int **IWORK)
/* computes all eigenvalues and corresponding eigenvectors of a symmetric
   tridiagonal matrix using the divide-and-conquer eigenvalue algorithm.
   We will have
      diag(D(in)) + diag(E, 1) + diag(E, -1) = Q * diag(D(out)) + Q'. */
{
    int subpbs; // number of submatrices
    int i, j, k, submat, smm1, msd2, matsiz;
    int *partition = &IWORK[0][0];
    int *perm1 = &IWORK[0][4*N + 3];

    // Determine the size and placement of the submatrices, and save in
    // the leading elements of IWORK.
    partition[0] = N;
    subpbs = 1;

    while (partition[subpbs-1] > SMLSIZ) {
        for (j = subpbs-1; j >= 0; j--) {
            partition[2*j + 1] = (partition[j] + 1) / 2;
            partition[2*j] = partition[j] / 2;
        }
        subpbs *= 2;
    }
    for (j = 1; j < subpbs; j++)
        partition[j] += partition[j-1];

    // Divide the matrix into subpbs submatricies of size at most
    // SMLSIZ+1 using rank-1 modifications (cuts).
    for (i = 0; i < subpbs - 1; i++) {
        submat = partition[i];
        smm1 = submat - 1;
        D[smm1] -= fabs(E[smm1]);
        D[submat] -= fabs(E[smm1]);
    }
    
    // Solve each submatrix eigenvalue problem at the bottom of the divide and
    // conquer tree.
    for (i = -1; i < subpbs - 1; i++) {
        if (i == -1) {
            submat = 0;
            matsiz = partition[0];
        } else {
            submat = partition[i];
            matsiz = partition[i+1] - partition[i];
        }
        LAPACKE_dsteqr_work(LAPACK_COL_MAJOR, 'I', matsiz, &D[submat],
            &E[submat], &Q[submat + submat * LDQ], LDQ, WORK[0]);
        k = 0;
        for (j = submat; j < partition[i+1]; j++)
            perm1[j] = k++;
    }

    // Successively merge eigensystems of adjacent submatrices into
    // eigensystem for the corresponding larger matrix.
    omp_set_num_threads(NGPU);
    while (subpbs > 1) {
        #pragma omp parallel for default(none) \
            private(i, j, k, submat, matsiz, msd2) firstprivate(subpbs, LDQ) \
            shared(partition, D, Q, perm1, E, WORK, IWORK, WORK_dev)
        for (j = 0; j < subpbs/2; j++) {
            i = 2*j - 1;
            if (i == -1) {
                submat = 0;
                matsiz = partition[1];
                msd2 = partition[0];
            } else {
                submat = partition[i];
                matsiz = partition[i+2] - partition[i];
                msd2 = matsiz / 2;
            }
            // Merge lower order eigensystems (of size msd2 and matsiz - msd2)
            // into an eigensystem of size matsiz.
            k = omp_get_thread_num();
            cudaSetDevice(k);
            dlaed1(matsiz, &D[submat], &Q[submat + submat * LDQ], LDQ,
                &perm1[submat], E[submat+msd2-1], msd2, WORK[k], WORK_dev[k],
                &IWORK[k][subpbs]);
        }
        for (i = -1; i < subpbs - 2; i += 2)
            partition[(i-1)/2 + 1] = partition[i+2];
        subpbs /= 2;
    }
    
    // Re-merge the eigenvalues and eigenvectors which were deflated at the
    // final merge step.
    // D = D(perm1);
    for (i = 0; i < N; i++)
        WORK[0][i] = D[perm1[i]];
    cblas_dcopy(N, WORK[0], 1, D, 1);

    // Q = Q(perm1);
    for (j = 0; j < N; j++) {
        i = perm1[j];
        cblas_dcopy(N, &Q[i * LDQ], 1, &WORK[0][(j + 1) * N], 1);
    }
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', N, N, &WORK[0][N], N, Q, LDQ); 
}
