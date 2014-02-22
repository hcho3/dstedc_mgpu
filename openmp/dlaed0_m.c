#include <stdio.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <omp.h>
#include "dstedc.h"
#include "timer.h"

#define SMLSIZ 128

void dlaed0_m(long NGRP, long NCORE, long N, double *D, double *E, double *Q,
    long LDQ, double *WORK, long *IWORK)
/* computes all eigenvalues and corresponding eigenvectors of a symmetric
   tridiagonal matrix using the divide-and-conquer eigenvalue algorithm.
   We will have
      diag(D(in)) + diag(E, 1) + diag(E, -1) = Q * diag(D(out)) + Q'. */
{
    long subpbs; // number of subproblems
    long i, j, k, submat, smm1, msd2, matsiz;
    long *partition = &IWORK[0];
    long *perm1 = &IWORK[4*N];

    long pbmax;
    long NCOREP; // # cores allocated to each compute group

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
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for default(none) \
        private(i, j, k, submat, matsiz) firstprivate(subpbs, LDQ) \
        shared(partition, D, E, Q, WORK, perm1)
    for (i = -1; i < subpbs - 1; i++) {
        if (i == -1) {
            submat = 0;
            matsiz = partition[0];
        } else {
            submat = partition[i];
            matsiz = partition[i+1] - partition[i];
        }
        LAPACKE_dsteqr_work(LAPACK_COL_MAJOR, 'I', matsiz, &D[submat],
            &E[submat], &Q[submat + submat * LDQ], LDQ, &WORK[2*submat]);
        k = 0;
        for (j = submat; j < partition[i+1]; j++)
            perm1[j] = k++;
    }

    // Successively merge eigensystems of adjacent submatrices into
    // eigensystem for the corresponding larger matrix.
    pbmax = 0;
    
    struct timeval timer1, timer2;

    omp_set_num_threads(NGRP);
    while (subpbs > 1) {
        // update pbmax.
        for (j = 0; j < subpbs/2; j++) {
            i = 2*j - 1;
            matsiz = (i == -1) ? partition[1] : partition[i+2]-partition[i];
            if (matsiz > pbmax)
                pbmax = matsiz;
        }

        get_time(&timer1);
        #pragma omp parallel for default(none) \
            private(i, j, submat, matsiz, msd2, NCOREP) \
            firstprivate(N, subpbs, LDQ, NGRP, NCORE) \
            shared(partition, D, Q, perm1, E, WORK, IWORK)
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
            NCOREP = NCORE / ((subpbs/2 >= NGRP) ? NGRP : (subpbs/2));
            dlaed1(NCOREP, matsiz, &D[submat], &Q[submat + submat * LDQ], LDQ,
                &perm1[submat], E[submat+msd2-1], msd2,
                &WORK[submat*(2*N+2*N*N)/N], &IWORK[subpbs+3*submat]);
        }
        get_time(&timer2);
        printf("cost per subproblem = %.3lf s, pbmax = %ld\n", 
            get_elapsed_ms(timer1, timer2) / 1000.0 / subpbs, pbmax);

        // update partition.
        for (i = -1; i < subpbs - 2; i += 2)
            partition[(i-1)/2 + 1] = partition[i+2];
        subpbs /= 2;
    }
    
    // Re-merge the eigenvalues and eigenvectors which were deflated at the
    // final merge step.
    // D = D(perm1);
    for (i = 0; i < N; i++)
        WORK[i] = D[perm1[i]];
    cblas_dcopy(N, WORK, 1, D, 1);

    // Q = Q(perm1);
    for (j = 0; j < N; j++) {
        i = perm1[j];
        cblas_dcopy(N, &Q[i * LDQ], 1, &WORK[(j + 1) * N], 1);
    }
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', N, N, &WORK[N], N, Q, LDQ); 
}
