#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>
#include "dstedc.h"

void dlaed1_cpu(long NCORE, long N, double *D, double *Q, long LDQ,
    long *perm1, double RHO, long CUTPNT, double *WORK, long *IWORK)
/*
computes the updated eigensystem of a diagonal matrix after modification by a
rank-one symmetric matrix.

  T = Q(in) ( D(in) + rho * z * z') Q'(in) = Q(out) * D(out) * Q'(out)
    where z = Q'(in) * u, u is a vector of length N with ones in the cutpnt
    and cutpnt + 1 th elements and zeros elsewhere. 
*/
{
    long i, j, k;
    long blki, blkj, blkk;
    long pardim = NCORE;

    long K; // number of non-deflated eigenvalues

    double *Z      = &WORK[0];
    double *DWORK  = &WORK[N];
    double *QWORK  = &WORK[2 * N];
    double *QHAT   = &WORK[2 * N + N * N];

    long *perm2    = &IWORK[0];
    long *permacc  = &IWORK[N];
    long *perm3    = &IWORK[2 * N];
    long *parN, *parK;

    WORK[0] = WORK[0];
    
    // set up temporary arrays
    parN = (long *)malloc((pardim+1) * sizeof(long));
    parK = (long *)malloc((pardim+1) * sizeof(long));

    // form the z vector
    cblas_dcopy(CUTPNT, &Q[CUTPNT-1], LDQ, &Z[0], 1);
    cblas_dcopy(N-CUTPNT, &Q[CUTPNT + CUTPNT * LDQ], LDQ,
        &Z[CUTPNT], 1);

    // deflate eigenvalues
    for (i = CUTPNT; i < N; i++)
        perm1[i] += CUTPNT;

    dlaed2(&K, N, CUTPNT, D, Q, LDQ, perm1, &RHO, Z, DWORK, QWORK, perm2,
        permacc, perm3);

    // sovle secular equation
    if (K > 0) {
        cblas_dcopy(K, D, 1, DWORK, 1);
        dlaed3_cpu(NCORE, K, D, QHAT, K, RHO, DWORK, Z, QWORK);

        // back-transformation

        // compute block matrix partition
        parN[0] = 0;
        parN[pardim] = N;
        for (i = 1; i < pardim; i++)
            parN[i] = parN[i-1] + N/pardim;
        parK[0] = 0;
        parK[pardim] = K;
        for (i = 1; i < pardim; i++)
            parK[i] = parK[i-1] + K/pardim;

        // out-of-core matrix multiplication
        omp_set_num_threads(NCORE);
        for (j = 0; j < pardim; j++) {
            #pragma omp parallel for default(none) \
                private(i, k, blki, blkj, blkk) \
                firstprivate(j, LDQ, K, N, pardim) \
                shared(parN, parK, Q, QHAT, QWORK)
            for (i = 0; i < pardim; i++) {
                blki = parN[i+1] - parN[i];
                blkk = parK[1] - parK[0];
                blkj = parK[j+1] - parK[j];
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    blki, blkj, blkk, 1.0, &Q[parN[i]], LDQ,
                    &QHAT[parK[j]*K], K, 0.0, &QWORK[parN[i] + parK[j]*N], N);
                for (k = 1; k < pardim; k++) {
                    blkk = parK[k+1] - parK[k];
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        blki, blkj, blkk, 1.0, &Q[parN[i] + parK[k]*LDQ], LDQ,
                        &QHAT[parK[k] + parK[j]*K], K, 1.0,
                        &QWORK[parN[i] + parK[j]*N], N);
                }
            }
        }
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', N, K, QWORK, N, Q, LDQ);

        // compute perm1 that would merge back deflated values.
        dlamrg(K, N-K, D, 1, -1, perm1);
    } else {
        for (i = 0; i < N; i++)
            perm1[i] = i;
    }

    free(parK);
    free(parN);
}
