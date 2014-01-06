#include <cblas.h>
#include <lapacke.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "dstedc.h"

void dlaed1(int N, double *D, double *Q, int LDQ, int *perm1, double RHO,
    int CUTPNT, double *WORK, double *WORK_dev, int *IWORK)
/*
computes the updated eigensystem of a diagonal matrix after modification by a
rank-one symmetric matrix.

  T = Q(in) ( D(in) + rho * z * z') Q'(in) = Q(out) * D(out) * Q'(out)
    where z = Q'(in) * u, u is a vector of length N with ones in the cutpnt
    and cutpnt + 1 th elements and zeros elsewhere. 
*/
{
    int i;
    int K;
    double *Z      = &WORK[0];
    double *DWORK  = &WORK[N];
    double *QWORK  = &WORK[2 * N];

    double *QHAT_dev   = &WORK_dev[2 * N + N * N];
    double *Q_dev  = &WORK_dev[2 * N + 2 * N * N];
    cublasHandle_t cb_handle;
    double dgemm_param[2] = {1.0, 0.0};

    int *perm2     = &IWORK[0];
    int *permacc   = &IWORK[N];
    int *perm3     = &IWORK[2 * N];

    cublasCreate(&cb_handle);
    cublasSetPointerMode(cb_handle, CUBLAS_POINTER_MODE_HOST);

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
        dlaed3(K, D, QHAT_dev, K, RHO, DWORK, Z, QWORK, WORK_dev);

        // back-transformation
        cublasSetMatrix(N, K, sizeof(double), Q, LDQ, WORK_dev, N);
        cublasDgemm(cb_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, K, K,
            &dgemm_param[0], WORK_dev, N, QHAT_dev, K, &dgemm_param[1],
            Q_dev, LDQ);
        cublasGetMatrix(N, K, sizeof(double), Q_dev, LDQ, Q, LDQ);

        // compute perm1 that would merge back deflated values.
        dlamrg(K, N-K, D, 1, -1, perm1);
    } else {
        for (i = 0; i < N; i++)
            perm1[i] = i;
    }

	cublasDestroy(cb_handle);
}
