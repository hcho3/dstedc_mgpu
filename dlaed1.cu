#include <cblas.h>
#include <lapacke.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "dstedc.h"
#include "nvtx.h"
#include "safety.h"

void dlaed1(long N, double *D, double *Q, long LDQ, long *perm1, double RHO,
    long CUTPNT, double *WORK, double *WORK_dev, long *IWORK)
/*
computes the updated eigensystem of a diagonal matrix after modification by a
rank-one symmetric matrix.

  T = Q(in) ( D(in) + rho * z * z') Q'(in) = Q(out) * D(out) * Q'(out)
    where z = Q'(in) * u, u is a vector of length N with ones in the cutpnt
    and cutpnt + 1 th elements and zeros elsewhere. 
*/
{
    RANGE_START("dlaed1", 1, 1);

    long i;
    long K;
    double *Z      = &WORK[0];
    double *DWORK  = &WORK[N];
    double *QWORK  = &WORK[2 * N];

    double *QHAT_dev   = &WORK_dev[N * N];
    double *Q_dev  = &WORK_dev[2 * N * N];
    cublasHandle_t cb_handle;
    double dgemm_param[2] = {1.0, 0.0};

    long *perm2     = &IWORK[0];
    long *permacc   = &IWORK[N];
    long *perm3     = &IWORK[2 * N];

    safe_cublasCreate(&cb_handle);
    safe_cublasSetPointerMode(cb_handle, CUBLAS_POINTER_MODE_HOST);

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
        dlaed3(K, D, QHAT_dev, K, RHO, DWORK, Z, WORK_dev);

        // back-transformation
        safe_cublasSetMatrix(N, K, sizeof(double), Q, LDQ, WORK_dev, N);
        safe_cublasDgemm(cb_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, K, K,
            &dgemm_param[0], WORK_dev, N, QHAT_dev, K, &dgemm_param[1],
            Q_dev, LDQ);
        safe_cublasGetMatrix(N, K, sizeof(double), Q_dev, LDQ, Q, LDQ);

        // compute perm1 that would merge back deflated values.
        dlamrg(K, N-K, D, 1, -1, perm1);
    } else {
        for (i = 0; i < N; i++)
            perm1[i] = i;
    }

	safe_cublasDestroy(cb_handle);

    RANGE_END(1);
}
