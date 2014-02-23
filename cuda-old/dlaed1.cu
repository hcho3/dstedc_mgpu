__device__ __forceinline__ void dlaed1(cublasHandle_t cb_handle, int N,
    double *D, double *Q, int LDQ, int *perm1, double RHO, int CUTPNT,
    double *WORK, int *IWORK)
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
    double *QHAT   = &WORK[2 * N];
    double *QWORK  = &WORK[2 * N + N * N];

    double *dgemm_param = (double *)malloc(2 * sizeof(double));

    int *perm2     = &IWORK[0];
    int *permacc   = &IWORK[N];
    int *perm3     = &IWORK[2 * N];

    // form the z vector
    cublasDcopy(cb_handle, CUTPNT, &Q[CUTPNT-1], LDQ, &Z[0], 1);
    cublasDcopy(cb_handle, N-CUTPNT, &Q[CUTPNT + CUTPNT * LDQ], LDQ,
        &Z[CUTPNT], 1);

    // deflate eigenvalues
    for (i = CUTPNT; i < N; i++)
        perm1[i] += CUTPNT;

    dlaed2(cb_handle, &K, N, CUTPNT, D, Q, LDQ, perm1, &RHO, Z, DWORK, QWORK,
        perm2, permacc, perm3);

    // sovle secular equation
    if (K > 0) {
        cublasDcopy(cb_handle, K, D, 1, DWORK, 1);
        dlaed3(cb_handle, K, D, QHAT, K, RHO, DWORK, Z, QWORK);

        // back-transformation
        dlacpy(N, K, Q, LDQ, QWORK, N);
        dgemm_param[0] = 1.0;
        dgemm_param[1] = 0.0;
        cublasDgemm(cb_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, K, K,
            &dgemm_param[0], QWORK, N, QHAT, K, &dgemm_param[1], Q, LDQ);

        // compute perm1 that would merge back deflated values.
        dlamrg(K, N-K, D, 1, -1, perm1);
    } else {
        for (i = 0; i < N; i++)
            perm1[i] = i;
    }

    free(dgemm_param);
}
