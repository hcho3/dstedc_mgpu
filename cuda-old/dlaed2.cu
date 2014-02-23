__device__ __forceinline__ void dlaed2(cublasHandle_t cb_handle,
    int *K, int N, int N1, double *D, double *Q, int LDQ,
    int *perm1, double *RHO, double *Z, double *DWORK, double *QWORK,
    int *perm2, int *permacc, int *perm3) 
// merges two lists of eigenvalues and carries out deflation.
{
    int N2 = N - N1;
    int imax, jmax;
    int i, j, k, ti, pi, ni;
    int K2;
    double tol;
    double t, tau, s, c;

    double *dscal_param = (double *)malloc(2 * sizeof(double));
    double *drot_param = dscal_param;
    int *maxidx = (int *)malloc(2 * sizeof(int));

    if (*RHO < 0) {
        // make rho positive.
        dscal_param[0] = -1.0;
        cublasDscal(cb_handle, N2, &dscal_param[0], &Z[N1], 1); 
    }

    // normalize z so that norm2(z) = 1. Since z is the concatenation of two
    // normalized vectors, norm2(z) = sqrt(2).
    dscal_param[0] = 1.0 / sqrt(2.0);
    cublasDscal(cb_handle, N, &dscal_param[0], Z, 1);
    *RHO = fabs(2.0 * *RHO);  // RHO = abs(norm(z)^2 * RHO)

    // apply perm1 to re-merge deflated eigenvalues.
    for (i = 0; i < N; i++)
        DWORK[i] = D[perm1[i]];
    // compute perm2 that merge-sorts D1, D2 into one sorted list.
    dlamrg(N1, N2, DWORK, 1, 1, perm2);
    // apply perm2.
    for (i = 0; i < N; i++)
        permacc[i] = perm1[perm2[i]];

    // compute allowable deflation tolerance.
    cublasIdamax(cb_handle, N, Z, 1, &maxidx[0]);
    cublasIdamax(cb_handle, N, D, 1, &maxidx[1]);
    imax = maxidx[0] - 1;
    jmax = maxidx[1] - 1;
    tol = 8.0 * DBL_EPSILON * fmax( fabs(D[jmax]), fabs(Z[imax]) );
 
    // If the rank-1 modifier is small enough, we're done: all eigenvalues
    // deflate.
    if (*RHO * fabs(Z[imax]) <= tol) {
        *K = 0;
        // D = D(permacc)
        for (i = 0; i < N; i++)
            DWORK[i] = D[permacc[i]];
        cublasDcopy(cb_handle, N, DWORK, 1, D, 1);
        // Q = Q(:, permacc)
        for (j = 0; j < N; j++) {
            i = permacc[j];
            cublasDcopy(cb_handle, N, &Q[i * LDQ], 1, &QWORK[j * N], 1);
        }
        dlacpy(N, N, QWORK, N, Q, LDQ);
        return;
    }

    // --deflation--
    *K = 0;
    K2 = N - 1;
    for (i = 0; i < N; i++) {
        pi = permacc[i];
        if (*RHO * fabs(Z[pi]) <= tol) {
            // 1) z-component is small
            // => move D(i) to the end of list.
            perm3[K2] = i;
            K2--;
        } else if (i < N-1) {
            ni = permacc[i+1];
            t = fabs(D[ni] - D[pi]);
            tau = dlapy2(Z[pi], Z[ni]);
            s = -Z[pi] / tau;
            c = Z[ni] / tau;
            if (fabs(t * c * s) <= tol) {
                // 2) D(i) and D(i+1) are close to each other compared to the
                //    z-weights given to them.
                // => zero out z(i) by applying a Givens rotation. After this
                // step, D(i) can be deflated away.
                Z[ni] = tau;
                Z[pi] = 0.0;
                drot_param[0] = c;
                drot_param[1] = s;
                cublasDrot(cb_handle, N, &Q[pi * LDQ], 1, &Q[ni * LDQ], 1,
                    &drot_param[0], &drot_param[1]);
                t     = c * c * D[pi] + s * s * D[ni];
                D[ni] = s * s * D[pi] + c * c * D[ni];
                D[pi] = t;
                perm3[K2] = i;
                k = 0;
                while (K2+k+1 < N &&
                    D[permacc[perm3[K2+k]]] < D[permacc[perm3[K2+k+1]]]) {
                    ti = perm3[K2+k];
                    perm3[K2+k] = perm3[K2+k+1];
                    perm3[K2+k+1] = ti;
                    k++;
                }
                K2--;
            } else {
                // 3) D(i) is not deflated.
                perm3[*K] = i;
                (*K)++;
            }
        } else {
            // 3) D(i) is not deflated.
            perm3[*K] = i;
            (*K)++;
        }
    }

    // Apply perm3 to eigenpairs.
    // permacc = permacc(perm3)
    for (i = 0; i < N; i++)
        perm2[i] = permacc[perm3[i]];
    memcpy(permacc, perm2, N * sizeof(int));
    // D = D(permacc)
    for (i = 0; i < N; i++)
        DWORK[i] = D[permacc[i]];
    cublasDcopy(cb_handle, N, DWORK, 1, D, 1);
    // Q = Q(:, permacc)
    for (j = 0; j < N; j++) {
        i = permacc[j];
        cublasDcopy(cb_handle, N, &Q[i * LDQ], 1, &QWORK[j * LDQ], 1);
    }
    dlacpy(N, N, QWORK, LDQ, Q, LDQ);
    // z = z(permacc)
    for (i = 0; i < N; i++)
        DWORK[i] = Z[permacc[i]];
    cublasDcopy(cb_handle, N, DWORK, 1, Z, 1);

    free(dscal_param);
    free(maxidx);
}
