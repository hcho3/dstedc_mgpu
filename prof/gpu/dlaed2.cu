#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include "dstedc.h"
#include "nvtx.h"

void dlaed2(long *K, long N, long N1, double *D, double *Q, long LDQ,
    long *perm1, double *RHO, double *Z, double *DWORK, double *QWORK,
    long *perm2, long *permacc, long *perm3) 
// merges two lists of eigenvalues and carries out deflation.
{
    RANGE_START("dlaed2", 1, 2);

    long N2 = N - N1;
    long imax, jmax;
    long i, j, k, ti, pi, ni;
    long K2;
    double tol;
    double t, tau, s, c;

    if (*RHO < 0)
        // make rho positive.
        cblas_dscal(N2, -1.0, &Z[N1], 1); 

    // normalize z so that norm2(z) = 1. Since z is the concatenation of two
    // normalized vectors, norm2(z) = sqrt(2).
    cblas_dscal(N, 1.0/sqrt(2.0), Z, 1);
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
    imax = cblas_idamax(N, Z, 1);
    jmax = cblas_idamax(N, D, 1);
    tol = 8.0 * DBL_EPSILON * fmax( fabs(D[jmax]), fabs(Z[imax]) );
 
    // If the rank-1 modifier is small enough, we're done: all eigenvalues
    // deflate.
    if (*RHO * fabs(Z[imax]) <= tol) {
        *K = 0;
        // D = D(permacc)
        for (i = 0; i < N; i++)
            DWORK[i] = D[permacc[i]];
        cblas_dcopy(N, DWORK, 1, D, 1);
        // Q = Q(:, permacc)
        for (j = 0; j < N; j++) {
            i = permacc[j];
            cblas_dcopy(N, &Q[i * LDQ], 1, &QWORK[j * N], 1);
        }
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', N, N, QWORK, N, Q, LDQ);

        RANGE_END(1);

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
                cblas_drot(N, &Q[pi * LDQ], 1, &Q[ni * LDQ], 1, c, s);
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
    memcpy(permacc, perm2, N * sizeof(long));
    // D = D(permacc)
    for (i = 0; i < N; i++)
        DWORK[i] = D[permacc[i]];
    cblas_dcopy(N, DWORK, 1, D, 1);
    // Q = Q(:, permacc)
    for (j = 0; j < N; j++) {
        i = permacc[j];
        cblas_dcopy(N, &Q[i * LDQ], 1, &QWORK[j * LDQ], 1);
    }
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', N, N, QWORK, LDQ, Q, LDQ);
    // z = z(permacc)
    for (i = 0; i < N; i++)
        DWORK[i] = Z[permacc[i]];
    cblas_dcopy(N, DWORK, 1, Z, 1);

    RANGE_END(1);
}
