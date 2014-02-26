#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "dstedc.h"

void dlaed3_cpu(long NCORE, long K, double *D, double *QHAT, long LDQHAT,
    double RHO, double *DLAMDA, double *W, double *S)
// stably computes the eigendecomposition Q * diag(lambda) * Q**T  of
// diag(delta) + RHO * z * z**T  by solving an inverse eigenvalue problem.
{
    double *tau  = &S[0];
    double *orig = &S[K];
    double *v    = &S[2 * K];

    long i, j;
    double temp;

    omp_set_num_threads(NCORE);
    
    #pragma omp parallel for default(none) \
        private(i) firstprivate(K, RHO) shared(DLAMDA, W, tau, orig)
    for (i = 0; i < K; i++)
        dlaed4_cpu(K, i, DLAMDA, W, RHO, &tau[i], &orig[i]);

    // inverse eigenvalue problem: find v such that lambda(1), ..., lambda(n)
    // are exact eigenvalues of the matrix D + v * v**T.
    #pragma omp parallel for default(none) \
        private(i, j) firstprivate(K) shared(DLAMDA, W, v, tau, orig)
    for (i = 0; i < K; i++) {
        v[i] = orig[i] - DLAMDA[i] + tau[i];
        for (j = 0; j < i; j++)
            v[i] *= ((DLAMDA[i] - orig[j] - tau[j]) / (DLAMDA[i] - DLAMDA[j]));
        for (j = i+1; j < K; j++)
            v[i] *= ((orig[j] - DLAMDA[i] + tau[j]) / (DLAMDA[j] - DLAMDA[i]));
        v[i] = copysign(sqrt(v[i]), W[i]);
    }

    // compute the eigenvectors of D + v * v**T
    #pragma omp parallel for default(none) \
        private(i, j, temp) firstprivate(K, LDQHAT) \
        shared(D, DLAMDA, QHAT, v, tau, orig)
    for (j = 0; j < K; j++) {
        D[j] = tau[j] + orig[j];
        for (i = 0; i < K; i++)
            QHAT[i + j * LDQHAT] = v[i] / (orig[j] - DLAMDA[i] + tau[j]);
        temp = cblas_dnrm2(K, &QHAT[j * LDQHAT], 1); 
        for (i = 0; i < K; i++)
            QHAT[i + j * LDQHAT] /= temp;
    }
}
