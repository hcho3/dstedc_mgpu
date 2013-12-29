#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include "dstedc.h"

void dlaed3(int K, double *D, double *QHAT, int LDQHAT, double RHO,
    double *DLAMDA, double *DLAMDA_dev, double *W, double *W_dev, double *S,
    double *S_dev)
// stably computes the eigendecomposition Q * diag(lambda) * Q**T  of
// diag(delta) + RHO * z * z**T  by solving an inverse eigenvalue problem.
{
    double *tau  = &S[0];
    double *orig = &S[K];
    double *v    = &S[2 * K];

    double *tau_dev = &S_dev[0];
    double *orig_dev = &S_dev[K];

    int i, j;
    double temp;
    
    cudaMemcpy(DLAMDA_dev, DLAMDA, K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(W_dev, W, K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(tau_dev, tau, K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(orig_dev, orig, K * sizeof(double), cudaMemcpyHostToDevice);

    dlaed4<<<(K+TPB-1)/TPB, TPB>>>(K, DLAMDA_dev, W_dev, RHO,
        tau_dev, orig_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(tau, tau_dev, K * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(orig, orig_dev, K * sizeof(double), cudaMemcpyDeviceToHost);

    // inverse eigenvalue problem: find v such that lambda(1), ..., lambda(n)
    // are exact eigenvalues of the matrix D + v * v**T.
    for (i = 0; i < K; i++) {
        v[i] = orig[i] - DLAMDA[i] + tau[i];
        for (j = 0; j < i; j++)
            v[i] *= ((DLAMDA[i] - orig[j] - tau[j]) / (DLAMDA[i] - DLAMDA[j]));
        for (j = i+1; j < K; j++)
            v[i] *= ((orig[j] - DLAMDA[i] + tau[j]) / (DLAMDA[j] - DLAMDA[i]));
        v[i] = copysign(sqrt(v[i]), W[i]);
    }

    // compute the eigenvectors of D + v * v**T
    for (i = 0; i < K; i++)
        D[i] = tau[i] + orig[i];
    for (j = 0; j < K; j++) {
        for (i = 0; i < K; i++)
            QHAT[i + j * LDQHAT] = v[i] / (orig[j] - DLAMDA[i] + tau[j]);
        temp = cblas_dnrm2(K, &QHAT[j * LDQHAT], 1); 
        for (i = 0; i < K; i++)
            QHAT[i + j * LDQHAT] /= temp;
    }
}
