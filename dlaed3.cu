#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include "dstedc.h"
#include "nvtx.h"
#include "safety.h"

__global__ static void inv_eig(long K, double *v, double *DLAMDA, double *W,
    double *tau, double *orig);
__global__ static void eigvec(long K, double *D, double *QHAT, long LDQHAT,
    double *v, double *DLAMDA, double *tau, double *orig);

void dlaed3(long K, double *D, double *QHAT_dev, long LDQHAT, double RHO,
    double *DLAMDA, double *W, double *WORK_dev)
// stably computes the eigendecomposition Q * diag(lambda) * Q**T  of
// diag(delta) + RHO * z * z**T  by solving an inverse eigenvalue problem.
{
    RANGE_START("dlaed3", 1, 3);

    double *W_dev      = &WORK_dev[0];
    double *DLAMDA_dev = &WORK_dev[K];
    double *tau_dev    = &WORK_dev[2 * K];
    double *orig_dev   = &WORK_dev[3 * K];
    double *v_dev      = &WORK_dev[4 * K];
    double *D_dev      = &WORK_dev[5 * K];

    safe_cudaMemcpy(DLAMDA_dev, DLAMDA, K * sizeof(double),
        cudaMemcpyHostToDevice);
    safe_cudaMemcpy(W_dev, W, K * sizeof(double), cudaMemcpyHostToDevice);

    dlaed4<<<(K+TPB-1)/TPB, TPB>>>(K, DLAMDA_dev, W_dev, RHO,
        tau_dev, orig_dev);
    safe_cudaDeviceSynchronize();

    // inverse eigenvalue problem: find v such that lambda(1), ..., lambda(n)
    // are exact eigenvalues of the matrix D + v * v**T.
    inv_eig<<<(K+TPB-1)/TPB, TPB>>>(K, v_dev, DLAMDA_dev, W_dev,
        tau_dev, orig_dev);

    // compute the eigenvectors of D + v * v**T
    eigvec<<<(K+TPB-1)/TPB, TPB>>>(K, D_dev, QHAT_dev, LDQHAT, v_dev,
        DLAMDA_dev, tau_dev, orig_dev);

    safe_cudaMemcpy(D, D_dev, K * sizeof(double), cudaMemcpyDeviceToHost); 

    RANGE_END(1);
}

__global__ static void inv_eig(long K, double *v, double *DLAMDA, double *W,
    double *tau, double *orig)
{
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long j;

    if (i >= K)
        return;

    v[i] = orig[i] - DLAMDA[i] + tau[i];
    for (j = 0; j < i; j++)
        v[i] *= ((DLAMDA[i] - orig[j] - tau[j]) / (DLAMDA[i] - DLAMDA[j]));
    for (j = i+1; j < K; j++)
        v[i] *= ((orig[j] - DLAMDA[i] + tau[j]) / (DLAMDA[j] - DLAMDA[i]));
    v[i] = copysign(sqrt(v[i]), W[i]);
}

__global__ static void eigvec(long K, double *D, double *QHAT, long LDQHAT,
    double *v, double *DLAMDA, double *tau, double *orig)
{
    long j = threadIdx.x + blockIdx.x * blockDim.x;
    long i;
    double temp = 0.0;

    if (j >= K)
        return;

    D[j] = tau[j] + orig[j];

    for (i = 0; i < K; i++) {
        QHAT[i + j * LDQHAT] = v[i] / (orig[j] - DLAMDA[i] + tau[j]);
        temp += SQ(QHAT[i + j * LDQHAT]);
    }
    temp = sqrt(temp);
    for (i = 0; i < K; i++)
        QHAT[i + j * LDQHAT] /= temp;
}
