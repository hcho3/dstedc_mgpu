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

void dlaed3_gpu(long K, double *D, double *QHAT_dev, long LDQHAT, double RHO,
    double *DLAMDA, double *W, double *WORK_dev)
// stably computes the eigendecomposition Q * diag(lambda) * Q**T  of
// diag(delta) + RHO * z * z**T  by solving an inverse eigenvalue problem.
{
    RANGE_START("dlaed3_gpu", 1, 3);

    double *W_dev      = &WORK_dev[0];
    double *DLAMDA_dev = &WORK_dev[K];
    double *tau_dev    = &WORK_dev[2 * K];
    double *orig_dev   = &WORK_dev[3 * K];
    double *v_dev      = &WORK_dev[4 * K];
    double *D_dev      = &WORK_dev[5 * K];
    int NBLK, maxNBLK;

    safe_cudaMemcpy(DLAMDA_dev, DLAMDA, K * sizeof(double),
        cudaMemcpyHostToDevice);
    safe_cudaMemcpy(W_dev, W, K * sizeof(double), cudaMemcpyHostToDevice);

    maxNBLK = max_num_block();
    NBLK = (K+TPB-1)/TPB;
    if (NBLK > maxNBLK)
        NBLK = maxNBLK;

    dlaed4_gpu<<<NBLK, TPB>>>(0, K, K, DLAMDA_dev, W_dev, RHO,
        tau_dev, orig_dev);
    //safe_cudaDeviceSynchronize();
    //safe_cudaGetLastError();

    // inverse eigenvalue problem: find v such that lambda(1), ..., lambda(n)
    // are exact eigenvalues of the matrix D + v * v**T.
    inv_eig<<<NBLK, TPB>>>(K, v_dev, DLAMDA_dev, W_dev,
        tau_dev, orig_dev);

    // compute the eigenvectors of D + v * v**T
    eigvec<<<NBLK, TPB>>>(K, D_dev, QHAT_dev, LDQHAT, v_dev,
        DLAMDA_dev, tau_dev, orig_dev);

    safe_cudaMemcpy(D, D_dev, K * sizeof(double), cudaMemcpyDeviceToHost); 

    RANGE_END(1);
}

__global__ static void inv_eig(long K, double *v, double *DLAMDA, double *W,
    double *tau, double *orig)
{
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long j;
    double temp;

    if (i >= K)
        return;

    while (i < K) {
        temp = orig[i] - DLAMDA[i] + tau[i];
        for (j = 0; j < i; j++)
            temp *= ((DLAMDA[i] - orig[j] - tau[j]) / (DLAMDA[i] - DLAMDA[j]));
        for (j = i+1; j < K; j++)
            temp *= ((orig[j] - DLAMDA[i] + tau[j]) / (DLAMDA[j] - DLAMDA[i]));
        temp = copysign(sqrt(temp), W[i]);
        v[i] = temp;

        i += blockDim.x * gridDim.x;
    }
}

__global__ static void eigvec(long K, double *D, double *QHAT, long LDQHAT,
    double *v, double *DLAMDA, double *tau, double *orig)
{
    long j = threadIdx.x + blockIdx.x * blockDim.x;
    long i;
    double temp = 0.0;

    if (j >= K)
        return;

    while (j < K) {
        D[j] = tau[j] + orig[j];

        for (i = 0; i < K; i++) {
            QHAT[i + j * LDQHAT] = v[i] / (orig[j] - DLAMDA[i] + tau[j]);
            temp += SQ(QHAT[i + j * LDQHAT]);
        }
        temp = sqrt(temp);
        for (i = 0; i < K; i++)
            QHAT[i + j * LDQHAT] /= temp;

        j += blockDim.x * gridDim.x;
    }
}
