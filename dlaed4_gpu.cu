#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include "dstedc.h"

#include "initial_guess_gpu.cu"
#include "middle_way_gpu.cu"
// CUDA toolchain does not support inlining of a function in different
// compilation unit.

__global__ void dlaed4_gpu(long IL, long IU, long K, double *D, double *Z,
    double RHO, double *tau, double *orig)
// compute the i-th eigenvalue of the perturbed matrix D + rho * z * z**T
// tau+orig gives the computed eigenvalue.
{
    long it;
    double eta, tol;
    long I = threadIdx.x + blockIdx.x * blockDim.x + IL;

    if (I >= IU)
        return;

    RHO = 1.0 / RHO;

    while (I < IU) {
        // make a good initial guess
        initial_guess_gpu(I, K, D, Z, RHO, &tau[I], &orig[I]);

        // iterations begin
        it = 1;
        if (I < K-1)
            tol = 16.0 * DBL_EPSILON * 
                fmin(fabs(D[I]-orig[I]-tau[I]), fabs(D[I+1]-orig[I]-tau[I]));
        else
            tol = 16.0 * DBL_EPSILON * fabs(D[K-1]-orig[I]-tau[I]);
        while (it < 100) {
            eta = middle_way_gpu(I, K, D, Z, RHO, tau[I], orig[I]);
            tau[I] += eta;
            if (fabs(eta) <= tol)
                break;
            it++;
        }

        I += blockDim.x * gridDim.x;
    }
}
