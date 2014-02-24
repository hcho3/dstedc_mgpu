#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include "dstedc.h"

#include "initial_guess_ph2.cu"
#include "middle_way_ph2.cu"

void dlaed4_ph2(long K, long I, double *D, double *Z, double RHO, double *tau,
    double *orig)
// compute the i-th eigenvalue of the perturbed matrix D + rho * z * z**T
// tau+orig gives the computed eigenvalue.
{
    long it;
    double eta, tol;

    RHO = 1.0 / RHO;
    
    // make a good initial guess
    initial_guess_ph2(I, K, D, Z, RHO, tau, orig);

    // iterations begin
    it = 1;
    if (I < K-1)
        tol = 16.0 * DBL_EPSILON * 
            fmin(fabs(D[I]-*orig-*tau), fabs(D[I+1]-*orig-*tau));
    else
        tol = 16.0 * DBL_EPSILON * fabs(D[K-1]-*orig-*tau);
    while (it < 100) {
        eta = middle_way_ph2(I, K, D, Z, RHO, *tau, *orig);
        *tau += eta;
        if (fabs(eta) <= tol)
            break;
        it++;
    }
}
