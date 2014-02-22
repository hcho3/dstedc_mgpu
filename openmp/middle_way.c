#include <stdio.h>
#include <math.h>
#include "dstedc.h"

double middle_way(long k, long n, double *delta, double *zeta, double rho,
    double tau, double orig)
{
    double DELTA_k, DELTA_k_1;
    double DELTA_n_2, DELTA_n_1;
    double delta_n;    // delta[n]
    double zzr  = 0.0; // zeta' * zeta / rho;
    double f_y, fderiv_y, psideriv_y, phideriv_y;
    double a, b, c, eta;
    long i;

    for (i = 0; i < n; i++)
        zzr += zeta[i] * zeta[i];
    zzr /= rho;

    if (k >= 0 && k < n-1) {
        DELTA_k = delta[k] - orig - tau;
        DELTA_k_1 = delta[k+1] - orig - tau;
        f_y = rho;
        fderiv_y = 0.0;
        psideriv_y = 0.0;
        phideriv_y = 0.0;
        for (i = 0; i < n; i++)
            f_y += SQ(zeta[i]) / (delta[i] - orig - tau);
        for (i = 0; i < n; i++)
            fderiv_y += SQ(zeta[i]) / SQ(delta[i] - orig - tau);
        for (i = 0; i <= k; i++)
            psideriv_y += SQ(zeta[i]) / SQ(delta[i] - orig - tau);
        for (i = k+1; i < n; i++)
            phideriv_y += SQ(zeta[i]) / SQ(delta[i] - orig - tau);

        a = (DELTA_k + DELTA_k_1) * f_y - DELTA_k * DELTA_k_1 * fderiv_y;
        b = DELTA_k * DELTA_k_1 * f_y;
        c = f_y - DELTA_k * psideriv_y - DELTA_k_1 * phideriv_y;
        if (a <= 0.0)
            eta = (a - sqrt(SQ(a) - 4 * b * c)) / (2 * c);
        else
            eta = (2 * b) / (a + sqrt(SQ(a) - 4 * b * c));
        // if eta + y falls below delta[k] or exceeds delta[k+1],
        // do a Newton step.
        if (DELTA_k - eta >= 0.0 || DELTA_k_1 - eta <= 0.0)
            eta = -f_y / fderiv_y;
    } else {
        delta_n   = delta[n-1] + zzr; // delta[n]
        DELTA_n_2 = delta[n-2] - orig - tau; // DELTA_(n-2)
        DELTA_n_1 = delta[n-1] - orig - tau; // DELTA_(n-1)
        f_y = rho;
        fderiv_y = 0.0;
        psideriv_y = 0.0;
        for (i = 0; i < n; i++)
            f_y += SQ(zeta[i]) / (delta[i] - orig - tau);
        for (i = 0; i < n; i++)
            fderiv_y += SQ(zeta[i]) / SQ(delta[i] - orig - tau);
        for (i = 0; i < n-1; i++)
            psideriv_y += SQ(zeta[i]) / SQ(delta[i] - orig - tau);

        a = (DELTA_n_2 + DELTA_n_1) * f_y - DELTA_n_2 * DELTA_n_1 * fderiv_y;
        b = DELTA_n_2 * DELTA_n_1 * f_y;
        c = f_y - DELTA_n_2 * psideriv_y - SQ(zeta[n-1]) / DELTA_n_1;

        if (a >= 0.0)
            eta = (a + sqrt(SQ(a) - 4 * b * c)) / (2 * c);
        else
            eta = (2 * b) / (a - sqrt(SQ(a) - 4 * b * c));
        // if eta + y falls below delta[n], simply do a Newton step.
        if (DELTA_n_1 - eta >= 0.0 || (delta_n - orig - tau) - eta <= 0.0)
            eta = -f_y / fderiv_y;
    }

    return eta;
}
