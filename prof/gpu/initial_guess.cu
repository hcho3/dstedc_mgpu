__device__ __forceinline__ void eval_midpoint(long k, long n, double *delta,
	double delta_n, double *zeta, double rho, double *fval, double *gval);

__device__ __forceinline__ void initial_guess(long k, long n, double *delta,
	double *zeta, double rho, double *tau, double *orig)
{
    double fval, gval;
    double DELTA;
    double delta_n;    // delta[n]
    double zzr  = 0.0; // zeta' * zeta / rho;
    long i, K;
    double a, b, c, h;

    for (i = 0; i < n; i++)
        zzr += zeta[i] * zeta[i];
    zzr /= rho;
    delta_n = delta[n-1] + zzr;

    if (k >= 0 && k < n - 1) {
        eval_midpoint(k, n, delta, delta_n, zeta, rho, &fval, &gval);
        DELTA = delta[k+1] - delta[k];
        if (fval >= 0.0) {
            K = k;
            c = gval;
            a = c * DELTA + SQ(zeta[k]) + SQ(zeta[k+1]);
            b = SQ(zeta[k]) * DELTA;
        } else {
            K = k+1;
            c = gval;
            a = -c * DELTA + SQ(zeta[k]) + SQ(zeta[k+1]);
            b = -SQ(zeta[k+1]) * DELTA;
        }
        if (a <= 0.0)
            *tau = (a - sqrt(SQ(a) - 4 * b * c)) / (2 * c);
        else
            *tau = (2 * b) / (a + sqrt(SQ(a) - 4 * b * c));
    } else {
        eval_midpoint(n, n, delta, delta_n, zeta, rho, &fval, &gval);
        h =   SQ(zeta[n-2]) / (delta[n-2]-delta_n)
            + SQ(zeta[n-1]) / (delta[n-1]-delta_n);
        if (fval <= 0.0) {
            if (gval <= -h) {
                *tau = zzr;
            } else {
                DELTA = delta[n-1] - delta[n-2];
                c = gval;
                a = -c * DELTA + (SQ(zeta[n-2]) + SQ(zeta[n-1]));
                b = -SQ(zeta[n-1]) * DELTA;
                if (a >= 0.0)
                    *tau = (a + sqrt(SQ(a) - 4 * b * c)) / (2 * c);
                else
                    *tau = (2 * b) / (a - sqrt(SQ(a) - 4 * b * c));
            }
        } else {
            DELTA = delta[n-1] - delta[n-2];
            c = gval;
            a = -c * DELTA + (SQ(zeta[n-2]) + SQ(zeta[n-1]));
            b = -SQ(zeta[n-1]) * DELTA;
            if (a >= 0.0)
                *tau = (a + sqrt(SQ(a) - 4 * b * c)) / (2 * c);
            else
                *tau = (2 * b) / (a - sqrt(SQ(a) - 4 * b * c));
        }
        K = n-1;
    }
    *orig = delta[K];
}

__device__ __forceinline__ void eval_midpoint(long k, long n, double *delta,
	double delta_n, double *zeta, double rho, double *fval, double *gval)
// evaluates the values of f and g at (delta[k] + delta[k+1]) / 2
{
    double mpt;
    long j;
    if (k < n-1) {
        *gval = rho;
        mpt = (delta[k] + delta[k+1]) / 2.0;
        for (j = 0; j < k; j++)
            *gval += SQ(zeta[j]) / (delta[j]-mpt);
        for (j = k+2; j < n; j++)
            *gval += SQ(zeta[j]) / (delta[j]-mpt);
        *fval = *gval + SQ(zeta[k])   / (delta[k]-mpt)
                      + SQ(zeta[k+1]) / (delta[k+1]-mpt);
    } else {
        *gval = rho;
        mpt = (delta[n-1] + delta_n) / 2.0;
        for (j = 0; j < n-2; j++)
            *gval += SQ(zeta[j]) / (delta[j]-mpt);
        *fval = *gval + SQ(zeta[n-2]) / (delta[n-2]-mpt)
                      + SQ(zeta[n-1]) / (delta[n-1]-mpt);
    }
}
