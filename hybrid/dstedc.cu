#include <cblas.h>
#include <lapacke.h>
#include <float.h>
#include <math.h>
#include "dstedc.h"

#define SMLSIZ 25

void dstedc(int N, double *D, double *E, double *Z, int LDZ,
    double *WORK, double *WORK_dev, int LWORK, int *IWORK, int LIWORK)
{
    int start, finish, M;
    double tiny, orgnrm;
    int i, ii, j, k;
    double tmp, p;

    if (N <= SMLSIZ) {
        LAPACKE_dsteqr_work(LAPACK_COL_MAJOR,'I', N, D, E, Z, LDZ, WORK);
        return;
    }
    
    start = 0;
    while (start < N) {
    /* Let finish be the position of the next subdiagonal entry such that
       E(finish) <= tiny or finish = N if no such subdiagonal exists. The
       matrix identified by the elements between start and finish
       constitutes an independent sub-problem. */
        finish = start;
        while (finish < N) {
            tiny = DBL_EPSILON * sqrt( fabs(D[finish]) ) *
                                 sqrt( fabs(D[finish+1]) );
            if (fabs(E[finish]) > tiny)
                finish++;
            else
                break;
        }

        // (Sub) Problem determined. Compute its size and solve it.
        M = finish - start + 1;
        if (M == 1) {
            start = finish + 1;
            continue;
        }
        if (M > SMLSIZ) {
            // Scale.
            orgnrm = 0.0;
            for (i = start; i <= finish; i++) {
                tmp = fabs(D[i]);
                if (tmp > orgnrm) 
                    orgnrm = tmp;
            }
            for (i = start; i <= finish-1; i++) {
                tmp = fabs(E[i]);
                if (tmp > orgnrm) 
                    orgnrm = tmp;
            }
            cblas_dscal(M,   1.0 / orgnrm, &D[start], 1);
            cblas_dscal(M-1, 1.0 / orgnrm, &E[start], 1);

            // Solve the sub-problem.
            dlaed0(M, &D[start], &E[start], &Z[start * LDZ + start],
                LDZ, WORK, WORK_dev, IWORK);

            // Scale back.
            cblas_dscal(M, orgnrm, &D[start], 1);
        } else {
            LAPACKE_dsteqr_work(LAPACK_COL_MAJOR, 'I', M, &D[start],
                &E[start], &Z[start * LDZ + start], LDZ, WORK);
        }
        start = finish + 1;
    }
    /* If the problem split any number of times, then the eigenvalues
    will not be properly ordered. Here we permute the eigenvalues (and
    the associated eigenvectors) into ascending order. */
    if (M != N) {
        for (ii = 1; ii < N; ii++) {
            i = ii - 1;
            k = i;
            p = D[i];
            for (j = ii; j < N; j++) {
                if (D[j] < p) {
                    k = j;
                    p = D[j];
                }
            }
            if (k != i) {
                D[k] = D[i];
                D[i] = p;
                cblas_dswap(N, &Z[i * LDZ], 1, &Z[k * LDZ], 1);
            }
        }
    }
}
