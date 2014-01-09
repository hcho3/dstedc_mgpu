#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <cublas_v2.h>
#include <omp.h>
#include "dstedc.h"

#ifdef DEBUG
#define check( x ) _check( (x), __LINE__ )
#else
#define check( x ) (x)
#endif

static void _check(cublasStatus_t cs, long line)
{
    const char *errstr;

    if (cs != CUBLAS_STATUS_SUCCESS) {
        switch(cs) {
            case CUBLAS_STATUS_SUCCESS:
                errstr = "CUBLAS_STATUS_SUCCESS"; break;
            case CUBLAS_STATUS_NOT_INITIALIZED:
                errstr = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                errstr = "CUBLAS_STATUS_ALLOC_FAILED"; break;
            case CUBLAS_STATUS_INVALID_VALUE:
                errstr = "CUBLAS_STATUS_INVALID_VALUE"; break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                errstr = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                errstr = "CUBLAS_STATUS_MAPPING_ERROR"; break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                errstr = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                errstr = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
            default:
                errstr = "unknown";
        }
        printf("CUBLAS error %s at %ld.\n", errstr, line);
        exit(1);
    }
}

void dlaed1_ph2(long NGPU, long N, double *D, double *Q, long LDQ, long *perm1,
    double RHO, long CUTPNT, double *WORK, double **WORK_dev, long *IWORK)
/*
computes the updated eigensystem of a diagonal matrix after modification by a
rank-one symmetric matrix.

  T = Q(in) ( D(in) + rho * z * z') Q'(in) = Q(out) * D(out) * Q'(out)
    where z = Q'(in) * u, u is a vector of length N with ones in the cutpnt
    and cutpnt + 1 th elements and zeros elsewhere. 
*/
{
    long i, j, k, id;
    long blki, blkj, blkk;

    // largest submatrices that fit into main memory and GPU memory
    long pbcap_gpu  = max_matsiz_gpu(NGPU);
    long pbcap_host = max_matsiz_host(NGPU); 
    long pardim = (long)ceil(((double)pbcap_host / pbcap_gpu) / NGPU) * NGPU;

    long K; // number of non-deflated eigenvalues

    double *Z      = &WORK[0];
    double *DWORK  = &WORK[N];
    double *QWORK  = &WORK[2 * N];
    double *QHAT   = &WORK[2 * N + N * N];

    double **QHAT_dev;
    double **Q_dev;
    long *parN, *parK;
    cublasHandle_t *cb_handle;
    double zero = 0.0;
    double one = 1.0;

    long *perm2     = &IWORK[0];
    long *permacc   = &IWORK[N];
    long *perm3     = &IWORK[2 * N];

    // set up workspace aliases and temporary arrays
    QHAT_dev = (double **)malloc(NGPU * sizeof(double *));
    Q_dev = (double **)malloc(NGPU * sizeof(double *));
    cb_handle = (cublasHandle_t *)malloc(NGPU * sizeof(cublasHandle_t));
    parN = (long *)malloc((pardim+1) * sizeof(long));
    parK = (long *)malloc((pardim+1) * sizeof(long));
    for (i = 0; i < NGPU; i++) {
        QHAT_dev[i] = &WORK_dev[i][pbcap_gpu * pbcap_gpu];
        Q_dev[i] = &WORK_dev[i][2 * pbcap_gpu * pbcap_gpu];
    }

    // form the z vector
    cblas_dcopy(CUTPNT, &Q[CUTPNT-1], LDQ, &Z[0], 1);
    cblas_dcopy(N-CUTPNT, &Q[CUTPNT + CUTPNT * LDQ], LDQ,
        &Z[CUTPNT], 1);

    // deflate eigenvalues
    for (i = CUTPNT; i < N; i++)
        perm1[i] += CUTPNT;

    dlaed2(&K, N, CUTPNT, D, Q, LDQ, perm1, &RHO, Z, DWORK, QWORK, perm2,
        permacc, perm3);

    // sovle secular equation
    if (K > 0) {
        cblas_dcopy(K, D, 1, DWORK, 1);
        dlaed3_ph2(K, D, QHAT, K, RHO, DWORK, Z, QWORK);

        // back-transformation
        for (i = 0; i < NGPU; i++) {
            cudaSetDevice(i);
            cublasCreate(&cb_handle[i]);
            cublasSetPointerMode(cb_handle[i], CUBLAS_POINTER_MODE_HOST);
        }
        // compute block matrix partition
        parN[0] = 0;
        parN[pardim] = N;
        for (i = 1; i < pardim; i++)
            parN[i] = parN[i-1] + N/pardim;
        parK[0] = 0;
        parK[pardim] = K;
        for (i = 1; i < pardim; i++)
            parK[i] = parK[i-1] + K/pardim;

        // out-of-core matrix multiplication
        omp_set_num_threads(NGPU);
        for (j = 0; j < pardim; j++) {
            #pragma omp parallel for default(none) \
                private(i, k, id, blki, blkj, blkk) \
                firstprivate(j, NGPU, LDQ, K, N, pardim) \
                shared(parN, parK, cb_handle, WORK_dev, QHAT_dev, Q_dev, \
                       zero, one, Q, QHAT, QWORK)
            for (i = 0; i < pardim; i++) {
                blki = parN[i+1] - parN[i];
                blkk = parK[1] - parK[0];
                blkj = parK[j+1] - parK[j];
                id = omp_get_thread_num();
                cudaSetDevice(id);
                check(cublasSetMatrix(blki, blkk, sizeof(double),
                    &Q[parN[i]], LDQ, WORK_dev[id], blki));
                check(cublasSetMatrix(blkk, blkj, sizeof(double),
                    &QHAT[parK[j]*K], K, QHAT_dev[id], blkk));
                check(cublasDgemm(cb_handle[id], CUBLAS_OP_N, CUBLAS_OP_N,
                    blki, blkj, blkk, &one,
                    WORK_dev[id], blki, QHAT_dev[id], blkk, &zero,
                    Q_dev[id], blki));
                for (k = 1; k < pardim; k++) {
                    blkk = parK[k+1] - parK[k];
                    check(cublasSetMatrix(blki, blkk, sizeof(double),
                        &Q[parN[i] + parK[k]*LDQ], LDQ, WORK_dev[id], blki));
                    check(cublasSetMatrix(blkk, blkj, sizeof(double),
                        &QHAT[parK[k] + parK[j]*K], K, QHAT_dev[id], blkk));
                    check(cublasDgemm(cb_handle[id], CUBLAS_OP_N, CUBLAS_OP_N,
                        blki, blkj, blkk, &one,
                        WORK_dev[id], blki, QHAT_dev[id], blkk, &one,
                        Q_dev[id], blki));
                }
                check(cublasGetMatrix(blki, blkj, sizeof(double),
                    Q_dev[id], blki, &QWORK[parN[i] + parK[j]*N], N));
            }
        }
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', N, K, QWORK, N, Q, LDQ);

        // compute perm1 that would merge back deflated values.
        dlamrg(K, N-K, D, 1, -1, perm1);
    } else {
        for (i = 0; i < N; i++)
            perm1[i] = i;
    }

    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cublasDestroy(cb_handle[i]);
    }
    free(cb_handle);
    free(QHAT_dev);
    free(Q_dev);
    free(parK);
    free(parN);
}
