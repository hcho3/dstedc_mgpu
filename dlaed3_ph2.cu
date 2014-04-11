#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "dstedc.h"
#include "nvtx.h"
#include "safety.h"
#include "timer.h"

void dlaed3_ph2(long NGPU, long NCPUW, long K, double *D, double *QHAT,
    long LDQHAT, double RHO, double *DLAMDA, double *W, double **WORK_dev,
    double *S, cfg_ent cfg)
// stably computes the eigendecomposition Q * diag(lambda) * Q**T  of
// diag(delta) + RHO * z * z**T  by solving an inverse eigenvalue problem.
{
    RANGE_START("dlaed3_ph2", 1, 3);

    double *tau  = &S[0];
    double *orig = &S[K];
    double *v    = &S[2 * K];

    double *DLAMDA_dev, *W_dev, *tau_dev, *orig_dev;

    long i, j, tid;
    long gpu_portion, cpu_portion;
    double temp;

#ifdef USE_TIMER
    timeval timer1, timer2;
#endif

    omp_set_num_threads((int)(NGPU+NCPUW));
    //printf("dlaed3_ph2 starting... NGPU+NCPUW=%ld\n", NGPU+NCPUW);
#ifdef USE_TIMER
    get_time(&timer1);
#endif

    gpu_portion = compute_dlaed4_partition(cfg, NGPU, NCPUW, K);
    cpu_portion = K - gpu_portion;
    #pragma omp parallel default(none) \
        private(i, j, tid, temp, DLAMDA_dev, W_dev, tau_dev, orig_dev) \
        firstprivate(K, RHO, LDQHAT, NGPU, NCPUW, gpu_portion, cpu_portion) \
        shared(D, DLAMDA, QHAT, W, v, tau, orig, colors, WORK_dev)
    {
        // solve the secular equation
        RANGE_START("dlaed3_ph2_dlaed4", 2, 6);
        tid = omp_get_thread_num();
        if (tid < NGPU) { // this thread controls a GPU worker
            safe_cudaSetDevice(tid);
            W_dev = &WORK_dev[tid][0];
            DLAMDA_dev = &WORK_dev[tid][K];
            tau_dev    = &WORK_dev[tid][2*K];
            orig_dev   = &WORK_dev[tid][3*K];
            safe_cudaMemcpy(W_dev, W, K * sizeof(double),
                cudaMemcpyHostToDevice); 
            safe_cudaMemcpy(DLAMDA_dev, DLAMDA, K * sizeof(double),
                cudaMemcpyHostToDevice);
            
            long IL = tid * gpu_portion / NGPU;
            long IU = (tid+1) * gpu_portion / NGPU;
            if (tid == NGPU-1)
                IU = gpu_portion;
            //printf("tid %ld: %ld to %ld\n", tid, IL, IU);

            long maxNBLK = max_num_block();
            long NBLK = (IU-IL+TPB-1)/TPB;
            if (NBLK > maxNBLK)
                NBLK = maxNBLK;

            dlaed4_gpu<<<NBLK,TPB,0>>>(IL, IU, K, DLAMDA_dev, W_dev,
                RHO, tau_dev, orig_dev); 
            //safe_cudaDeviceSynchronize();
            //safe_cudaGetLastError();
            safe_cudaMemcpy(&tau[IL], &tau_dev[IL], (IU-IL) * sizeof(double),
                cudaMemcpyDeviceToHost);
            safe_cudaMemcpy(&orig[IL], &orig_dev[IL], (IU-IL) * sizeof(double),
                cudaMemcpyDeviceToHost);
        } else { // this thread is itself a worker
            long IL = gpu_portion + (tid-NGPU) * cpu_portion / NCPUW; 
            long IU = gpu_portion + (tid-NGPU + 1) * cpu_portion / NCPUW;
            //printf("tid %ld: %ld to %ld\n", tid, IL, IU);

            if (tid == NGPU+NCPUW-1)
                IU = K;
            for (i = IL; i < IU; i++)
                dlaed4_cpu(K, i, DLAMDA, W, RHO, &tau[i], &orig[i]);
        }
        RANGE_END(2);
        #pragma omp barrier

        // inverse eigenvalue problem: find v such that lambda(1), ...,
        // lambda(n) are exact eigenvalues of the matrix D + v * v**T.
        RANGE_START("dlaed3_ph2_inveig", 3, 6);
        #pragma omp for
        for (j = 0; j < K; j++) {
            temp = orig[j] - DLAMDA[j] + tau[j];
            for (i = 0; i < j; i++)
                temp *= ((DLAMDA[j] - orig[i] - tau[i])
                         / (DLAMDA[j] - DLAMDA[i]));
            for (i = j+1; i < K; i++)
                temp *= ((orig[i] - DLAMDA[j] + tau[i])
                        / (DLAMDA[i] - DLAMDA[j]));
            temp = copysign(sqrt(temp), W[j]);
            v[j] = temp;
        }
        RANGE_END(3);

        // compute the eigenvectors of D + v * v**T
        RANGE_START("dlaed3_ph2_eigvec", 4, 6);
        #pragma omp for
        for (j = 0; j < K; j++) {
            D[j] = tau[j] + orig[j];
            for (i = 0; i < K; i++)
                QHAT[i + j * LDQHAT] = v[i] / (orig[j] - DLAMDA[i] + tau[j]);
            temp = cblas_dnrm2(K, &QHAT[j * LDQHAT], 1); 
            for (i = 0; i < K; i++)
                QHAT[i + j * LDQHAT] /= temp;
        }
        RANGE_END(4);
    }

#ifdef USE_TIMER
    get_time(&timer2);
    printf("dlaed3_ph2 done. time = %6.2lf s\n",
        get_elapsed_ms(timer1, timer2) / 1000.0);
#endif

    RANGE_END(1);
}
