#include <stdio.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <omp.h>
#include "dstedc.h"
#include "nvtx.h"
#include "safety.h"
#include "timer.h"

#define SMLSIZ 128

static void get_bounds(long i, const long *partition, long *submat,
    long *matsiz, long *msd2);
static long get_NCORE(int cpu_portion, int NCPUW, int NGPU, int tid);

void dlaed0_m(long NGPU, long NCPUW, long N, double *D, double *E, double *Q,
    long LDQ, double *WORK, double **WORK_dev, long *IWORK, cfg_ent cfg)
/* computes all eigenvalues and corresponding eigenvectors of a symmetric
   tridiagonal matrix using the divide-and-conquer eigenvalue algorithm.
   We will have
      diag(D(in)) + diag(E, 1) + diag(E, -1) = Q * diag(D(out)) + Q'. */
{
    RANGE_START("dlaed0_m", 1, 0);

    long subpbs; // number of subproblems
    long i, j, k, tid, submat, smm1, msd2, matsiz;
    long pbcap = max_matsiz_gpu(NGPU); // limit on subproblem size
    long pbmax; // largest subproblem seen so far
    long *partition = &IWORK[0];
    long *perm1 = &IWORK[4*N];

    long NCORE; // # of CPU workers assigned to each specific problem
    long gpu_portion, cpu_portion, curlvl;

#ifdef USE_TIMER
    timeval timer1, timer2;
#endif

    // Determine the size and placement of the submatrices, and save in
    // the leading elements of IWORK.
    partition[0] = N;
    subpbs = 1;

    while (partition[subpbs-1] > SMLSIZ) {
        for (j = subpbs-1; j >= 0; j--) {
            partition[2*j + 1] = (partition[j] + 1) / 2;
            partition[2*j] = partition[j] / 2;
        }
        subpbs *= 2;
    }
    for (j = 1; j < subpbs; j++)
        partition[j] += partition[j-1];

    // Divide the matrix into subpbs submatricies of size at most
    // SMLSIZ+1 using rank-1 modifications (cuts).
    for (i = 0; i < subpbs - 1; i++) {
        submat = partition[i];
        smm1 = submat - 1;
        D[smm1] -= fabs(E[smm1]);
        D[submat] -= fabs(E[smm1]);
    }
    
    // Solve each submatrix eigenvalue problem at the bottom of the divide and
    // conquer tree.
    omp_set_num_threads(omp_get_num_procs());
#ifdef USE_TIMER
    get_time(&timer1);
#endif
    #pragma omp parallel for default(none) \
        private(i, j, k, submat, matsiz) firstprivate(subpbs, LDQ) \
        shared(partition, D, E, Q, WORK, perm1)
    for (i = -1; i < subpbs - 1; i++) {
        if (i == -1) {
            submat = 0;
            matsiz = partition[0];
        } else {
            submat = partition[i];
            matsiz = partition[i+1] - partition[i];
        }
        LAPACKE_dsteqr_work(LAPACK_COL_MAJOR, 'I', matsiz, &D[submat],
            &E[submat], &Q[submat + submat * LDQ], LDQ, &WORK[2*submat]);
        k = 0;
        for (j = submat; j < partition[i+1]; j++)
            perm1[j] = k++;
    }
#ifdef USE_TIMER
    get_time(&timer2);
    printf("  for: dsteqr = %6.2lf\n", get_elapsed_ms(timer1, timer2) / 1000.0);
#endif

    // Successively merge eigensystems of adjacent submatrices into
    // eigensystem for the corresponding larger matrix.
    pbmax = 0;
    
    /* Phase 1: Fine-grained, in-core */
    omp_set_num_threads((int)(NGPU+NCPUW));
    curlvl = 1;
    while (subpbs > 1) {
        // update pbmax.
        for (j = 0; j < subpbs/2; j++) {
            i = 2*j - 1;
            matsiz = (i == -1) ? partition[1] : partition[i+2]-partition[i];
            if (matsiz > pbmax)
                pbmax = matsiz;
        }
        if (pbmax > pbcap)
            break;

        gpu_portion =
            compute_dlaed1_partition(cfg, NGPU, NCPUW, pbmax, subpbs/2);
        cpu_portion = subpbs/2 - gpu_portion;
        //printf("gpu_portion = %ld, cpu_portion = %ld\n",
        //    gpu_portion, cpu_portion);

        RANGE_START("level", 2, 5);
#ifdef USE_TIMER
        get_time(&timer1);
#endif
        #pragma omp parallel default(none) \
            private(i, j, tid, submat, matsiz, msd2, NCORE) \
            shared(partition, D, Q, perm1, E, WORK, IWORK, WORK_dev) \
            firstprivate(N, subpbs, LDQ, NGPU, NCPUW, gpu_portion, cpu_portion)
        {
            tid = omp_get_thread_num();
            if (tid < NGPU) { // this thread controls a GPU worker
                for (j = tid; j < gpu_portion; j += NGPU) {
                    i = 2*j - 1;
                    get_bounds(i, partition, &submat, &matsiz, &msd2);

                    // Merge lower order eigensystems (of size msd2 and
                    // matsiz - msd2) into an eigensystem of size matsiz.
                    safe_cudaSetDevice(tid);
                    dlaed1_gpu(matsiz, &D[submat], &Q[submat + submat * LDQ],
                        LDQ, &perm1[submat], E[submat+msd2-1], msd2,
                        &WORK[submat*(2*N+2*N*N)/N],
                        WORK_dev[tid], &IWORK[subpbs+3*submat]);
                }
            } else { // this thread is itself a worker
                for (j = gpu_portion + tid-NGPU; j < subpbs/2; j += NCPUW) {
                    // determine # of CPU workers to assign to this subproblem
                    NCORE = get_NCORE(cpu_portion, NCPUW, NGPU, tid);

                    i = 2*j - 1;
                    get_bounds(i, partition, &submat, &matsiz, &msd2);
                    
                    // Merge lower order eigensystems (of size msd2 and
                    // matsiz - msd2) into an eigensystem of size matsiz.
                    dlaed1_cpu(NCORE, matsiz, &D[submat],
                        &Q[submat + submat * LDQ], LDQ, &perm1[submat],
                        E[submat+msd2-1], msd2, &WORK[submat*(2*N+2*N*N)/N],
                        &IWORK[subpbs+3*submat]);
                }
            }
        }
        // update partition.
        for (i = -1; i < subpbs - 2; i += 2)
            partition[(i-1)/2 + 1] = partition[i+2];
        subpbs /= 2;
        ++curlvl;

#ifdef USE_TIMER
        get_time(&timer2);
        printf("%ld: time:  %6.2lf\n\n",
            curlvl, get_elapsed_ms(timer1, timer2) / 1000.0);
#endif
        RANGE_END(2);
    }

    /* Phase 2: Coarse-grained, out-of-core */
    while (subpbs > 1) {
        // update pbmax.
        for (j = 0; j < subpbs/2; j++) {
            i = 2*j - 1;
            matsiz = (i == -1) ? partition[1] : partition[i+2]-partition[i];
            if (matsiz > pbmax)
                pbmax = matsiz;
        }

        RANGE_START("level", 3, 5);
#ifdef USE_TIMER
        get_time(&timer1);
#endif
        for (j = 0; j < subpbs/2; j++) {
            i = 2*j - 1;
            if (i == -1) {
                submat = 0;
                matsiz = partition[1];
                msd2 = partition[0];
            } else {
                submat = partition[i];
                matsiz = partition[i+2] - partition[i];
                msd2 = matsiz / 2;
            }

            // Merge lower order eigensystems (of size msd2 and matsiz - msd2)
            // into an eigensystem of size matsiz.
            dlaed1_ph2(NGPU, NCPUW, matsiz, &D[submat],
                &Q[submat + submat * LDQ], LDQ, &perm1[submat],
                E[submat+msd2-1], msd2, WORK, WORK_dev, &IWORK[subpbs], cfg);
        }
        // update partition.
        for (i = -1; i < subpbs - 2; i += 2)
            partition[(i-1)/2 + 1] = partition[i+2];
        subpbs /= 2;
        ++curlvl;

#ifdef USE_TIMER
        get_time(&timer2);
        printf("%ld: time: %6.2lf s\n",
            curlvl, get_elapsed_ms(timer1, timer2) / 1000.0);
#endif
        RANGE_END(3);
    }
    
    // Re-merge the eigenvalues and eigenvectors which were deflated at the
    // final merge step.
    // D = D(perm1);
    for (i = 0; i < N; i++)
        WORK[i] = D[perm1[i]];
    cblas_dcopy(N, WORK, 1, D, 1);

    // Q = Q(perm1);
    for (j = 0; j < N; j++) {
        i = perm1[j];
        cblas_dcopy(N, &Q[i * LDQ], 1, &WORK[(j + 1) * N], 1);
    }
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', N, N, &WORK[N], N, Q, LDQ); 

    RANGE_END(1);
}

static void get_bounds(long i, const long *partition, long *submat,
    long *matsiz, long *msd2)
{
    if (i == -1) {
        *submat = 0;
        *matsiz = partition[1];
        *msd2 = partition[0];
    } else {
        *submat = partition[i];
        *matsiz = partition[i+2] - partition[i];
        *msd2 = *matsiz / 2;
    }
}

static long get_NCORE(int cpu_portion, int NCPUW, int NGPU, int tid)
{
    int NCORE;

    if (cpu_portion >= NCPUW) {
        NCORE = 1;
    } else { // cpu_portion < NCPUW
        NCORE = NCPUW / cpu_portion;
        if (tid-NGPU < (NCPUW % cpu_portion))
            NCORE++;
    }

    return NCORE;
}
