#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "dstedc.h"

#include "dlapy2.cu"
#include "dlacpy.cu"
#include "dlamrg.cu"
#include "dlaed3.cu"
#include "dlaed2.cu"
#include "dlaed1.cu"
// CUDA toolchain does not support inlining of a function in different
// compilation unit.

#define SMLSIZ 25

void dlaed0_bootstrap(int N, double *D, double *D_dev, double *E,
	double *E_dev, double *Q, double *Q_dev, int LDQ, double *WORK_dev,
	int *IWORK_dev)
{
    int subpbs; // number of submatrices
    int i, j, k, submat, smm1, matsiz;
    int *partition = (int *)malloc(N * sizeof(int));
    int *perm1 = (int *)malloc(N * sizeof(int));

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
    for (i = -1; i < subpbs - 1; i++) {
        if (i == -1) {
            submat = 0;
            matsiz = partition[0];
        } else {
            submat = partition[i];
            matsiz = partition[i+1] - partition[i];
        }
        LAPACKE_dsteqr(LAPACK_COL_MAJOR, 'I', matsiz, &D[submat],
            &E[submat], &Q[submat + submat * LDQ], LDQ);
        k = 0;
        for (j = submat; j < partition[i+1]; j++)
            perm1[j] = k++;
    }

	// Marshall inputs to device
	cudaMemcpy(D_dev, D, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(E_dev, E, (N-1) * sizeof(double), cudaMemcpyHostToDevice);
    cublasSetMatrix(N, N, sizeof(double), Q, LDQ, Q_dev, LDQ);
	cudaMemcpy(IWORK_dev, partition, subpbs * sizeof(int),
		cudaMemcpyHostToDevice);
    cudaMemcpy(&IWORK_dev[4*N+3], perm1, N * sizeof(int),
        cudaMemcpyHostToDevice);

	dlaed0<<<1, 1>>>(subpbs, N, D_dev, E_dev, Q_dev, LDQ, WORK_dev, IWORK_dev);
	cudaDeviceSynchronize();

	// Marshall outputs to host
	cudaMemcpy(D, D_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    cublasGetMatrix(N, N, sizeof(double), Q_dev, LDQ, Q, LDQ);

    free(partition);
    free(perm1);
}

__global__ void dlaed0(int subpbs, int N, double *D, double *E, double *Q,
    int LDQ, double *WORK, int *IWORK)
/* computes all eigenvalues and corresponding eigenvectors of a symmetric
   tridiagonal matrix using the divide-and-conquer eigenvalue algorithm.
   We will have
      diag(D(in)) + diag(E, 1) + diag(E, -1) = Q * diag(D(out)) + Q'. */
{
    int i, j, submat, msd2, curprb, matsiz;
    int *partition = &IWORK[0];
    int *perm1 = &IWORK[4*N + 3];

    cublasHandle_t cb_handle;
    cublasCreate(&cb_handle);

    // Successively merge eigensystems of adjacent submatrices into
    // eigensystem for the corresponding larger matrix.
    while (subpbs > 1) {
        for (i = -1; i < subpbs - 2; i += 2) {
            if (i == -1) {
                submat = 0;
                matsiz = partition[1];
                msd2 = partition[0];
                curprb = 0;
            } else {
                submat = partition[i];
                matsiz = partition[i+2] - partition[i];
                msd2 = matsiz / 2;
                curprb++;
            }
            // Merge lower order eigensystems (of size msd2 and matsiz - msd2)
            // into an eigensystem of size matsiz.
            dlaed1(cb_handle, matsiz, &D[submat], &Q[submat + submat * LDQ],
                LDQ, &perm1[submat], E[submat+msd2-1], msd2, WORK,
                &IWORK[subpbs]);
            partition[(i-1)/2 + 1] = partition[i+2];
        }
        subpbs /= 2;
    }
    
    // Re-merge the eigenvalues and eigenvectors which were deflated at the
    // final merge step.
    // D = D(perm1);
    for (i = 0; i < N; i++)
        WORK[i] = D[perm1[i]];
    cublasDcopy(cb_handle, N, WORK, 1, D, 1);

    // Q = Q(perm1);
    for (j = 0; j < N; j++) {
        i = perm1[j];
        cublasDcopy(cb_handle, N, &Q[i * LDQ], 1, &WORK[(j + 1) * N], 1);
    }
    dlacpy(N, N, &WORK[N], N, Q, LDQ); 

	cublasDestroy(cb_handle);
}
