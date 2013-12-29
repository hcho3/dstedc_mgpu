__device__ __forceinline__ void dlacpy(int M, int N, double *A, int LDA,
    double *B, int LDB)
{
    int i, j;
    for (j = 0; j < N; j++)
        for (i = 0; i < M; i++) 
            B[i + j * LDB] = A[i + j * LDA];
}
