__device__ __forceinline__ void dlamrg(int N1, int N2, double *A,
    int DTRD1, int DTRD2, int *perm)
// computes a permutation which merges two sorted lists A(1:N1) and A(N1+1:end)
// into a single sorted list in ascending order.
{
    int i = (DTRD1 > 0) ? 0 : (N1 - 1);
    int j = (DTRD2 > 0) ? N1 : (N1 + N2 - 1);
    int k;
    int idx = 0;

    while (N1 > 0 && N2 > 0) {
        if (A[i] <= A[j]) {
            perm[idx] = i;
            idx++;
            i += DTRD1;
            N1--;
        } else {
            perm[idx] = j;
            idx++;
            j += DTRD2;
            N2--;
        }
    }
    if (N1 == 0) {
        for (k = 0; k < N2; k++) {
            perm[idx] = j;
            idx++;
            j += DTRD2;
        }
    } else {
        for (k = 0; k < N1; k++) {
            perm[idx] = i;
            idx++;
            i += DTRD1;
        }
    }
}
