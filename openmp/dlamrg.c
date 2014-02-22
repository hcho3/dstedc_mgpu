void dlamrg(long N1, long N2, double *A, long DTRD1, long DTRD2, long *perm)
// computes a permutation which merges two sorted lists A(1:N1) and A(N1+1:end)
// into a single sorted list in ascending order.
{
    long i = (DTRD1 > 0) ? 0 : (N1 - 1);
    long j = (DTRD2 > 0) ? N1 : (N1 + N2 - 1);
    long k;
    long idx = 0;

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
