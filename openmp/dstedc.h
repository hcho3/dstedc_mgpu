#undef I
#define SQ(x)   ((x) * (x))
void dlaed0_m(long NGRP, long NCORE, long N, double *D, double *E, double *Q,
    long LDQ, double *WORK, long *IWORK);
void dlaed1(long NCOREP, long N, double *D, double *Q, long LDQ, long *perm1,
    double RHO, long CUTPNT, double *WORK, long *IWORK);
void dlaed2(long *K, long N, long N1, double *D, double *Q, long LDQ,
    long *perm1, double *RHO, double *Z, double *DWORK, double *QWORK,
    long *perm2, long *permacc, long *perm3);
void dlaed3(long NCORE, long K, double *D, double *QHAT, long LDQHAT,
    double RHO, double *DLAMDA, double *W, double *S);
void dlaed4(long K, long I, double *D, double *Z, double RHO, double *tau,
    double *orig);
void initial_guess(long k, long n, double *delta, double *zeta, double rho,
    double *tau, double *orig);
double middle_way(long k, long n, double *delta, double *zeta, double rho,
    double tau, double orig);
void dlamrg(long N1, long N2, double *A, long DTRD1, long DTRD2, long *perm);
double dlapy2(double x, double y);

double *read_mat(const char *filename, long *dims);
void write_mat(const char *filename, double *array, long *dims);

long max_matsiz_host(void);
double *allocate_work(long N);
long *allocate_iwork(long N);
void free_work(double *WORK);
void free_iwork(long *IWORK);
