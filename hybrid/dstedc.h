#undef I
#define SQ(x)   ((x) * (x))
void dstedc(int N, double *D, double *E, double *Z, int LDZ,
    double *WORK, double *WORK_dev, int LWORK, int *IWORK, int LIWORK);
void dlaed0(int N, double *D, double *E, double *Q, int LDQ,
    double *WORK, double *WORK_dev, int *IWORK);
void dlaed1(int N, double *D, double *Q, int LDQ, int *perm1, double RHO,
    int CUTPNT, double *WORK, double *WORK_dev, int *IWORK);
void dlaed2(int *K, int N, int N1, double *D, double *Q, int LDQ,
    int *perm1, double *RHO, double *Z, double *DWORK, double *QWORK,
    int *perm2, int *permacc, int *perm3); 
void dlaed3(int K, double *D, double *QHAT, int LDQHAT, double RHO,
    double *DLAMDA, double *DLAMDA_dev, double *W, double *W_dev, double *S,
    double *S_dev);
__global__ void dlaed4(int K, double *D, double *Z, double RHO,
    double *tau, double *orig);
void dlamrg(int N1, int N2, double *A, int DTRD1, int DTRD2, int *perm);
double dlapy2(double x, double y);
