#undef I
#define SQ(x)   ((x) * (x))
void dlaed0_m(int NGPU, int N, double *D, double *E, double *Q, int LDQ,
    double **WORK, double **WORK_dev, int **IWORK);
void dlaed1(int N, double *D, double *Q, int LDQ, int *perm1, double RHO,
    int CUTPNT, double *WORK, double *WORK_dev, int *IWORK);
void dlaed2(int *K, int N, int N1, double *D, double *Q, int LDQ,
    int *perm1, double *RHO, double *Z, double *DWORK, double *QWORK,
    int *perm2, int *permacc, int *perm3); 
void dlaed3(int K, double *D, double *QHAT_dev, int LDQHAT, double RHO,
    double *DLAMDA, double *W, double *S, double *WORK_dev);
__global__ void dlaed4(int K, double *D, double *Z, double RHO,
    double *tau, double *orig);
void dlamrg(int N1, int N2, double *A, int DTRD1, int DTRD2, int *perm);
double dlapy2(double x, double y);

long max_matsiz_host(int NGPU);
long max_matsiz_gpu(int NGPU);
double **allocate_work(int NGPU, int N);
double **allocate_work_dev(int NGPU, int N);
int **allocate_iwork(int NGPU, int N);
void free_work(double **WORK, int NGPU);
void free_work_dev(double **WORK_dev, int NGPU);
void free_iwork(int **IWORK, int NGPU);
