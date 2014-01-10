#undef I
#define SQ(x)   ((x) * (x))
void dlaed0_m(long NGPU, long N, double *D, double *E, double *Q, long LDQ,
    double **WORK, double **WORK_dev, long **IWORK);
void dlaed1(long N, double *D, double *Q, long LDQ, long *perm1, double RHO,
    long CUTPNT, double *WORK, double *WORK_dev, long *IWORK);
void dlaed1_ph2(long NGPU, long N, double *D, double *Q, long LDQ, long *perm1,
    double RHO, long CUTPNT, double *WORK, double **WORK_dev, long *IWORK);
void dlaed2(long *K, long N, long N1, double *D, double *Q, long LDQ,
    long *perm1, double *RHO, double *Z, double *DWORK, double *QWORK,
    long *perm2, long *permacc, long *perm3); 
void dlaed3(long K, double *D, double *QHAT_dev, long LDQHAT, double RHO,
    double *DLAMDA, double *W, double *S, double *WORK_dev);
void dlaed3_ph2(long K, double *D, double *QHAT, long LDQHAT, double RHO,
    double *DLAMDA, double *W, double *S);
__global__ void dlaed4(long K, double *D, double *Z, double RHO,
    double *tau, double *orig);
void dlaed4_ph2(long K, long I, double *D, double *Z, double RHO, double *tau,
    double *orig);
void dlamrg(long N1, long N2, double *A, long DTRD1, long DTRD2, long *perm);
double dlapy2(double x, double y);

long max_matsiz_host(long NGPU);
long max_matsiz_gpu(long NGPU);
double **allocate_work(long NGPU, long N);
double **allocate_work_dev(long NGPU, long N);
long **allocate_iwork(long NGPU, long N);
void free_work(double **WORK, long NGPU);
void free_work_dev(double **WORK_dev, long NGPU);
void free_iwork(long **IWORK, long NGPU);

double *read_mat(const char *filename, long *dims);
void write_mat(const char *filename, double *array, long *dims);
