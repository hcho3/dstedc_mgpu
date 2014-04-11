#undef I
#define SQ(x)   ((x) * (x))
struct cfg_ent {
    double Gparam[6];
    double Cparam[6];
};

void dlaed0_m(long NGPU, long NCPUW, long N, double *D, double *E, double *Q,
    long LDQ, double *WORK, double **WORK_dev, long *IWORK, cfg_ent cfg);
void dlaed1_gpu(long N, double *D, double *Q, long LDQ, long *perm1,
    double RHO, long CUTPNT, double *WORK, double *WORK_dev, long *IWORK);
void dlaed1_cpu(long NCORE, long N, double *D, double *Q, long LDQ,
    long *perm1, double RHO, long CUTPNT, double *WORK, long *IWORK);
void dlaed1_ph2(long NGPU, long NCPUW, long N, double *D, double *Q, long LDQ,
    long *perm1, double RHO, long CUTPNT, double *WORK, double **WORK_dev,
    long *IWORK, cfg_ent cfg);
void dlaed2(long *K, long N, long N1, double *D, double *Q, long LDQ,
    long *perm1, double *RHO, double *Z, double *DWORK, double *QWORK,
    long *perm2, long *permacc, long *perm3); 
void dlaed3_gpu(long K, double *D, double *QHAT_dev, long LDQHAT, double RHO,
    double *DLAMDA, double *W, double *WORK_dev);
void dlaed3_cpu(long NCORE, long K, double *D, double *QHAT, long LDQHAT,
    double RHO, double *DLAMDA, double *W, double *S);
void dlaed3_ph2(long NGPU, long NCPUW, long K, double *D, double *QHAT,
    long LDQHAT, double RHO, double *DLAMDA, double *W, double **WORK_dev,
    double *S, cfg_ent cfg);
__global__ void dlaed4_gpu(long IL, long IU, long K, double *D, double *Z,
    double RHO, double *tau, double *orig);
void dlaed4_cpu(long K, long I, double *D, double *Z, double RHO, double *tau,
    double *orig);
double middle_way_cpu(long k, long n, double *delta, double *zeta, double rho,
    double tau, double orig);
void initial_guess_cpu(long k, long n, double *delta, double *zeta, double rho,
    double *tau, double *orig);
void dlaed4_ph2(long K, long I, double *D, double *Z, double RHO, double *tau,
    double *orig);
void dlamrg(long N1, long N2, double *A, long DTRD1, long DTRD2, long *perm);
double dlapy2(double x, double y);

long max_matsiz_host(void);
long max_matsiz_gpu(long NGPU);
long max_num_block(void);
double *allocate_work(long N);
double **allocate_work_dev(long NGPU, long N);
long *allocate_iwork(long N);
void free_work(double *WORK);
void free_work_dev(double **WORK_dev, long NGPU);
void free_iwork(long *IWORK);

struct cfg_ent load_cfg(const char *filename);
int compute_dlaed1_partition(const struct cfg_ent cfg, const int NGPU,
    const int NCPU, const int N, const int subpbs);
int compute_dlaed4_partition(const cfg_ent cfg, const int NGPU,
    const int NCPU, const int K);

double *read_mat(const char *filename, long *dims);
void write_mat(const char *filename, double *array, long *dims);
