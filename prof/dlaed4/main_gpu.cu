#include <stdio.h>
#include <omp.h>
#include "safety.h"
#include "timer.h"

double *read_mat(const char *filename, long *dims);
void write_mat(const char *filename, double *array, long *dims);
__global__ void dlaed4_gpu(long IL, long IU, long K, double *D, double *Z,
    double RHO, double *tau, double *orig);
double **allocate_work_dev(long NGPU, long N);
void free_work_dev(double **WORK_dev, long NGPU);
long max_num_block(void);

int main(int argc, char **argv)
{
    const char *fRHO    = argv[2];
    const char *fW      = argv[3];
    const char *fDLAMDA = argv[4];

    double *W, *DLAMDA;
    double *tau, *orig, *rho;
    double RHO;
    long dims[2];
    long K;
    long tid;
    long NGPU, maxNGPU;
    int temp;
    timeval timer1, timer2;

    double **WORK_dev;
    double *DLAMDA_dev, *tau_dev, *orig_dev, *W_dev;
    double tmp;

    safe_cudaGetDeviceCount(&temp);
    maxNGPU = (long)temp;

    if (argc < 5 ||
        sscanf(argv[1], "%ld", &NGPU) < 1 || NGPU <= 0 || NGPU > maxNGPU ) {
        fprintf(stderr, "Usage: %s [# of GPUs] [RHO.bin] "
                        "[W.bin] [DLAMDA.bin]\n", argv[0]);
        exit(1);
    }

    rho = read_mat(fRHO, dims);
    RHO = rho[0];
    W = read_mat(fW, dims);
    DLAMDA = read_mat(fDLAMDA, dims);
    K = dims[0];
    tau = (double *)malloc(K * sizeof(double));
    orig = (double *)malloc(K * sizeof(double));

    WORK_dev = allocate_work_dev(NGPU, 16384);

    omp_set_num_threads((int)NGPU);
    get_time(&timer1);
    #pragma omp parallel default(none) \
        private(tid, W_dev, DLAMDA_dev, tau_dev, orig_dev) \
        firstprivate(NGPU, K, RHO) \
        shared(W, DLAMDA, tau, orig, WORK_dev)
    {
        tid = omp_get_thread_num();
        safe_cudaSetDevice(tid);
        W_dev = &WORK_dev[tid][0];
        DLAMDA_dev = &WORK_dev[tid][K];
        tau_dev    = &WORK_dev[tid][2*K];
        orig_dev   = &WORK_dev[tid][3*K];
        safe_cudaMemcpy(W_dev, W, K * sizeof(double),
            cudaMemcpyHostToDevice); 
        safe_cudaMemcpy(DLAMDA_dev, DLAMDA, K * sizeof(double),
            cudaMemcpyHostToDevice);
        
        long IL = tid * K / NGPU;
        long IU = (tid+1) * K / NGPU;
        if (tid == NGPU-1)
            IU = K;

        long maxNBLK = max_num_block();
        long NBLK = (IU-IL+TPB-1)/TPB;
        if (NBLK > maxNBLK)
            NBLK = maxNBLK;

        dlaed4_gpu<<<NBLK, TPB>>>(IL, IU, K, DLAMDA_dev, W_dev, RHO,
            tau_dev, orig_dev); 
        safe_cudaMemcpy(&tau[IL], &tau_dev[IL], (IU-IL) * sizeof(double),
            cudaMemcpyDeviceToHost);
        safe_cudaMemcpy(&orig[IL], &orig_dev[IL], (IU-IL) * sizeof(double),
            cudaMemcpyDeviceToHost);
    }
    get_time(&timer2);
    tmp = get_elapsed_ms(timer1, timer2) / 1000.0;
    printf("GPU:%ld:%.20lf\n", NGPU, tmp);
    fprintf(stderr, "# eigenvalues = %ld, %ld GPUs: %.3lf s\n", K, NGPU, tmp);

    free(W);
    free(DLAMDA);
    free(rho);
    free(tau);
    free(orig);

    free_work_dev(WORK_dev, NGPU);

    return 0;
}
