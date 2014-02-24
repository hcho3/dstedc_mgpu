#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <lapacke.h>
#include <cuda_runtime.h>

#define TEST_DSTDC "testmat/dstedc"
#define TEST_DGEMM "testmat/dgemm"

struct input_ent {
    int prbsiz;
    int nworker;
    double perf;
};

int main(void)
{
    char buf[BUFSIZ] = {0};
    FILE *pipein_fp, *cfg_fp;
    int NCPU, NGPU;
    int i, inc;
    int input_len, input_cap;
    struct input_ent *input;

    /*
        Data matrices

      * CX     = measurements of independent variables in CPU performance
        dimension (*, 3)
            X1 = subproblem size
            X2 = number of CPU cores used
            extra column used to hold 1's

      * CY     = measurements of dependent varable in CPU performance
        dimension (*, 1)
            Y  = CPU performance (sec)

      * GX     = measurements of independent variables in GPU performance
        dimension (*, 3)
            X1 = subproblem size
            X2 = number of GPU devices used
            extra column used to hold 1's

      * GY     = measurements of dependent varable in GPU performance
        dimension (*, 1)
            Y  = GPU performance (sec)
    */
    int Clen, Glen;
    double *CX, *CY, *GX, *GY;

    NCPU = (int)sysconf( _SC_NPROCESSORS_ONLN );
    cudaGetDeviceCount(&NGPU);

    printf("Number of CPU cores   = %d\n", NCPU);
    printf("Number of GPU devices = %d\n", NGPU);

    input_len = 0;
    input_cap = 32;
    input = (struct input_ent *)malloc(input_cap * sizeof(struct input_ent));
    for (i = 1; i <= NGPU; i++) { 
        sprintf(buf, "prof/gpu/gpuprof %d %s/D_8192.bin %s/E_8192.bin "
            "Dout.bin Q.bin quiet", i, TEST_DSTDC, TEST_DSTDC);
        if ( (pipein_fp = popen(buf, "r")) == NULL) {
            perror("popen");
            exit(1);
        }
        while (fgets(buf, BUFSIZ, pipein_fp)) {
            sscanf(buf, "%d:%d:%lf", &input[input_len].prbsiz,
                &input[input_len].nworker, &input[input_len].perf);
            input_len++;
            if (input_len == input_cap) {
                input_cap *= 2;
                input = (struct input_ent *)
                    realloc(input, input_cap * sizeof(struct input_ent));
            }
        }
        
        pclose(pipein_fp);
    }

    // transform input
    Glen = input_len;
    GX = (double *)malloc(Glen * 3 * sizeof(double));
    GY = (double *)malloc(Glen * sizeof(double)); 
    for (i = 0; i < Glen; i++) {
        GX[i]          = log2((double)input[i].prbsiz);
        GX[Glen   + i] = log2((double)input[i].nworker);
        GX[Glen*2 + i] = 1.0;
        GY[i]          = log2(input[i].perf);
    }
    free(input);

    input_len = 0;
    input_cap = 32;
    input = (struct input_ent *)malloc(input_cap * sizeof(struct input_ent));
    inc = (NCPU/8 > 1) ? NCPU/8 : 1;
    for (i = inc; i <= NCPU; i += inc) {
        sprintf(buf, "prof/cpu/cpuprof %d %d %s/D_8192.bin %s/E_8192.bin "
            "Dout.bin Q.bin quiet", i, i, TEST_DSTDC, TEST_DSTDC);
        if ( (pipein_fp = popen(buf, "r")) == NULL) {
            perror("popen");
            exit(1);
        }
        while (fgets(buf, BUFSIZ, pipein_fp)) {
            sscanf(buf, "%d:%d:%lf", &input[input_len].prbsiz,
                &input[input_len].nworker, &input[input_len].perf);
            input_len++;
            if (input_len == input_cap) {
                input_cap *= 2;
                input = (struct input_ent *)
                    realloc(input, input_cap * sizeof(struct input_ent));
            }
        }
        
        pclose(pipein_fp);
    }

    // transform input
    Clen = input_len;
    CX = (double *)malloc(Clen * 3 * sizeof(double));
    CY = (double *)malloc(Clen * sizeof(double));

    for (i = 0; i < Clen; i++) {
        CX[i]          = log2((double)input[i].prbsiz);
        CX[Clen   + i] = log2((double)input[i].nworker);
        CX[Clen*2 + i] = 1.0;
        CY[i]          = log2(input[i].perf);
    }
    free(input);

    // estimate parameters using linear least-squares.
    LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', Glen, 3, 1, GX, Glen, GY, Glen);
    LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', Clen, 3, 1, CX, Clen, CY, Clen);

    cfg_fp = fopen("params.cfg", "w");
    if (!cfg_fp) {
        perror("fopen");
        exit(1);
    }
    
    fprintf(cfg_fp, "# Parameters for GPU performance model: dstedc\n"
                    "# Y = (X1^P1)(X2^P2)(2^P3)\n"
                    "#     where Y  = performance (sec)\n"
                    "#           X1 = subproblem size\n"
                    "#           X2 = # of GPU devices used\n");
    for (i = 0; i < 3; i++)
        fprintf(cfg_fp, "Gparam[%d] = %.20lf\n", i, GY[i]);
    fprintf(cfg_fp, "# Parameters for CPU performance model: dstedc\n");
    fprintf(cfg_fp, "# Y = (X1^P1)(X2^P2)(2^P3)\n"
                    "#     where Y  = performance (sec)\n"
                    "#           X1 = subproblem size\n"
                    "#           X2 = # of CPU cores used\n");
    for (i = 0; i < 3; i++)
        fprintf(cfg_fp, "Cparam[%d] = %.20lf\n", i, CY[i]);

    printf("GPU performance: Y = (%.5le)*(X1^%.5lf)*(X2^%.5lf)\n",
           exp2(GY[2]), GY[0], GY[1]);
    printf("CPU performance: Y = (%.5le)*(X1^%.5lf)*(X2^%.5lf)\n",
           exp2(CY[2]), CY[0], CY[1]);
    printf("Performance profile has been recorded to params.cfg.\n");
    
    free(GX);
    free(GY);
    free(CX);
    free(CY);

    /* profile dgemm */
    /*
        Data matrices

      * CX     = measurements of independent variables in CPU performance
        dimension (*, 2)
            X1 = subproblem size
            extra column used to hold 1's

      * CY     = measurements of dependent varable in CPU performance
        dimension (*, 1)
            Y  = CPU performance (sec)

      * GX     = measurements of independent variables in GPU performance
        dimension (*, 2)
            X1 = subproblem size
            extra column used to hold 1's

      * GY     = measurements of dependent varable in GPU performance
        dimension (*, 1)
            Y  = GPU performance (sec)
    */
    input_len = 0;
    input_cap = 32;
    input = (struct input_ent *)malloc(input_cap * sizeof(struct input_ent));
    fprintf(stderr, "Testing dgemm on GPU...\n");
    for (i = 1024; i <= 8192; i *= 2) { 
        sprintf(buf, "prof/dgemm/gpu %s/A_%d.bin %s/B_%d.bin C.bin quiet", 
            TEST_DGEMM, i, TEST_DGEMM, i);
        if ( (pipein_fp = popen(buf, "r")) == NULL) {
            perror("popen");
            exit(1);
        }
        while (fgets(buf, BUFSIZ, pipein_fp)) {
            sscanf(buf, "%d:%lf", &input[input_len].prbsiz,
                &input[input_len].perf);
            input_len++;
            if (input_len == input_cap) {
                input_cap *= 2;
                input = (struct input_ent *)
                    realloc(input, input_cap * sizeof(struct input_ent));
            }
        }
        
        pclose(pipein_fp);
    }
    Glen = input_len;
    GX = (double *)malloc(Glen * 2 * sizeof(double));
    GY = (double *)malloc(Glen * sizeof(double));

    for (i = 0; i < Glen; i++) {
        GX[i]        = log2((double)input[i].prbsiz);
        GX[Glen + i] = 1.0;
        GY[i]        = log2(input[i].perf);
    }
    free(input);

    input_len = 0;
    input_cap = 32;
    input = (struct input_ent *)malloc(input_cap * sizeof(struct input_ent));
    fprintf(stderr, "Testing dgemm on CPU...\n");
    for (i = 1024; i <= 8192; i *= 2) { 
        sprintf(buf, "prof/dgemm/cpu %s/A_%d.bin %s/B_%d.bin C.bin quiet", 
            TEST_DGEMM, i, TEST_DGEMM, i);
        if ( (pipein_fp = popen(buf, "r")) == NULL) {
            perror("popen");
            exit(1);
        }
        while (fgets(buf, BUFSIZ, pipein_fp)) {
            sscanf(buf, "%d:%lf", &input[input_len].prbsiz,
                &input[input_len].perf);
            input_len++;
            if (input_len == input_cap) {
                input_cap *= 2;
                input = (struct input_ent *)
                    realloc(input, input_cap * sizeof(struct input_ent));
            }
        }
        
        pclose(pipein_fp);
    }
    Clen = input_len;
    CX = (double *)malloc(Clen * 2 * sizeof(double));
    CY = (double *)malloc(Clen * sizeof(double));

    for (i = 0; i < Clen; i++) {
        CX[i]        = log2((double)input[i].prbsiz);
        CX[Clen + i] = 1.0;
        CY[i]        = log2(input[i].perf);
    }
    free(input);

    // estimate parameters using linear least-squares.
    LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', Glen, 2, 1, GX, Glen, GY, Glen);
    LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', Clen, 2, 1, CX, Clen, CY, Clen);

    fprintf(cfg_fp, "# Parameters for GPU performance model: dgemm\n"
                    "# Y = (X1^P1)(2^P2)\n"
                    "#     where Y  = performance (sec)\n"
                    "#           X1 = subproblem size\n");
    for (i = 0; i < 2; i++)
        fprintf(cfg_fp, "Gparam[%d] = %.20lf\n", i, GY[i]);
    fprintf(cfg_fp, "# Parameters for CPU performance model: dgemm\n");
    fprintf(cfg_fp, "# Y = (X1^P1)(2^P2)\n"
                    "#     where Y  = performance (sec)\n"
                    "#           X1 = subproblem size\n");
    for (i = 0; i < 2; i++)
        fprintf(cfg_fp, "Cparam[%d] = %.20lf\n", i, CY[i]);

    printf("GPU performance: Y = (%.5le)*(X1^%.5lf)\n",
           exp2(GY[1]), GY[0]);
    printf("CPU performance: Y = (%.5le)*(X1^%.5lf)\n",
           exp2(CY[1]), CY[0]);
    printf("Performance profile has been recorded to params.cfg.\n");

    fclose(cfg_fp);
    free(GX);
    free(GY);
    free(CX);
    free(CY);

    return 0;
}
