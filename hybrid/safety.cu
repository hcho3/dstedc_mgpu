#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include "safety.h"

void _cuda_check(cudaError_t cs, long line)
{
    const char *errstr;

    if (cs != cudaSuccess) {
        errstr = cudaGetErrorString(cs);
        printf("CUDA error %s at %ld.\n", errstr, line);
        exit(1);
    }
}
void _cublas_check(cublasStatus_t cs, long line)
{
    const char *errstr;

    if (cs != CUBLAS_STATUS_SUCCESS) {
        switch(cs) {
            case CUBLAS_STATUS_SUCCESS:
                errstr = "CUBLAS_STATUS_SUCCESS"; break;
            case CUBLAS_STATUS_NOT_INITIALIZED:
                errstr = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                errstr = "CUBLAS_STATUS_ALLOC_FAILED"; break;
            case CUBLAS_STATUS_INVALID_VALUE:
                errstr = "CUBLAS_STATUS_INVALID_VALUE"; break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                errstr = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                errstr = "CUBLAS_STATUS_MAPPING_ERROR"; break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                errstr = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                errstr = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
            default:
                errstr = "unknown";
        }
        printf("CUBLAS error %s at %ld.\n", errstr, line);
        exit(1);
    }
}
