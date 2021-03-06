#ifdef SAFE_CUDA
#define safe_cudaSetDevice(...) \
    _cuda_check(cudaSetDevice(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cudaMemcpy(...) \
    _cuda_check(cudaMemcpy(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cudaMalloc(...) \
    _cuda_check(cudaMalloc(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cudaHostAlloc(...) \
    _cuda_check(cudaHostAlloc(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cudaDeviceSynchronize(...) \
    _cuda_check(cudaDeviceSynchronize(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cudaGetDeviceCount(...) \
    _cuda_check(cudaGetDeviceCount(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cudaGetDeviceProperties(...) \
    _cuda_check(cudaGetDeviceProperties(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cudaFree(...) \
    _cuda_check(cudaFree(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cudaFreeHost(...) \
    _cuda_check(cudaFreeHost(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cudaDeviceGetAttribute(...) \
    _cuda_check(cudaDeviceGetAttribute(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cudaGetLastError(...) \
    _cuda_check(cudaGetLastError(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cudaDeviceReset(...) \
    _cuda_check(cudaDeviceReset(__VA_ARGS__), __FILE__, __LINE__)

void _cuda_check(cudaError_t cs, const char *file, long line);

#else
#define safe_cudaSetDevice(...) \
    cudaSetDevice(__VA_ARGS__)
#define safe_cudaMemcpy(...) \
    cudaMemcpy(__VA_ARGS__)
#define safe_cudaMalloc(...) \
    cudaMalloc(__VA_ARGS__)
#define safe_cudaHostAlloc(...) \
    cudaHostAlloc(__VA_ARGS__)
#define safe_cudaDeviceSynchronize(...) \
    cudaDeviceSynchronize(__VA_ARGS__)
#define safe_cudaGetDeviceCount(...) \
    cudaGetDeviceCount(__VA_ARGS__)
#define safe_cudaGetDeviceProperties(...) \
    cudaGetDeviceProperties(__VA_ARGS__)
#define safe_cudaFree(...) \
    cudaFree(__VA_ARGS__)
#define safe_cudaFreeHost(...) \
    cudaFreeHost(__VA_ARGS__)
#define safe_cudaDeviceGetAttribute(...) \
    cudaDeviceGetAttribute(__VA_ARGS__)
#define safe_cudaGetLastError(...) \
    cudaGetLastError(__VA_ARGS__)
#define safe_cudaDeviceReset(...) \
    cudaDeviceReset(__VA_ARGS__)
#endif

#ifdef SAFE_CUBLAS
#define safe_cublasCreate(...) \
    _cublas_check(cublasCreate(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cublasDestroy(...) \
    _cublas_check(cublasDestroy(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cublasSetPointerMode(...) \
    _cublas_check(cublasSetPointerMode(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cublasSetMatrix(...) \
    _cublas_check(cublasSetMatrix(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cublasGetMatrix(...) \
    _cublas_check(cublasGetMatrix(__VA_ARGS__), __FILE__, __LINE__)
#define safe_cublasDgemm(...) \
    _cublas_check(cublasDgemm(__VA_ARGS__), __FILE__, __LINE__)
void _cublas_check(int cs, const char *file, long line);

#else
#define safe_cublasCreate(...) \
    cublasCreate(__VA_ARGS__)
#define safe_cublasDestroy(...) \
    cublasDestroy(__VA_ARGS__)
#define safe_cublasSetPointerMode(...) \
    cublasSetPointerMode(__VA_ARGS__)
#define safe_cublasSetMatrix(...) \
    cublasSetMatrix(__VA_ARGS__)
#define safe_cublasGetMatrix(...) \
    cublasGetMatrix(__VA_ARGS__)
#define safe_cublasDgemm(...) \
    cublasDgemm(__VA_ARGS__)
#endif
