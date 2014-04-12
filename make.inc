# library paths
CUDA_PATH=/usr/local/cuda/lib64
LAPACKE=/usr/local/plasma2.5.0/lib/liblapacke.a
ATLAS_PATH=/usr/local/atlas3.10.1/lib/

# CUDA compilation options
NV_SM= -arch=sm_35 -gencode arch=compute_35,code=sm_35 \
	   -gencode arch=compute_35,code=compute_35

# CUDA parameter: # threads per block
TPB=64

# Flags
#USE_NVTX=-DUSE_NVTX         # emit extra info for NVIDIA visual profiler
#SAFE_CUBLAS=-DSAFE_CUBLAS   # check return values of cuBLAS functions
#SAFE_CUDA=-DSAFE_CUDA       # check return values of CUDA API calls
