all: gpuprof

USE_NVTX=-DUSE_NVTX
SAFE_CUBLAS=-DSAFE_CUBLAS
SAFE_CUDA=-DSAFE_CUDA

DEBUG=-arch=sm_35 -rdc=true -O3 -Xcompiler -Wall -Xcompiler -Wextra \
	-Xcompiler -fopenmp $(USE_NVTX)

ATLAS_PATH=/usr/local/atlas3.10.1/lib/
CUDA_PATH=/usr/local/cuda/lib64

LAPACKE=/usr/local/plasma2.5.0/lib/liblapacke.a
ATLAS=-L$(ATLAS_PATH) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm
CUDA=-L$(CUDA_PATH) -lcublas -lcudadevrt -lnvToolsExt

TPB=128

gpuprof: dlaed0_m.o dlaed1.o dlaed1_ph2.o dlaed2.o dlaed3.o dlaed3_ph2.o  \
		dlaed4.o dlaed4_ph2.o dlamrg.o dlapy2.o matio.o main.o workspace.o \
		safety.o timer.o
	nvcc $(DEBUG) -o $@ $^ $(LAPACKE) $(ATLAS) $(CUDA)
	mv $@ ..

dlaed0_m.o: dlaed0_m.cu
	nvcc $(DEBUG) -c $< -o $@ $(SAFE_CUDA)
dlaed1.o: dlaed1.cu
	nvcc $(DEBUG) -c $< -o $@ $(SAFE_CUBLAS)
dlaed1_ph2.o: dlaed1_ph2.cu
	nvcc $(DEBUG) -c $< -o $@ $(SAFE_CUDA) $(SAFE_CUBLAS)
dlaed2.o: dlaed2.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed3.o: dlaed3.cu
	nvcc $(DEBUG) -c $< -o $@ $(SAFE_CUDA) -DTPB=$(TPB)
dlaed3_ph2.o: dlaed3_ph2.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed4.o: dlaed4.cu
	nvcc $(DEBUG) -c $< -o $@
dlaed4_ph2.o: dlaed4_ph2.cu
	nvcc $(DEBUG) -c $< -o $@
dlamrg.o: dlamrg.cu
	nvcc $(DEBUG) -c $< -o $@
dlapy2.o: dlapy2.cu
	nvcc $(DEBUG) -c $< -o $@
matio.o: matio.cu
	nvcc $(DEBUG) -c $< -o $@
main.o: main.cu
	nvcc $(DEBUG) -c $< -o $@ $(SAFE_CUDA)
workspace.o: workspace.cu
	nvcc $(DEBUG) -c $< -o $@ $(SAFE_CUDA)
safety.o: safety.cu
	nvcc $(DEBUG) -c $< -o $@ $(SAFE_CUBLAS)
timer.o: timer.cu
	nvcc $(DEBUG) -c $< -o $@

clean:
	rm -fv ../gpuprof *.o
