all: gpuprof

USE_NVTX=-DUSE_NVTX
SAFE_CUBLAS=-DSAFE_CUBLAS
SAFE_CUDA=-DSAFE_CUDA

DEBUG=-arch=sm_35 -rdc=true -O3 -Xcompiler -Wall -Xcompiler -Wextra \
	-Xcompiler -fopenmp $(USE_NVTX)
OBJDIR=obj

ATLAS_PATH=/usr/local/atlas3.10.1/lib/
CUDA_PATH=/usr/local/cuda/lib64

LAPACKE=/usr/local/plasma2.5.0/lib/liblapacke.a
ATLAS=-L$(ATLAS_PATH) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm
CUDA=-L$(CUDA_PATH) -lcublas -lcudadevrt -lnvToolsExt

TPB=128

_OBJ= dlaed0_m.o dlaed1.o dlaed1_ph2.o dlaed2.o dlaed3.o dlaed3_ph2.o  \
		dlaed4.o dlaed4_ph2.o dlamrg.o dlapy2.o matio.o main.o workspace.o \
		safety.o timer.o
OBJ=$(patsubst %,$(OBJDIR)/%,$(_OBJ))

gpuprof: $(OBJ)
	nvcc $(DEBUG) -o $@ $^ $(LAPACKE) $(ATLAS) $(CUDA)

$(OBJDIR)/%.o: %.cu
	nvcc $(DEBUG) -c $< -o $@ $(SAFE_CUDA) $(SAFE_CUBLAS) -DTPB=$(TPB)
$(OBJDIR)/matio.o: ../../matio/matio.c
	g++ -O3 -Wall -Wextra -c $< -o $@

clean:
	rm -fv gpuprof $(OBJDIR)/*.o
