all: dstedc prof
.PHONY: prof

include make.inc

#USE_NVTX=-DUSE_NVTX
#SAFE_CUBLAS=-DSAFE_CUBLAS
#SAFE_CUDA=-DSAFE_CUDA

DEBUG=-rdc=true -O3 -Xcompiler -Wall -Xcompiler -Wextra \
	-Xcompiler -fopenmp $(USE_NVTX)
OBJDIR=obj

ATLAS_PATH=/usr/local/atlas3.10.1/lib/
CUDA_PATH=/usr/local/cuda/lib64

LAPACKE=/usr/local/plasma2.5.0/lib/liblapacke.a
ATLAS=-L$(ATLAS_PATH) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm
CUDA=-L$(CUDA_PATH) -lcublas -lcudadevrt -lnvToolsExt

NV_SM= -arch=sm_35 -gencode arch=compute_35,code=sm_35 \
	   -gencode arch=compute_35,code=compute_35

_OBJ= dlaed0_m.o dlaed1_gpu.o dlaed1_cpu.o dlaed1_ph2.o \
		dlaed2.o dlaed3_gpu.o dlaed3_cpu.o dlaed3_ph2.o  \
		dlaed4_gpu.o dlaed4_cpu.o initial_guess_cpu.o middle_way_cpu.o \
		dlamrg.o dlapy2.o matio.o main.o workspace.o \
		safety.o timer.o cfg.o nvtx.o
OBJ=$(patsubst %,$(OBJDIR)/%,$(_OBJ))

dstedc: $(OBJ)
	nvcc $(NV_SM) $(DEBUG) -o $@ $^ $(LAPACKE) $(ATLAS) $(CUDA)

$(OBJDIR)/%.o: %.cu
	nvcc $(NV_SM) $(DEBUG) -c $< -o $@ $(SAFE_CUDA) $(SAFE_CUBLAS) -DTPB=$(TPB)
$(OBJDIR)/matio.o: matio/matio.c
	g++ -O3 -Wall -Wextra -c $< -o $@

prof:
	$(MAKE) -C prof/

clean:
	rm -fv dstedc $(OBJDIR)/*.o

cleanall:
	$(MAKE) -C prof/ clean
	rm -fv dstedc $(OBJDIR)/*.o
