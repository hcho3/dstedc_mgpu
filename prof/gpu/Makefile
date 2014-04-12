all: gpuprof

include ../../make.inc

DEBUG=-rdc=true -O3 -Xcompiler -Wall -Xcompiler -Wextra \
	-Xcompiler -fopenmp $(USE_NVTX)
OBJDIR=obj

ATLAS=-L$(ATLAS_PATH) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm
CUDA=-L$(CUDA_PATH) -lcublas -lcudadevrt -lnvToolsExt

_OBJ= dlaed0_m.o dlaed1.o dlaed2.o dlaed3.o \
		dlaed4.o dlamrg.o dlapy2.o matio.o main.o workspace.o \
		safety.o timer.o nvtx.o
OBJ=$(patsubst %,$(OBJDIR)/%,$(_OBJ))

gpuprof: $(OBJ)
	nvcc $(NV_SM) $(DEBUG) -o $@ $^ $(LAPACKE) $(ATLAS) $(CUDA)

$(OBJDIR)/%.o: %.cu
	nvcc $(NV_SM) $(DEBUG) -c $< -o $@ $(SAFE_CUDA) $(SAFE_CUBLAS) -DTPB=$(TPB)
$(OBJDIR)/matio.o: ../../matio/matio.c
	g++ -O3 -Wall -Wextra -c $< -o $@

clean:
	rm -fv gpuprof $(OBJDIR)/*.o
