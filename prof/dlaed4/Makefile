all: gpu cpu

include ../../make.inc

DEBUG=-rdc=true -O3 -Xcompiler -Wall -Xcompiler -Wextra \
	-Xcompiler -fopenmp $(USE_NVTX)
OBJDIR=obj

ATLAS=-L$(ATLAS_PATH) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm
CUDA=-L$(CUDA_PATH) -lcublas -lcudadevrt -lnvToolsExt

_OBJ= matio.o safety.o timer.o workspace.o
_OBJ_GPU= dlaed4_gpu.o main_gpu.o
_OBJ_CPU = dlaed4_cpu.o initial_guess_cpu.o middle_way_cpu.o main_cpu.o

OBJ=$(patsubst %,$(OBJDIR)/%,$(_OBJ))
OBJ_GPU=$(patsubst %,$(OBJDIR)/%,$(_OBJ_GPU))
OBJ_CPU=$(patsubst %,$(OBJDIR)/%,$(_OBJ_CPU))

gpu: $(OBJ) $(OBJ_GPU)
	nvcc $(NV_SM) $(DEBUG) -o $@ $^ $(LAPACKE) $(ATLAS) $(CUDA)

cpu: $(OBJ) $(OBJ_CPU)
	nvcc $(NV_SM) $(DEBUG) -o $@ $^ $(LAPACKE) $(ATLAS) $(CUDA)

$(OBJDIR)/%.o: %.cu
	nvcc $(NV_SM) $(DEBUG) -c $< -o $@ $(SAFE_CUDA) $(SAFE_CUBLAS) -DTPB=$(TPB)
$(OBJDIR)/matio.o: matio.c
	g++ -O3 -Wall -Wextra -c $< -o $@

clean:
	rm -fv gpu cpu $(OBJDIR)/*.o
