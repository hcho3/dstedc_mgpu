all: gpuprof cpuprof

.PHONY: gpuprof cpuprof

gpuprof:
	$(MAKE) -C gpu/

cpuprof:
	$(MAKE) -C cpu/

clean:
	$(MAKE) -C gpu/ clean
	$(MAKE) -C cpu/ clean
