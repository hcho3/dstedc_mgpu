all: tester

DEBUG=-O3 -Wall -Wextra -pedantic -fopenmp -std=gnu99

ATLAS_PATH=/usr/local/atlas3.10.1/lib/

LAPACKE=/usr/local/plasma2.5.0/lib/liblapacke.a
ATLAS=-L$(ATLAS_PATH) -llapack -lcblas -lf77blas -latlas -lgfortran -lz -lm

TPB=128

tester: dlaed0_m.o dlaed1.o dlaed2.o dlaed3.o dlaed4.o  \
		dlamrg.o dlapy2.o matio.o tester.o workspace.o \
		timer.o initial_guess.o middle_way.o
	gcc $(DEBUG) -o $@ $^ $(LAPACKE) $(ATLAS) -lm

dlaed0_m.o: dlaed0_m.c
	gcc $(DEBUG) -c $< -o $@
dlaed1.o: dlaed1.c
	gcc $(DEBUG) -c $< -o $@
dlaed2.o: dlaed2.c
	gcc $(DEBUG) -c $< -o $@
dlaed3.o: dlaed3.c
	gcc $(DEBUG) -c $< -o $@
dlaed4.o: dlaed4.c
	gcc $(DEBUG) -c $< -o $@
initial_guess.o: initial_guess.c
	gcc $(DEBUG) -c $< -o $@
middle_way.o: middle_way.c
	gcc $(DEBUG) -c $< -o $@
dlamrg.o: dlamrg.c
	gcc $(DEBUG) -c $< -o $@
dlapy2.o: dlapy2.c
	gcc $(DEBUG) -c $< -o $@
matio.o: matio.c
	gcc $(DEBUG) -c $< -o $@
tester.o: tester.c
	gcc $(DEBUG) -c $< -o $@
workspace.o: workspace.c
	gcc $(DEBUG) -c $< -o $@
timer.o: timer.c
	gcc $(DEBUG) -c $< -o $@

clean:
	rm -fv tester *.o
