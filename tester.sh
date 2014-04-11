#!/bin/bash
function dstedc()
{
    set -vx
    for i in 16384 32768 36000 50000
    do
        ./dstedc $1 $2 testmat/dlaed1/D_$i.bin testmat/dlaed1/E_$i.bin Dout.bin Q.bin
    done
}

function magma()
{
    set -vx
    #for i in 1024 2048 4096 8192 16384 32768 36000
    for i in 32768 36000
    do
        magma/magma $1 testmat/dlaed1/D_$i.bin testmat/dlaed1/E_$i.bin Dout.bin Q.bin
    done
}

function omp()
{
    set -vx
    for i in 1024 2048 4096 8192 16384 32768 36000
    do
        prof/cpu/cpuprof $1 $1 testmat/dlaed1/D_$i.bin testmat/dlaed1/E_$i.bin Dout.bin Q.bin
    done
}

dstedc 4 20
dstedc 3 21
dstedc 2 22
dstedc 1 23
#magma 1
#magma 2
#magma 3
#magma 4
#omp 24
