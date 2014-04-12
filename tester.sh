#!/bin/bash
function dstedc()
{
    set -vx
    for i in 16384 32768 36000 50000
    do
        ./dstedc $1 $2 testmat/dlaed1/D_$i.bin testmat/dlaed1/E_$i.bin Dout.bin Q.bin
    done
}

dstedc 4 20
dstedc 3 21
dstedc 2 22
dstedc 1 23
